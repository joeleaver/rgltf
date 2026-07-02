//! rgltf-render — a from-scratch wgpu PBR renderer for the glTF viewer.
//!
//! Phase 2 renders a loaded [`rgltf_asset::Scene`] with metallic-roughness PBR:
//! per-material bind groups (base-color / metallic-roughness / normal / occlusion
//! / emissive textures + factors), a directional key light, and a hemisphere
//! ambient term standing in for IBL. Output is tone-mapped + sRGB-encoded into an
//! offscreen texture that rinch composites zero-copy.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rgltf_asset::{AlphaMode, ImageData, Material, TexFormat, TexRef};
use wgpu::util::DeviceExt;

mod camera;
pub use camera::Camera;
pub use rgltf_asset::Scene;

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient_sky: [f32; 4],
    ambient_ground: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniform {
    base_color_factor: [f32; 4],
    emissive: [f32; 4], // rgb + strength
    params: [f32; 4],   // metallic, roughness, normal_scale, occlusion_strength
    flags: [f32; 4],    // has: base, mr, normal, occlusion
    flags2: [f32; 4],   // has_emissive, alpha_cutoff, alpha_mode, unlit
}

struct GpuMaterial {
    #[allow(dead_code)]
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct GpuPrimitive {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    material: usize,
}

/// Everything uploaded for the current scene (dropped/replaced on `set_scene`).
#[derive(Default)]
struct GpuScene {
    #[allow(dead_code)]
    textures: Vec<wgpu::Texture>,
    texture_views: Vec<wgpu::TextureView>,
    materials: Vec<GpuMaterial>,
    primitives: Vec<GpuPrimitive>,
}

struct Targets {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl Targets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgltf-color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgltf-depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        Self { color, color_view, depth_view, width, height }
    }
}

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    material_bgl: wgpu::BindGroupLayout,
    frame_uniform: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    dummy_view: wgpu::TextureView,
    scene: GpuScene,
    targets: Targets,
    pub clear: wgpu::Color,
    // Lighting (world space).
    pub light_dir: [f32; 3],
    pub light_color: [f32; 3],
    pub ambient_sky: [f32; 3],
    pub ambient_ground: [f32; 3],
}

impl Renderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rgltf-pbr"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pbr.wgsl").into()),
        });

        // ── Bind group layouts ──
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgltf-frame-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgltf-material-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                tex_entry(4),
                tex_entry(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let frame_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgltf-frame-uniform"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgltf-frame-bg"),
            layout: &frame_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: frame_uniform.as_entire_binding() }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rgltf-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 1x1 white dummy for unused texture slots (guarded by material flags).
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgltf-dummy"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &dummy,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let dummy_view = dummy.create_view(&Default::default());

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgltf-pl"),
            bind_group_layouts: &[&frame_bgl, &material_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rgltf-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<rgltf_asset::Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Two-sided: the shader flips the normal for back faces.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        let targets = Targets::new(&device, width, height);

        Self {
            device,
            queue,
            pipeline,
            material_bgl,
            frame_uniform,
            frame_bind_group,
            sampler,
            dummy_view,
            scene: GpuScene::default(),
            targets,
            clear: wgpu::Color { r: 0.055, g: 0.065, b: 0.085, a: 1.0 },
            light_dir: [0.5, 0.8, 0.6],
            light_color: [3.0, 3.0, 2.95],
            ambient_sky: [0.42, 0.47, 0.55],
            ambient_ground: [0.20, 0.18, 0.16],
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.targets.width, self.targets.height)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if width == self.targets.width && height == self.targets.height {
            return;
        }
        self.targets = Targets::new(&self.device, width, height);
    }

    fn upload_image(&self, img: &ImageData) -> wgpu::Texture {
        let format = wgpu_format(img.format, img.srgb);
        let (w, h) = (img.width.max(1), img.height.max(1));
        let mip_count = img.mips.len().max(1) as u32;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgltf-image"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let block = img.format.block_bytes();
        // queue.write_texture allows arbitrary bytes_per_row (no 256 alignment). For
        // compressed formats a "row" is a row of 4×4 blocks, and the copy extent must
        // be block-aligned — round up so the tail mips (2×2, 1×1) fill a full block.
        for (level, mip) in img.mips.iter().enumerate() {
            let (mw, mh) = (mip.width.max(1), mip.height.max(1));
            let (bytes_per_row, rows, copy_w, copy_h) = if img.format.is_compressed() {
                let (bx, by) = (mw.div_ceil(4), mh.div_ceil(4));
                (bx * block, by, bx * 4, by * 4)
            } else {
                (mw * block, mh, mw, mh)
            };
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &mip.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d { width: copy_w, height: copy_h, depth_or_array_layers: 1 },
            );
        }
        tex
    }

    fn material_uniform(m: &Material) -> MaterialUniform {
        let f = |b: bool| if b { 1.0 } else { 0.0 };
        MaterialUniform {
            base_color_factor: m.base_color_factor,
            emissive: [m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2], m.emissive_strength],
            params: [m.metallic_factor, m.roughness_factor, m.normal_scale, m.occlusion_strength],
            flags: [
                f(m.base_color_tex.is_some()),
                f(m.metallic_roughness_tex.is_some()),
                f(m.normal_tex.is_some()),
                f(m.occlusion_tex.is_some()),
            ],
            flags2: [
                f(m.emissive_tex.is_some()),
                m.alpha_cutoff,
                match m.alpha_mode {
                    AlphaMode::Opaque => 0.0,
                    AlphaMode::Mask => 1.0,
                    AlphaMode::Blend => 2.0,
                },
                f(m.unlit),
            ],
        }
    }

    /// Upload a scene: textures, per-material bind groups, per-primitive buffers.
    pub fn set_scene(&mut self, scene: &Scene) {
        let mut gpu = GpuScene::default();

        for img in &scene.images {
            let tex = self.upload_image(img);
            gpu.texture_views.push(tex.create_view(&Default::default()));
            gpu.textures.push(tex);
        }

        let view_for = |t: &Option<TexRef>| -> &wgpu::TextureView {
            match t {
                Some(r) => gpu.texture_views.get(r.image).unwrap_or(&self.dummy_view),
                None => &self.dummy_view,
            }
        };

        for m in &scene.materials {
            let uniform = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rgltf-material-uniform"),
                contents: bytemuck::bytes_of(&Self::material_uniform(m)),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rgltf-material-bg"),
                layout: &self.material_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view_for(&m.base_color_tex)) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(view_for(&m.metallic_roughness_tex)) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(view_for(&m.normal_tex)) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(view_for(&m.occlusion_tex)) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(view_for(&m.emissive_tex)) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            gpu.materials.push(GpuMaterial { uniform, bind_group });
        }

        for prim in &scene.primitives {
            if prim.vertices.is_empty() || prim.indices.is_empty() {
                continue;
            }
            let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rgltf-prim-vbuf"),
                contents: bytemuck::cast_slice(&prim.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rgltf-prim-ibuf"),
                contents: bytemuck::cast_slice(&prim.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            gpu.primitives.push(GpuPrimitive {
                vbuf,
                ibuf,
                index_count: prim.indices.len() as u32,
                material: prim.material.min(gpu.materials.len().saturating_sub(1)),
            });
        }

        self.scene = gpu;
    }

    pub fn render(&mut self, camera: &Camera) {
        let eye = camera.eye();
        let frame = FrameUniform {
            view_proj: camera.view_proj().to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [self.light_dir[0], self.light_dir[1], self.light_dir[2], 0.0],
            light_color: [self.light_color[0], self.light_color[1], self.light_color[2], 1.0],
            ambient_sky: [self.ambient_sky[0], self.ambient_sky[1], self.ambient_sky[2], 1.0],
            ambient_ground: [self.ambient_ground[0], self.ambient_ground[1], self.ambient_ground[2], 1.0],
        };
        self.queue.write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(&frame));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rgltf-enc") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rgltf-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(self.clear), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if !self.scene.primitives.is_empty() && !self.scene.materials.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.frame_bind_group, &[]);
                for prim in &self.scene.primitives {
                    let mat = &self.scene.materials[prim.material];
                    pass.set_bind_group(1, &mat.bind_group, &[]);
                    pass.set_vertex_buffer(0, prim.vbuf.slice(..));
                    pass.set_index_buffer(prim.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..prim.index_count, 0, 0..1);
                }
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn color_target(&self) -> (wgpu::Texture, wgpu::TextureView) {
        (self.targets.color.clone(), self.targets.color_view.clone())
    }
}

/// Map an asset [`TexFormat`] + sRGB flag to the wgpu texture format. BC4/BC5 are
/// always linear (single/two-channel data), so they ignore `srgb`.
fn wgpu_format(format: TexFormat, srgb: bool) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as W;
    match (format, srgb) {
        (TexFormat::Rgba8, false) => W::Rgba8Unorm,
        (TexFormat::Rgba8, true) => W::Rgba8UnormSrgb,
        (TexFormat::Bc1Rgba, false) => W::Bc1RgbaUnorm,
        (TexFormat::Bc1Rgba, true) => W::Bc1RgbaUnormSrgb,
        (TexFormat::Bc3Rgba, false) => W::Bc3RgbaUnorm,
        (TexFormat::Bc3Rgba, true) => W::Bc3RgbaUnormSrgb,
        (TexFormat::Bc4R, _) => W::Bc4RUnorm,
        (TexFormat::Bc5Rg, _) => W::Bc5RgUnorm,
        (TexFormat::Bc7Rgba, false) => W::Bc7RgbaUnorm,
        (TexFormat::Bc7Rgba, true) => W::Bc7RgbaUnormSrgb,
    }
}

/// The center + radius of a scene's bounds as glam types.
pub fn scene_sphere(scene: &Scene) -> (Vec3, f32) {
    (scene.bounds.center(), scene.bounds.radius())
}
