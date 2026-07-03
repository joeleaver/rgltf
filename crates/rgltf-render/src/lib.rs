//! rgltf-render — a from-scratch wgpu PBR renderer for the glTF viewer.
//!
//! Renders a loaded [`rgltf_asset::Scene`] with metallic-roughness PBR: per-material
//! bind groups (base-color / metallic-roughness / normal / occlusion / emissive
//! textures + factors), a directional key light, and split-sum image-based lighting
//! (irradiance + prefiltered specular + BRDF LUT baked from a procedural environment;
//! see [`ibl`]) with an analytic skybox. Output is tone-mapped + sRGB-encoded into an
//! offscreen texture that rinch composites zero-copy.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use rgltf_asset::{AlphaMode, ImageData, Material, MorphTarget, TexFormat, TexRef, Vertex};
use wgpu::util::DeviceExt;

mod camera;
mod ibl;
pub use camera::Camera;
pub use ibl::EnvParams;
pub use rgltf_asset::Scene;

use ibl::{Ibl, IblGen, PREFILTER_MIPS};

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    env_sky: [f32; 4],
    env_horizon: [f32; 4],
    env_ground: [f32; 4],
    flags: [f32; 4], // x = use textures, y = prefilter mip count, z = IBL intensity
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniform {
    base_color_factor: [f32; 4],
    emissive: [f32; 4], // rgb + strength
    params: [f32; 4],   // metallic, roughness, normal_scale, occlusion_strength
    flags: [f32; 4],    // has: base, mr, normal, occlusion
    flags2: [f32; 4],   // has_emissive, alpha_cutoff, alpha_mode, unlit
    // Material extensions:
    specular: [f32; 4],       // specular_factor, ior, _, _
    specular_color: [f32; 4], // specular_color_factor rgb, _
    clearcoat: [f32; 4],      // factor, roughness, normal_scale, _
    sheen: [f32; 4],          // sheen_color_factor rgb, w = sheen_roughness
    ext_flags: [f32; 4],      // has: specular, specular_color, clearcoat, clearcoat_roughness
    ext_flags2: [f32; 4],     // has: clearcoat_normal, sheen_color, sheen_roughness, _
}

struct GpuMaterial {
    #[allow(dead_code)]
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// CPU-side morph data for a primitive: the base (un-morphed) vertices + the per-target
/// deltas. Present only for morph primitives; drives the per-frame vertex re-upload.
struct MorphCpu {
    base: Vec<Vertex>,
    targets: Vec<MorphTarget>,
}

/// Per-mesh-primitive GPU geometry (local space). Uploaded once, unless `morph` is set —
/// then `vbuf` is `COPY_DST` and rewritten each frame from the base + weighted deltas.
struct GpuPrimitive {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    material: usize,
    morph: Option<MorphCpu>,
}

/// One draw: a primitive instantiated by a scene node, with a slice into the shared
/// instance buffer (`instance_count` = 1 for now; >1 once `EXT_mesh_gpu_instancing`
/// lands). `node` indexes the per-frame world-matrix array passed to [`Renderer::render`].
/// `skin` = index into [`GpuScene::skins`] when the node instantiates a skinned mesh.
struct Draw {
    prim: usize,
    node: usize,
    skin: Option<usize>,
    instance_base: u32,
    instance_count: u32,
}

/// Per-instance data fed to the vertex shader: the node's model + normal matrix (used
/// for un-skinned draws), plus a skin selector (`skin[0]` = skinned flag, `skin[1]` =
/// base offset into the joint palette). Skinned draws ignore `model`/`normal` — their
/// world transform comes entirely from the joint matrices.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],  // columns → shader locations 4..7
    normal: [[f32; 4]; 3], // normal-matrix columns (xyz) → locations 8..10
    skin: [u32; 4],        // [skinned, palette_base, _, _] → location 13
}

impl InstanceRaw {
    fn new(model: Mat4, skin: Option<u32>) -> Self {
        let n = Mat3::from_mat4(model).inverse().transpose().to_cols_array();
        InstanceRaw {
            model: model.to_cols_array_2d(),
            normal: [
                [n[0], n[1], n[2], 0.0],
                [n[3], n[4], n[5], 0.0],
                [n[6], n[7], n[8], 0.0],
            ],
            skin: match skin {
                Some(base) => [1, base, 0, 0],
                None => [0, 0, 0, 0],
            },
        }
    }
}

/// A skin's palette layout: its joint node indices + inverse bind matrices, and the
/// `base` offset of its slice in the shared joint-matrix storage buffer.
struct GpuSkin {
    joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
    base: u32,
}

/// Everything uploaded for the current scene (dropped/replaced on `set_scene`).
#[derive(Default)]
struct GpuScene {
    #[allow(dead_code)]
    textures: Vec<wgpu::Texture>,
    texture_views: Vec<wgpu::TextureView>,
    materials: Vec<GpuMaterial>,
    prims: Vec<GpuPrimitive>,
    draws: Vec<Draw>,
    instance_buffer: Option<wgpu::Buffer>,
    instance_count: u32,
    /// Per-instance-slot *local* transform (relative to the draw's node), aligned with
    /// the instance buffer. `EXT_mesh_gpu_instancing` fills this; un-instanced draws get
    /// a single identity. Final model = `node_world[draw.node] · instances_local[slot]`.
    instances_local: Vec<Mat4>,
    skins: Vec<GpuSkin>,
    /// Concatenated per-skin joint palettes (`joint_count` matrices), rewritten each
    /// frame from the animated node world matrices. Min 1 element so the bind group is
    /// always valid, even for un-skinned scenes.
    joint_buffer: Option<wgpu::Buffer>,
    joint_bind_group: Option<wgpu::BindGroup>,
    joint_count: u32,
    /// Skeleton bone segments as `[parent_node, child_node]` pairs (joints whose parent
    /// is also a joint). Positions are read from `node_world` each frame into `bone_buffer`.
    bone_lines: Vec<[usize; 2]>,
    bone_buffer: Option<wgpu::Buffer>,
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
    /// Line-mode variant of `pipeline` (needs the `POLYGON_MODE_LINE` device feature).
    pipeline_wire: wgpu::RenderPipeline,
    /// Analytic environment backdrop (drawn first, depth writes off).
    skybox_pipeline: wgpu::RenderPipeline,
    /// Skeleton line overlay (drawn last, depth test disabled).
    skeleton_pipeline: wgpu::RenderPipeline,
    material_bgl: wgpu::BindGroupLayout,
    joint_bgl: wgpu::BindGroupLayout,
    frame_uniform: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    dummy_view: wgpu::TextureView,
    scene: GpuScene,
    targets: Targets,
    /// IBL machinery (persistent) + the baked environment (rebuilt on env change).
    ibl_gen: IblGen,
    ibl: Ibl,
    env: EnvParams,
    pub clear: wgpu::Color,
    // Lighting (world space).
    pub light_dir: [f32; 3],
    pub light_color: [f32; 3],
    /// Scales the IBL ambient/reflection contribution.
    pub ibl_intensity: f32,
    /// Draw the analytic environment behind the model.
    pub skybox: bool,
    /// Sample material textures (false = show material factors only).
    pub textured: bool,
    /// Draw triangle edges instead of filled faces.
    pub wireframe: bool,
    /// Colour vertices by their bone-weight blend instead of shading them.
    pub debug_weights: bool,
    /// Overlay the skeleton (bone segments) on top of the scene.
    pub show_skeleton: bool,
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
                // Material extensions: specular(7), specular_color(8), clearcoat(9),
                // clearcoat_roughness(10), clearcoat_normal(11), sheen_color(12),
                // sheen_roughness(13). (12 sampled textures total, under the 16 limit.)
                tex_entry(7),
                tex_entry(8),
                tex_entry(9),
                tex_entry(10),
                tex_entry(11),
                tex_entry(12),
                tex_entry(13),
            ],
        });

        // Joint-matrix palette: a read-only storage buffer indexed in the vertex shader.
        let joint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgltf-joint-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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

        // IBL generator + group-3 bind-group layout (irradiance/prefilter/BRDF-LUT).
        let ibl_gen = IblGen::new(device.clone(), queue.clone());

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgltf-pl"),
            bind_group_layouts: &[&frame_bgl, &material_bgl, &joint_bgl, ibl_gen.bind_group_layout()],
            push_constant_ranges: &[],
        });

        // Vertex buffer 0: per-vertex geometry + skin influences (locations 0..3, 11, 12).
        // Buffer 1: per-instance model+normal matrices + skin selector (4..10, 13).
        let vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2,
            11 => Uint16x4, 12 => Float32x4
        ];
        let instance_attrs = wgpu::vertex_attr_array![
            4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4,
            8 => Float32x4, 9 => Float32x4, 10 => Float32x4,
            13 => Uint32x4
        ];
        let vbuffers = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<rgltf_asset::Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attrs,
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attrs,
            },
        ];
        // Fill + line (wireframe) variants of the same pipeline; the app picks per frame.
        let make_pipeline = |polygon_mode: wgpu::PolygonMode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rgltf-pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &vbuffers,
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
                    polygon_mode,
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
            })
        };
        let pipeline = make_pipeline(wgpu::PolygonMode::Fill);
        let pipeline_wire = make_pipeline(wgpu::PolygonMode::Line);

        // Skybox: a fullscreen triangle sharing the frame uniform (group 0 only). No
        // depth writes and always-pass so it fills the background and meshes overwrite it.
        let skybox_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgltf-skybox-pl"),
            bind_group_layouts: &[&frame_bgl],
            push_constant_ranges: &[],
        });
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rgltf-skybox"),
            layout: Some(&skybox_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // Skeleton overlay: world-space line segments (group 0 only), depth test off so
        // the whole skeleton shows through the mesh.
        let skeleton_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgltf-skeleton-pl"),
            bind_group_layouts: &[&frame_bgl],
            push_constant_ranges: &[],
        });
        let bone_attrs = wgpu::vertex_attr_array![0 => Float32x3];
        let skeleton_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rgltf-skeleton"),
            layout: Some(&skeleton_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_bone"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &bone_attrs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_bone"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // Default environment (Studio-ish); the app overrides it from the lighting preset.
        let env = EnvParams {
            sky: [0.42, 0.47, 0.55],
            horizon: [0.30, 0.30, 0.32],
            ground: [0.20, 0.18, 0.16],
        };
        let ibl = ibl_gen.build(env);

        let targets = Targets::new(&device, width, height);

        Self {
            device,
            queue,
            pipeline,
            pipeline_wire,
            skybox_pipeline,
            skeleton_pipeline,
            material_bgl,
            joint_bgl,
            frame_uniform,
            frame_bind_group,
            sampler,
            dummy_view,
            scene: GpuScene::default(),
            targets,
            ibl_gen,
            ibl,
            env,
            clear: wgpu::Color { r: 0.055, g: 0.065, b: 0.085, a: 1.0 },
            light_dir: [0.5, 0.8, 0.6],
            light_color: [3.0, 3.0, 2.95],
            ibl_intensity: 1.0,
            skybox: true,
            textured: true,
            wireframe: false,
            debug_weights: false,
            show_skeleton: false,
        }
    }

    /// Set the procedural environment (linear radiance). Rebuilds the IBL cubes only
    /// when the parameters change — cheap no-op otherwise, so it's safe to call every
    /// frame. The skybox reads the environment straight from the frame uniform.
    pub fn set_environment(&mut self, env: EnvParams) {
        if env != self.env {
            self.env = env;
            self.ibl = self.ibl_gen.build(env);
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
            specular: [m.specular_factor, m.ior, 0.0, 0.0],
            specular_color: [m.specular_color_factor[0], m.specular_color_factor[1], m.specular_color_factor[2], 0.0],
            clearcoat: [m.clearcoat_factor, m.clearcoat_roughness_factor, m.clearcoat_normal_scale, 0.0],
            sheen: [m.sheen_color_factor[0], m.sheen_color_factor[1], m.sheen_color_factor[2], m.sheen_roughness_factor],
            ext_flags: [
                f(m.specular_tex.is_some()),
                f(m.specular_color_tex.is_some()),
                f(m.clearcoat_tex.is_some()),
                f(m.clearcoat_roughness_tex.is_some()),
            ],
            ext_flags2: [
                f(m.clearcoat_normal_tex.is_some()),
                f(m.sheen_color_tex.is_some()),
                f(m.sheen_roughness_tex.is_some()),
                0.0,
            ],
        }
    }

    /// Upload a scene: textures, per-material bind groups, per-mesh-primitive geometry,
    /// and a draw list (one draw per node-instantiated primitive). Per-draw model
    /// matrices are written into a shared instance buffer each frame in [`Self::render`].
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
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(view_for(&m.specular_tex)) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(view_for(&m.specular_color_tex)) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(view_for(&m.clearcoat_tex)) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(view_for(&m.clearcoat_roughness_tex)) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(view_for(&m.clearcoat_normal_tex)) },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(view_for(&m.sheen_color_tex)) },
                    wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(view_for(&m.sheen_roughness_tex)) },
                ],
            });
            gpu.materials.push(GpuMaterial { uniform, bind_group });
        }

        // Upload each mesh's primitives once; record their global prim indices.
        let mut mesh_prims: Vec<Vec<usize>> = Vec::with_capacity(scene.meshes.len());
        for mesh in &scene.meshes {
            let mut ids = Vec::new();
            for prim in &mesh.primitives {
                if prim.vertices.is_empty() || prim.indices.is_empty() {
                    continue;
                }
                // Morph primitives need a rewritable vertex buffer (re-uploaded per frame).
                let morphed = !prim.morph_targets.is_empty();
                let vusage = if morphed {
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
                } else {
                    wgpu::BufferUsages::VERTEX
                };
                let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rgltf-prim-vbuf"),
                    contents: bytemuck::cast_slice(&prim.vertices),
                    usage: vusage,
                });
                let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rgltf-prim-ibuf"),
                    contents: bytemuck::cast_slice(&prim.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                ids.push(gpu.prims.len());
                gpu.prims.push(GpuPrimitive {
                    vbuf,
                    ibuf,
                    index_count: prim.indices.len() as u32,
                    material: prim.material.min(gpu.materials.len().saturating_sub(1)),
                    morph: morphed.then(|| MorphCpu {
                        base: prim.vertices.clone(),
                        targets: prim.morph_targets.clone(),
                    }),
                });
            }
            mesh_prims.push(ids);
        }

        // Skins: lay each skin's joint palette out contiguously in one storage buffer.
        let mut base = 0u32;
        for skin in &scene.skins {
            let count = skin.joints.len() as u32;
            gpu.skins.push(GpuSkin {
                joints: skin.joints.clone(),
                inverse_bind: skin.inverse_bind.clone(),
                base,
            });
            base += count;
        }
        gpu.joint_count = base;

        // Draw list: one draw per (node, primitive). Each draw reserves `instance_count`
        // consecutive instance slots, whose local transforms go into `instances_local`.
        // A node with `EXT_mesh_gpu_instancing` draws its instances (model-driven, one per
        // instance); else a valid skin draws skinned (palette-driven); else a single draw.
        let mut instance_base = 0u32;
        for (node_idx, node) in scene.nodes.iter().enumerate() {
            let Some(mi) = node.mesh else { continue };
            let Some(ids) = mesh_prims.get(mi) else { continue };
            // Instancing takes precedence over skinning (they aren't combined).
            let (locals, skin): (&[Mat4], Option<usize>) = if !node.instances.is_empty() {
                (&node.instances, None)
            } else {
                (&[Mat4::IDENTITY], node.skin.filter(|&s| s < gpu.skins.len()))
            };
            for &prim in ids {
                let count = locals.len() as u32;
                gpu.instances_local.extend_from_slice(locals);
                gpu.draws.push(Draw { prim, node: node_idx, skin, instance_base, instance_count: count });
                instance_base += count;
            }
        }
        gpu.instance_count = instance_base;

        if instance_base > 0 {
            gpu.instance_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rgltf-instances"),
                size: instance_base as u64 * std::mem::size_of::<InstanceRaw>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Joint-matrix palette storage buffer (min 1 element so the bind group is always
        // valid). Filled per-frame in `render` from the animated node world matrices.
        let joint_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgltf-joints"),
            size: gpu.joint_count.max(1) as u64 * std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.joint_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgltf-joint-bg"),
            layout: &self.joint_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: joint_buffer.as_entire_binding() }],
        }));
        gpu.joint_buffer = Some(joint_buffer);

        // Skeleton segments: one line per joint whose parent is also a joint of the same
        // skin. Positions are written per-frame from the animated node world matrices.
        for skin in &scene.skins {
            for &j in &skin.joints {
                if let Some(p) = scene.nodes.get(j).and_then(|n| n.parent) {
                    if skin.joints.contains(&p) {
                        gpu.bone_lines.push([p, j]);
                    }
                }
            }
        }
        if !gpu.bone_lines.is_empty() {
            gpu.bone_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rgltf-bones"),
                size: (gpu.bone_lines.len() * 2 * std::mem::size_of::<[f32; 3]>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        self.scene = gpu;
    }

    /// Render one frame. `node_world` is the world matrix per scene node (from
    /// `Scene::node_world_matrices` at the current time); `morph_weights` is the active
    /// morph-weight vector per node (from `Scene::morph_weights`), empty for un-morphed
    /// nodes.
    pub fn render(&mut self, camera: &Camera, node_world: &[Mat4], morph_weights: &[Vec<f32>]) {
        let eye = camera.eye();
        let view_proj = camera.view_proj();
        let frame = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [self.light_dir[0], self.light_dir[1], self.light_dir[2], 0.0],
            light_color: [self.light_color[0], self.light_color[1], self.light_color[2], 1.0],
            env_sky: [self.env.sky[0], self.env.sky[1], self.env.sky[2], 1.0],
            env_horizon: [self.env.horizon[0], self.env.horizon[1], self.env.horizon[2], 1.0],
            env_ground: [self.env.ground[0], self.env.ground[1], self.env.ground[2], 1.0],
            flags: [
                if self.textured { 1.0 } else { 0.0 },
                PREFILTER_MIPS as f32,
                self.ibl_intensity,
                if self.debug_weights { 1.0 } else { 0.0 },
            ],
        };
        self.queue.write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(&frame));

        // Morph targets: rewrite each morph primitive's vertex buffer from base + Σ
        // weight·delta (position/normal/tangent), using the weights of the node that
        // instantiates it. Normals/tangents are renormalized in the shader. (A morph mesh
        // shared by several nodes with different weights uses the last draw's weights.)
        for draw in &self.scene.draws {
            let prim = &self.scene.prims[draw.prim];
            let Some(morph) = &prim.morph else { continue };
            let mut verts = morph.base.clone();
            if let Some(w) = morph_weights.get(draw.node) {
                for (t, tgt) in morph.targets.iter().enumerate() {
                    let wt = w.get(t).copied().unwrap_or(0.0);
                    if wt == 0.0 {
                        continue;
                    }
                    for (i, v) in verts.iter_mut().enumerate() {
                        if let Some(d) = tgt.positions.get(i) {
                            for k in 0..3 {
                                v.pos[k] += wt * d[k];
                            }
                        }
                        if let Some(d) = tgt.normals.get(i) {
                            for k in 0..3 {
                                v.normal[k] += wt * d[k];
                            }
                        }
                        if let Some(d) = tgt.tangents.get(i) {
                            for k in 0..3 {
                                v.tangent[k] += wt * d[k];
                            }
                        }
                    }
                }
            }
            self.queue.write_buffer(&prim.vbuf, 0, bytemuck::cast_slice(&verts));
        }

        // Joint-matrix palette: for each skin, `world[joint] · inverse_bind[joint]`,
        // laid out at the skin's `base` offset. Rewritten every frame (poses animate).
        if let Some(joint_buf) = &self.scene.joint_buffer {
            if self.scene.joint_count > 0 {
                let mut palette = vec![Mat4::IDENTITY; self.scene.joint_count as usize];
                for skin in &self.scene.skins {
                    for (j, &node) in skin.joints.iter().enumerate() {
                        let ibm = skin.inverse_bind.get(j).copied().unwrap_or(Mat4::IDENTITY);
                        let world = node_world.get(node).copied().unwrap_or(Mat4::IDENTITY);
                        palette[skin.base as usize + j] = world * ibm;
                    }
                }
                let raw: Vec<[[f32; 4]; 4]> = palette.iter().map(Mat4::to_cols_array_2d).collect();
                self.queue.write_buffer(joint_buf, 0, bytemuck::cast_slice(&raw));
            }
        }

        // Per-instance model + normal matrices (un-skinned) + skin selector → instance
        // buffer. Each instance slot's world model = node_world · its local transform
        // (identity for un-instanced draws; the EXT_mesh_gpu_instancing TRS otherwise).
        if let Some(inst_buf) = &self.scene.instance_buffer {
            let mut instances: Vec<InstanceRaw> =
                Vec::with_capacity(self.scene.instance_count as usize);
            for draw in &self.scene.draws {
                let node_mat = node_world.get(draw.node).copied().unwrap_or(Mat4::IDENTITY);
                let skin_base = draw.skin.map(|s| self.scene.skins[s].base);
                for slot in draw.instance_base..draw.instance_base + draw.instance_count {
                    let local = self.scene.instances_local.get(slot as usize).copied().unwrap_or(Mat4::IDENTITY);
                    instances.push(InstanceRaw::new(node_mat * local, skin_base));
                }
            }
            self.queue.write_buffer(inst_buf, 0, bytemuck::cast_slice(&instances));
        }

        // Skeleton: current world-space endpoints of each bone segment.
        if let Some(bone_buf) = &self.scene.bone_buffer {
            let mut verts: Vec<[f32; 3]> = Vec::with_capacity(self.scene.bone_lines.len() * 2);
            for &[p, c] in &self.scene.bone_lines {
                let pp = node_world.get(p).map_or(Vec3::ZERO, |m| m.w_axis.truncate());
                let cc = node_world.get(c).map_or(Vec3::ZERO, |m| m.w_axis.truncate());
                verts.push(pp.to_array());
                verts.push(cc.to_array());
            }
            self.queue.write_buffer(bone_buf, 0, bytemuck::cast_slice(&verts));
        }

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
            // Environment backdrop (behind everything; depth writes off).
            if self.skybox {
                pass.set_pipeline(&self.skybox_pipeline);
                pass.set_bind_group(0, &self.frame_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            if let (Some(inst_buf), Some(joint_bg)) =
                (&self.scene.instance_buffer, &self.scene.joint_bind_group)
            {
                if !self.scene.draws.is_empty() && !self.scene.materials.is_empty() {
                    let stride = std::mem::size_of::<InstanceRaw>() as u64;
                    let pipeline = if self.wireframe { &self.pipeline_wire } else { &self.pipeline };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &self.frame_bind_group, &[]);
                    pass.set_bind_group(2, joint_bg, &[]);
                    pass.set_bind_group(3, &self.ibl.bind_group, &[]);
                    for draw in &self.scene.draws {
                        let prim = &self.scene.prims[draw.prim];
                        let mat = &self.scene.materials[prim.material];
                        pass.set_bind_group(1, &mat.bind_group, &[]);
                        pass.set_vertex_buffer(0, prim.vbuf.slice(..));
                        let base = draw.instance_base as u64 * stride;
                        let end = base + draw.instance_count as u64 * stride;
                        pass.set_vertex_buffer(1, inst_buf.slice(base..end));
                        pass.set_index_buffer(prim.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..prim.index_count, 0, 0..draw.instance_count);
                    }
                }
            }

            // Skeleton overlay (depth test off → drawn on top of the mesh).
            if self.show_skeleton {
                if let Some(bone_buf) = &self.scene.bone_buffer {
                    pass.set_pipeline(&self.skeleton_pipeline);
                    pass.set_bind_group(0, &self.frame_bind_group, &[]);
                    pass.set_vertex_buffer(0, bone_buf.slice(..));
                    pass.draw(0..(self.scene.bone_lines.len() as u32 * 2), 0..1);
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
