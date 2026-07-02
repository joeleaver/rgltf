//! Image-based lighting precompute (split-sum). From a procedural environment we
//! bake three things a PBR fragment shader samples: a diffuse **irradiance** cube,
//! a roughness-mipped **prefiltered specular** cube, and the environment-BRDF
//! **LUT** (which is environment-independent, so it's computed once).
//!
//! The environment itself is an analytic sky gradient (see `ibl.wgsl` `sky_radiance`)
//! parameterised by [`EnvParams`]; regenerated only when the lighting preset changes.
//! Kept deliberately simple so a loadable-HDR environment can slot in later by
//! swapping the `fs_sky` face pass for an equirect→cube sample.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

const ENV_SIZE: u32 = 128;
const IRR_SIZE: u32 = 32;
const PRE_SIZE: u32 = 128;
const LUT_SIZE: u32 = 256;
/// Mip count of the prefiltered specular cube (roughness = mip / (PREFILTER_MIPS-1)).
pub const PREFILTER_MIPS: u32 = 6;

const CUBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// The six cube faces as `(forward, right, up)` bases. A face texel at clip-space
/// `uv ∈ [-1,1]` reconstructs the direction `fwd + uv.x*right + uv.y*up`; these
/// bases are chosen so that direction equals what the hardware cube lookup samples
/// for that face/texel (D3D/Vulkan/WebGPU convention), making the baked gradient
/// seamless across face edges. Layer order: +X,-X,+Y,-Y,+Z,-Z.
const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),   // +X
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),   // -X
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),   // +Y
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),   // -Y
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),    // +Z
    ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  // -Z
];

/// Procedural environment description (linear-space radiance). `horizon` is the band
/// at the equator; `sky`/`ground` are the zenith/nadir the horizon blends toward.
#[derive(Clone, Copy, PartialEq)]
pub struct EnvParams {
    pub sky: [f32; 3],
    pub horizon: [f32; 3],
    pub ground: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GenRaw {
    fwd: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    params: [f32; 4], // x = roughness
    sky: [f32; 4],
    horizon: [f32; 4],
    ground: [f32; 4],
}

/// Baked IBL for one environment: the group-3 bind group plus the textures it views
/// (kept alive here). The source env cube is dropped after baking — nothing samples
/// it at render time (the skybox is analytic).
pub struct Ibl {
    pub bind_group: wgpu::BindGroup,
    _irradiance: wgpu::Texture,
    _prefilter: wgpu::Texture,
}

/// Persistent IBL machinery: the precompute pipelines, the shared samplers, the
/// env-independent BRDF LUT, and the group-3 bind-group layout the mesh pipeline uses.
/// [`Self::build`] bakes an [`Ibl`] for a given environment.
pub struct IblGen {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// group 3 (mesh): irradiance cube, prefilter cube, brdf lut, cube samp, lut samp.
    bgl: wgpu::BindGroupLayout,

    u_buffer: wgpu::Buffer,
    u_bind_group: wgpu::BindGroup,
    src_bgl: wgpu::BindGroupLayout,

    sky_pipeline: wgpu::RenderPipeline,
    irradiance_pipeline: wgpu::RenderPipeline,
    prefilter_pipeline: wgpu::RenderPipeline,

    cube_sampler: wgpu::Sampler,
    lut_sampler: wgpu::Sampler,
    _brdf_lut: wgpu::Texture,
    brdf_lut_view: wgpu::TextureView,
}

impl IblGen {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rgltf-ibl-gen"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ibl.wgsl").into()),
        });

        // Gen uniform (group 0) + source-cube (group 1) layouts.
        let u_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgltf-ibl-u-bgl"),
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
        let src_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgltf-ibl-src-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // group 3 (mesh) layout: two cubes + a 2D LUT + two samplers.
        let cube_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgltf-ibl-bgl"),
            entries: &[
                cube_entry(0),
                cube_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgltf-ibl-u"),
            size: std::mem::size_of::<GenRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let u_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgltf-ibl-u-bg"),
            layout: &u_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: u_buffer.as_entire_binding() }],
        });

        // Cube sampler: linear + mip-linear + clamp (roughness lod on the prefilter cube).
        let cube_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rgltf-ibl-cube-samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rgltf-ibl-lut-samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Precompute pipelines.
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgltf-ibl-sky-pl"),
            bind_group_layouts: &[&u_bgl],
            push_constant_ranges: &[],
        });
        let conv_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgltf-ibl-conv-pl"),
            bind_group_layouts: &[&u_bgl, &src_bgl],
            push_constant_ranges: &[],
        });
        let brdf_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgltf-ibl-brdf-pl"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let make = |layout: &wgpu::PipelineLayout, fs: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rgltf-ibl-pipeline"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_gen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        let sky_pipeline = make(&sky_layout, "fs_sky", CUBE_FORMAT);
        let irradiance_pipeline = make(&conv_layout, "fs_irradiance", CUBE_FORMAT);
        let prefilter_pipeline = make(&conv_layout, "fs_prefilter", CUBE_FORMAT);
        let brdf_pipeline = make(&brdf_layout, "fs_brdf", LUT_FORMAT);

        // Bake the environment-independent BRDF LUT once.
        let brdf_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgltf-ibl-brdf"),
            size: wgpu::Extent3d { width: LUT_SIZE, height: LUT_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let brdf_lut_view = brdf_lut.create_view(&Default::default());
        {
            let mut enc = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rgltf-ibl-brdf-enc") });
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("rgltf-ibl-brdf-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &brdf_lut_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&brdf_pipeline);
                pass.draw(0..3, 0..1);
            }
            queue.submit(std::iter::once(enc.finish()));
        }

        Self {
            device,
            queue,
            bgl,
            u_buffer,
            u_bind_group,
            src_bgl,
            sky_pipeline,
            irradiance_pipeline,
            prefilter_pipeline,
            cube_sampler,
            lut_sampler,
            _brdf_lut: brdf_lut,
            brdf_lut_view,
        }
    }

    /// The group-3 bind-group layout the mesh pipeline is built against.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl
    }

    fn cube(&self, size: u32, mips: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgltf-ibl-cube"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CUBE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn face_view(tex: &wgpu::Texture, face: u32, mip: u32) -> wgpu::TextureView {
        tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("rgltf-ibl-face"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: mip,
            mip_level_count: Some(1),
            base_array_layer: face,
            array_layer_count: Some(1),
            ..Default::default()
        })
    }

    fn cube_view(tex: &wgpu::Texture) -> wgpu::TextureView {
        tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("rgltf-ibl-cubeview"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        })
    }

    /// Write the gen uniform, then render one fullscreen-triangle pass into `target`.
    /// Uniform-write and submit are interleaved per face so each pass reads its own
    /// value (queue writes are ordered before the following submit).
    fn face_pass(
        &self,
        target: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        src_bg: Option<&wgpu::BindGroup>,
        uniform: &GenRaw,
    ) {
        self.queue.write_buffer(&self.u_buffer, 0, bytemuck::bytes_of(uniform));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rgltf-ibl-enc") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rgltf-ibl-face-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.u_bind_group, &[]);
            if let Some(bg) = src_bg {
                pass.set_bind_group(1, bg, &[]);
            }
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    fn uniform(face: usize, roughness: f32, p: &EnvParams) -> GenRaw {
        let (fwd, right, up) = FACES[face];
        let v3 = |a: [f32; 3], w: f32| [a[0], a[1], a[2], w];
        GenRaw {
            fwd: v3(fwd, 0.0),
            right: v3(right, 0.0),
            up: v3(up, 0.0),
            params: [roughness, 0.0, 0.0, 0.0],
            sky: v3(p.sky, 1.0),
            horizon: v3(p.horizon, 1.0),
            ground: v3(p.ground, 1.0),
        }
    }

    /// Bake the irradiance + prefiltered-specular cubes for `params` and assemble the
    /// group-3 bind group (the BRDF LUT is shared/env-independent).
    pub fn build(&self, params: EnvParams) -> Ibl {
        // 1. Analytic environment cube (dropped after baking).
        let env = self.cube(ENV_SIZE, 1);
        for face in 0..6 {
            let view = Self::face_view(&env, face as u32, 0);
            self.face_pass(&view, &self.sky_pipeline, None, &Self::uniform(face, 0.0, &params));
        }
        let env_cube = Self::cube_view(&env);
        let src_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgltf-ibl-src-bg"),
            layout: &self.src_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&env_cube) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.cube_sampler) },
            ],
        });

        // 2. Diffuse irradiance cube.
        let irradiance = self.cube(IRR_SIZE, 1);
        for face in 0..6 {
            let view = Self::face_view(&irradiance, face as u32, 0);
            self.face_pass(&view, &self.irradiance_pipeline, Some(&src_bg), &Self::uniform(face, 0.0, &params));
        }

        // 3. Prefiltered specular cube (roughness per mip).
        let prefilter = self.cube(PRE_SIZE, PREFILTER_MIPS);
        for mip in 0..PREFILTER_MIPS {
            let roughness = mip as f32 / (PREFILTER_MIPS - 1) as f32;
            for face in 0..6 {
                let view = Self::face_view(&prefilter, face as u32, mip);
                self.face_pass(&view, &self.prefilter_pipeline, Some(&src_bg), &Self::uniform(face, roughness, &params));
            }
        }

        let irr_view = Self::cube_view(&irradiance);
        let pre_view = Self::cube_view(&prefilter);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgltf-ibl-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&irr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&pre_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.cube_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.lut_sampler) },
            ],
        });

        Ibl { bind_group, _irradiance: irradiance, _prefilter: prefilter }
    }
}
