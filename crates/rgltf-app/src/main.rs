//! rgltf — a native glTF viewer.
//!
//! The app owns the winit event loop and the wgpu device (the "game-embed" model):
//! we render the 3D scene into an offscreen texture and composite rinch's UI over it.
//! rinch is an embedded UI layer (`rinch::embed`), not the application shell — this
//! gives us a continuous render loop for animation, which rinch's event-driven paint
//! cycle can't self-drive.
//!
//! Pass a file path as argv[1] to open it on launch; the "Open glTF…" button uses the
//! native file dialog.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rinch::embed::{
    capture_texture_rgba, RinchContext, RinchContextConfig, RinchOverlayRenderer,
};
use rinch::platform::{MouseButton as PlatformMouseButton, PlatformEvent};
use rinch::prelude::*;
use rinch::wgpu;

use winit::application::ApplicationHandler;
use winit::event::{ButtonSource, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use rgltf_render::{Camera, Renderer, Scene};

/// The `data-viewport` name the 3D is composited into (matched by `viewport_rect`).
const VIEWPORT: &str = "viewport";

// ── Shared UI state ─────────────────────────────────────────────────────────
// The UI component creates these reactive signals and stashes them here; the app's
// render loop reads/writes them (same pattern as the game-embed example's GAME_STATE).

#[derive(Clone, PartialEq)]
struct TreeRow {
    id: usize,
    depth: u32,
    name: String,
    has_mesh: bool,
}

#[derive(Clone)]
struct UiState {
    tree_rows: Signal<Vec<TreeRow>>,
    status: Signal<String>,
}

thread_local! {
    static UI_STATE: RefCell<Option<UiState>> = const { RefCell::new(None) };
    /// Set by the "Open glTF…" button's onclick; polled by the render loop.
    static OPEN_REQUESTED: Cell<bool> = const { Cell::new(false) };
}

fn build_rows(scene: &Scene) -> Vec<TreeRow> {
    scene
        .nodes
        .iter()
        .enumerate()
        .map(|(id, n)| TreeRow { id, depth: n.depth, name: n.name.clone(), has_mesh: n.has_mesh() })
        .collect()
}

fn build_status(scene: &Scene) -> String {
    let exts = if scene.extensions_used.is_empty() {
        "no extensions".to_string()
    } else {
        format!("ext: {}", scene.extensions_used.join(", "))
    };
    let anim = if scene.animations.is_empty() {
        String::new()
    } else {
        format!(" · {} anim", scene.animations.len())
    };
    format!(
        "{} — {} nodes · {} tris · {} verts{} · {}",
        scene.name,
        scene.nodes.len(),
        scene.triangle_count(),
        scene.vertex_count(),
        anim,
        exts
    )
}

// ── UI component (embedded rinch DOM) ─────────────────────────────────────────

#[component]
fn ui() -> NodeHandle {
    let tree_rows = Signal::new(Vec::<TreeRow>::new());
    let status = Signal::new(String::from("Ready"));
    UI_STATE.with(|s| *s.borrow_mut() = Some(UiState { tree_rows, status }));

    // The transparent hole the 3D is composited into. `data-viewport` lets
    // `RinchContext::viewport_rect` find its rect and `wants_mouse` treat it as the
    // game area (so drags/scrolls there drive the camera, not the UI).
    let viewport = __scope.create_element("div");
    viewport.set_attribute("data-viewport", VIEWPORT);
    viewport.set_attribute("style", "flex:1; min-width:0; height:100%; background:transparent;");

    let on_open = move || OPEN_REQUESTED.with(|f| f.set(true));

    rsx! {
        div {
            style: "display:flex; flex-direction:column; height:100vh; overflow:hidden; \
                    font-family: system-ui, sans-serif; background:#1e1e22; color:#dcdce0;",

            // ── Toolbar ──────────────────────────────────────────────────
            div {
                style: "display:flex; align-items:center; gap:12px; padding:8px 14px; \
                        background:#26262b; border-bottom:1px solid #35353c;",
                span { style: "font-weight:700; letter-spacing:0.5px;", "rgltf" }
                button {
                    onclick: on_open,
                    style: "padding:4px 10px; border-radius:6px; border:1px solid #3a3a42; \
                            background:#2f2f36; color:#dcdce0; cursor:pointer; font-size:13px;",
                    "Open glTF…"
                }
                span { style: "margin-left:auto; color:#7a7a84; font-size:12px;",
                    "Phase 5 · animation" }
            }

            // ── Body: scene tree · viewport · inspector ──────────────────
            div {
                style: "display:flex; flex:1; min-height:0;",

                div {
                    style: "width:260px; flex-shrink:0; padding:10px 8px; overflow:auto; \
                            background:#232327; border-right:1px solid #35353c;",
                    div { style: "font-size:11px; text-transform:uppercase; letter-spacing:0.6px; \
                                  color:#7a7a84; margin:0 0 8px 6px;", "Scene" }
                    for row in tree_rows.get() {
                        div {
                            key: row.id,
                            style: {format!(
                                "display:flex; align-items:center; gap:7px; padding:2px 0 2px {}px; \
                                 font-size:13px; color:{};",
                                6 + row.depth * 14,
                                if row.has_mesh { "#d3d6dc" } else { "#8a8a92" }
                            )},
                            span { style: {format!(
                                "width:6px; height:6px; border-radius:50%; flex-shrink:0; background:{};",
                                if row.has_mesh { "#5aa0ff" } else { "#4a4a52" }
                            )} }
                            {row.name.clone()}
                        }
                    }
                }

                {viewport}

                div {
                    style: "width:280px; flex-shrink:0; padding:12px; overflow:auto; \
                            background:#232327; border-left:1px solid #35353c;",
                    div { style: "font-size:11px; text-transform:uppercase; letter-spacing:0.6px; \
                                  color:#7a7a84; margin-bottom:8px;", "Inspector" }
                    div { style: "color:#9a9aa2; font-size:13px; line-height:1.7;",
                        "Drag to orbit" br {} "Scroll to zoom" br {} "Open a .glb / .gltf file" }
                }
            }

            // ── Status bar ───────────────────────────────────────────────
            div {
                style: "padding:5px 14px; background:#26262b; border-top:1px solid #35353c; \
                        font-size:12px; color:#9a9aa2; white-space:nowrap; overflow:hidden; \
                        text-overflow:ellipsis;",
                {move || status.get()}
            }
        }
    }
}

// ── Blit shaders: composite the 3D texture + UI overlay onto the swapchain ─────
// Fullscreen triangle. `fs_scene` converts our offscreen (already sRGB-encoded) back
// to linear so the sRGB swapchain re-encodes it 1:1. `fs_ui` samples Vello's overlay
// verbatim (premultiplied-alpha blended over the scene).

const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let x = f32(i32(i) / 2) * 4.0 - 1.0;
    let y = f32(i32(i) % 2) * 4.0 - 1.0;
    var o: VsOut;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    o.uv = vec2<f32>((x + 1.0) / 2.0, 1.0 - (y + 1.0) / 2.0);
    return o;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment fn fs_scene(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(t, s, in.uv);
    return vec4<f32>(srgb_to_linear(c.rgb), c.a);
}

@fragment fn fs_ui(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
}
"#;

// ── App ────────────────────────────────────────────────────────────────────

struct App {
    window: Option<Arc<dyn Window>>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
    surface: Option<wgpu::Surface<'static>>,
    surface_format: wgpu::TextureFormat,
    surface_config: Option<wgpu::SurfaceConfiguration>,

    blit_scene: Option<wgpu::RenderPipeline>,
    blit_ui: Option<wgpu::RenderPipeline>,
    blit_bgl: Option<wgpu::BindGroupLayout>,
    blit_sampler: Option<wgpu::Sampler>,

    rinch_ctx: Option<RinchContext>,
    overlay: Option<RinchOverlayRenderer>,

    renderer: Option<Renderer>,
    camera: Camera,
    scene: Option<Scene>,
    anim: Option<usize>,
    start: Instant,

    dragging: bool,
    mouse_phys: (f32, f32),
    pending_events: Vec<PlatformEvent>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            queue: None,
            surface: None,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            surface_config: None,
            blit_scene: None,
            blit_ui: None,
            blit_bgl: None,
            blit_sampler: None,
            rinch_ctx: None,
            overlay: None,
            renderer: None,
            camera: Camera::new(1.0),
            scene: None,
            anim: None,
            start: Instant::now(),
            dragging: false,
            mouse_phys: (0.0, 0.0),
            pending_events: Vec::new(),
        }
    }

    fn scale_factor(&self) -> f64 {
        self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0)
    }

    fn size(&self) -> (u32, u32) {
        self.window
            .as_ref()
            .map(|w| {
                let s = w.surface_size();
                (s.width.max(1), s.height.max(1))
            })
            .unwrap_or((1280, 800))
    }

    fn logical_mouse(&self) -> (f32, f32) {
        let sf = self.scale_factor() as f32;
        (self.mouse_phys.0 / sf, self.mouse_phys.1 / sf)
    }

    /// Physical-pixel rect of the 3D viewport hole (or the full window if absent).
    fn viewport_px(&self) -> (u32, u32, u32, u32) {
        let (w, h) = self.size();
        let sf = self.scale_factor() as f32;
        match self.rinch_ctx.as_ref().and_then(|c| c.viewport_rect(VIEWPORT)) {
            Some(r) => (
                (r.x * sf).round().max(0.0) as u32,
                (r.y * sf).round().max(0.0) as u32,
                (r.width * sf).round().max(1.0) as u32,
                (r.height * sf).round().max(1.0) as u32,
            ),
            None => (0, 0, w, h),
        }
    }

    fn init_gpu(&mut self) {
        let window = self.window.as_ref().unwrap().clone();
        let (w, h) = self.size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rgltf-device"),
            // KTX2/Basis textures transcode to BC7 (see rgltf-asset).
            required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
            experimental_features: Default::default(),
        }))
        .expect("failed to create device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Prefer an sRGB surface (the compositor writes linear → auto-encodes).
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        self.surface_format = format;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Blit pipelines (scene = opaque + sRGB→linear, ui = premultiplied).
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rgltf-blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgltf-blit-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgltf-blit-pl"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });
        let make_blit = |entry: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rgltf-blit-pipeline"),
                layout: Some(&blit_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        self.blit_scene = Some(make_blit("fs_scene", None));
        self.blit_ui = Some(make_blit("fs_ui", Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)));
        self.blit_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rgltf-blit-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        self.blit_bgl = Some(blit_bgl);

        // Embedded rinch UI + overlay renderer.
        let mut rinch_ctx = RinchContext::new(
            RinchContextConfig {
                width: w,
                height: h,
                scale_factor: self.scale_factor(),
                theme: Some(ThemeProviderProps { dark_mode: true, ..Default::default() }),
            },
            ui,
        );
        // Poll loop repaints every frame, so the debug-notify hook can be a no-op.
        let _ = rinch_ctx.attach_debug("rgltf", || {});
        let overlay = RinchOverlayRenderer::new(&device, w, h, wgpu::TextureFormat::Rgba8Unorm);

        self.renderer = Some(Renderer::new(device.clone(), queue.clone(), w, h));
        self.rinch_ctx = Some(rinch_ctx);
        self.overlay = Some(overlay);
        self.surface = Some(surface);
        self.surface_config = Some(config);
        self.device = Some(device);
        self.queue = Some(queue);

        // Initial model from argv[1], else the demo cube.
        let initial = std::env::args()
            .nth(1)
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .and_then(|p| match rgltf_asset::load(&p) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("[rgltf][load-error] {e}");
                    None
                }
            })
            .unwrap_or_else(Scene::demo_cube);
        self.apply_scene(initial);
    }

    /// Upload a scene, frame the camera, select the first animation, refresh the UI.
    fn apply_scene(&mut self, scene: Scene) {
        if let Some(r) = &mut self.renderer {
            r.set_scene(&scene);
        }
        let (center, radius) = rgltf_render::scene_sphere(&scene);
        self.camera.frame(center, radius);
        self.anim = if scene.animations.is_empty() { None } else { Some(0) };
        self.start = Instant::now();
        UI_STATE.with(|s| {
            if let Some(ui) = s.borrow().as_ref() {
                ui.tree_rows.set(build_rows(&scene));
                ui.status.set(build_status(&scene));
            }
        });
        self.scene = Some(scene);
    }

    fn open_dialog(&mut self) {
        let picked = rfd::FileDialog::new()
            .set_title("Open glTF / GLB")
            .add_filter("glTF", &["glb", "gltf"])
            .pick_file();
        if let Some(path) = picked {
            match rgltf_asset::load(&path) {
                Ok(scene) => self.apply_scene(scene),
                Err(e) => UI_STATE.with(|s| {
                    if let Some(ui) = s.borrow().as_ref() {
                        ui.status.set(format!("Failed to open: {e}"));
                    }
                }),
            }
        }
    }

    fn handle_resize(&mut self, w: u32, h: u32) {
        let (w, h) = (w.max(1), h.max(1));
        if let (Some(cfg), Some(device), Some(surface)) =
            (&mut self.surface_config, &self.device, &self.surface)
        {
            cfg.width = w;
            cfg.height = h;
            surface.configure(device, cfg);
        }
        if let Some(overlay) = &mut self.overlay {
            if let Some(device) = &self.device {
                overlay.resize(device, w, h);
            }
        }
        if let Some(ctx) = &mut self.rinch_ctx {
            ctx.resize(w, h);
        }
    }

    fn render(&mut self) {
        if OPEN_REQUESTED.with(|f| f.replace(false)) {
            self.open_dialog();
        }

        let (Some(device), Some(queue)) = (self.device.clone(), self.queue.clone()) else {
            return;
        };

        // Feed input to rinch and pull its UI scene.
        let events: Vec<_> = self.pending_events.drain(..).collect();
        if let Some(ctx) = &mut self.rinch_ctx {
            let _ = ctx.update(&events);
        }

        // Size the 3D render target to the viewport hole.
        let (vx, vy, vw, vh) = self.viewport_px();
        if let Some(r) = &mut self.renderer {
            r.resize(vw, vh);
        }
        self.camera.set_aspect(vw as f32 / vh.max(1) as f32);

        // Evaluate the scene at the current animation time and render the 3D.
        let world = match &self.scene {
            Some(scene) => {
                let time = match self.anim.and_then(|i| scene.animations.get(i)) {
                    Some(clip) => self.start.elapsed().as_secs_f32() % clip.duration.max(1e-4),
                    None => 0.0,
                };
                scene.node_world_matrices(self.anim, time)
            }
            None => Vec::new(),
        };
        let scene_view = if let Some(r) = &mut self.renderer {
            r.render(&self.camera, &world);
            Some(r.color_target().1)
        } else {
            None
        };

        // Render the rinch UI to the overlay texture.
        let overlay_view = match (&mut self.overlay, &mut self.rinch_ctx) {
            (Some(overlay), Some(ctx)) => Some(overlay.render(&device, &queue, ctx.scene())),
            _ => None,
        };

        let frame = match self.surface.as_ref().unwrap().get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                if let (Some(surface), Some(cfg)) = (&self.surface, &self.surface_config) {
                    surface.configure(&device, cfg);
                }
                return;
            }
            Err(e) => {
                eprintln!("[rgltf] surface error: {e:?}");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let (fw, fh) = self.size();

        // Bind groups for the two blit passes.
        let bgl = self.blit_bgl.as_ref().unwrap();
        let sampler = self.blit_sampler.as_ref().unwrap();
        let mut make_bg = |tex_view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rgltf-blit-bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(tex_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            })
        };
        let scene_bg = scene_view.as_ref().map(&mut make_bg);
        let overlay_bg = overlay_view.as_ref().map(&mut make_bg);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rgltf-composite") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rgltf-composite-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.07, g: 0.07, b: 0.08, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // 3D into the viewport rect.
            if let (Some(pipe), Some(bg)) = (&self.blit_scene, &scene_bg) {
                pass.set_viewport(vx as f32, vy as f32, vw as f32, vh as f32, 0.0, 1.0);
                pass.set_scissor_rect(vx.min(fw), vy.min(fh), vw.min(fw), vh.min(fh));
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // UI overlay over the full frame (panels opaque, viewport hole transparent).
            if let (Some(pipe), Some(bg)) = (&self.blit_ui, &overlay_bg) {
                pass.set_viewport(0.0, 0.0, fw as f32, fh as f32, 0.0, 1.0);
                pass.set_scissor_rect(0, 0, fw, fh);
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));

        // MCP debug screenshots capture the composited surface before present.
        if let Some(ctx) = &mut self.rinch_ctx {
            for req in ctx.process_debug_commands() {
                match capture_texture_rgba(&device, &queue, &frame.texture, fw, fh, self.surface_format) {
                    Ok(rgba) => req.respond(fw, fh, rgba),
                    Err(e) => req.fail(e),
                }
            }
        }

        frame.present();
    }
}

impl ApplicationHandler for App {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("rgltf")
            .with_surface_size(winit::dpi::LogicalSize::new(1280u32, 800));
        let window: Arc<dyn Window> = Arc::from(event_loop.create_window(attrs).unwrap());
        self.window = Some(window);
        self.init_gpu();
        self.start = Instant::now();
    }

    fn window_event(&mut self, event_loop: &dyn ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::SurfaceResized(size) => self.handle_resize(size.width, size.height),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(ctx) = &mut self.rinch_ctx {
                    ctx.set_scale_factor(scale_factor);
                }
            }
            WindowEvent::PointerMoved { position, .. } => {
                let (px, py) = (position.x as f32, position.y as f32);
                if self.dragging {
                    let (dx, dy) = (px - self.mouse_phys.0, py - self.mouse_phys.1);
                    self.camera.orbit(dx, dy);
                }
                self.mouse_phys = (px, py);
                self.pending_events.push(PlatformEvent::MouseMove { x: px, y: py });
            }
            WindowEvent::PointerButton { state, button, .. } => {
                let btn = match button {
                    ButtonSource::Mouse(MouseButton::Left) => PlatformMouseButton::Left,
                    ButtonSource::Mouse(MouseButton::Right) => PlatformMouseButton::Right,
                    ButtonSource::Mouse(MouseButton::Middle) => PlatformMouseButton::Middle,
                    _ => return,
                };
                let (px, py) = self.mouse_phys;
                let (lx, ly) = self.logical_mouse();
                let wants_ui = self.rinch_ctx.as_ref().is_some_and(|c| c.wants_mouse(lx, ly));
                match state {
                    ElementState::Pressed => {
                        if wants_ui {
                            self.pending_events.push(PlatformEvent::MouseDown { x: px, y: py, button: btn });
                        } else if btn == PlatformMouseButton::Left {
                            self.dragging = true;
                        }
                    }
                    ElementState::Released => {
                        if self.dragging && btn == PlatformMouseButton::Left {
                            self.dragging = false;
                        } else {
                            self.pending_events.push(PlatformEvent::MouseUp { x: px, y: py, button: btn });
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64 * 40.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y,
                };
                let (lx, ly) = self.logical_mouse();
                let wants_ui = self.rinch_ctx.as_ref().is_some_and(|c| c.wants_mouse(lx, ly));
                if wants_ui {
                    let (px, py) = self.mouse_phys;
                    self.pending_events.push(PlatformEvent::MouseWheel {
                        x: px,
                        y: py,
                        delta_x: 0.0,
                        delta_y: scroll_y,
                    });
                } else {
                    self.camera.zoom(scroll_y as f32);
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &dyn ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(App::new()).unwrap();
}
