//! rgltf — a native glTF viewer built on the rinch GUI framework.
//!
//! Phase 1: open a standard `.glb`/`.gltf`, render it lit in the docked viewport,
//! populate the scene tree, and frame the camera to the model. The 3D is rendered
//! into an offscreen texture on rinch's shared wgpu device and composited zero-copy.
//!
//! Pass a file path as argv[1] to open it on launch (handy for testing); the
//! "Open glTF…" button uses the native file dialog.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;

use rinch::prelude::*;
use rinch::render_surface::create_render_surface_with_name;
use rinch::wgpu;

use rgltf_render::{Camera, Renderer, Scene};

/// Binds the render surface to the `data-viewport` hole below.
const VIEWPORT: &str = "viewport";

/// One row of the scene-tree panel.
#[derive(Clone, PartialEq)]
struct TreeRow {
    id: usize,
    depth: u32,
    name: String,
    has_mesh: bool,
}

fn build_rows(scene: &Scene) -> Vec<TreeRow> {
    scene
        .nodes
        .iter()
        .enumerate()
        .map(|(id, n)| TreeRow { id, depth: n.depth, name: n.name.clone(), has_mesh: n.has_mesh })
        .collect()
}

fn build_status(scene: &Scene) -> String {
    let exts = if scene.extensions_used.is_empty() {
        "no extensions".to_string()
    } else {
        format!("ext: {}", scene.extensions_used.join(", "))
    };
    format!(
        "{} — {} nodes · {} tris · {} verts · {}",
        scene.name,
        scene.nodes.len(),
        scene.triangle_count(),
        scene.vertex_count(),
        exts
    )
}

#[component]
fn app() -> NodeHandle {
    let handle = gpu_handle().expect("gpu_handle available after the renderer starts");
    let device = handle.device.clone();
    let queue = handle.queue.clone();

    let renderer = Rc::new(RefCell::new(Renderer::new(device, queue, 960, 720)));
    let camera = Rc::new(RefCell::new(Camera::new(960.0 / 720.0)));

    let surface = create_render_surface_with_name(VIEWPORT);
    let registrar = surface.gpu_registrar();

    // Reactive UI state.
    let tree_rows = Signal::new(Vec::<TreeRow>::new());
    let status = Signal::new(String::from("Ready"));

    // Render one frame and publish it to the compositor (zero-copy).
    let redraw = {
        let renderer = renderer.clone();
        let camera = camera.clone();
        let registrar = registrar.clone();
        move || {
            let (w, h) = {
                let mut r = renderer.borrow_mut();
                r.render(&camera.borrow());
                r.size()
            };
            if w == 0 || h == 0 {
                return;
            }
            let (tex, view) = renderer.borrow().color_target();
            registrar.set_texture_source(tex, view, w, h);
            registrar.notify_frame_ready();
        }
    };

    // Apply a freshly loaded scene: upload geometry, frame the camera, redraw,
    // refresh the tree + status.
    let apply = {
        let renderer = renderer.clone();
        let camera = camera.clone();
        let redraw = redraw.clone();
        move |scene: Scene| {
            renderer.borrow_mut().set_scene(&scene);
            let (center, radius) = rgltf_render::scene_sphere(&scene);
            camera.borrow_mut().frame(center, radius);
            redraw();
            tree_rows.set(build_rows(&scene));
            status.set(build_status(&scene));
        }
    };

    // Initial scene: a file from argv[1] if present & loadable, else the demo cube.
    let initial = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .and_then(|p| match rgltf_asset::load(&p) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("[rgltf][load-error] {e}");
                status.set(format!("Failed to open: {e}"));
                None
            }
        })
        .unwrap_or_else(Scene::demo_cube);
    apply(initial);

    // Resize: rinch reports the viewport rect in physical pixels.
    {
        let renderer = renderer.clone();
        let camera = camera.clone();
        let redraw = redraw.clone();
        surface.set_resize_callback(move |w, h| {
            if w == 0 || h == 0 {
                return;
            }
            renderer.borrow_mut().resize(w, h);
            camera.borrow_mut().set_aspect(w as f32 / h as f32);
            redraw();
        });
    }

    // Input: drag to orbit, wheel to zoom (local viewport coordinates).
    {
        let camera = camera.clone();
        let redraw = redraw.clone();
        let dragging = Rc::new(Cell::new(false));
        let last = Rc::new(Cell::new((0.0f32, 0.0f32)));
        surface.set_event_handler(move |ev| match ev {
            SurfaceEvent::MouseDown { x, y, button: SurfaceMouseButton::Left } => {
                dragging.set(true);
                last.set((x, y));
            }
            SurfaceEvent::MouseUp { .. } | SurfaceEvent::MouseLeave => dragging.set(false),
            SurfaceEvent::MouseMove { x, y } => {
                if dragging.get() {
                    let (lx, ly) = last.get();
                    last.set((x, y));
                    camera.borrow_mut().orbit(x - lx, y - ly);
                    redraw();
                }
            }
            SurfaceEvent::MouseWheel { delta_y, .. } => {
                camera.borrow_mut().zoom(delta_y);
                redraw();
            }
            _ => {}
        });
    }

    // The transparent hole the compositor samples our texture through.
    //  - `data-viewport` (name)      → matched by the compositor (hole-punch + GPU layer)
    //  - `data-render-surface` (id)  → matched by input routing so SurfaceEvents reach us
    let viewport = __scope.create_element("div");
    viewport.set_attribute("data-viewport", VIEWPORT);
    viewport.set_attribute("data-render-surface", &surface.id().to_string());
    viewport.set_attribute("style", "flex:1; min-width:0; height:100%; background:transparent;");

    // The Open button loads a file via the native dialog and applies it.
    let on_open = {
        let apply = apply.clone();
        move || {
            let picked = rfd::FileDialog::new()
                .set_title("Open glTF / GLB")
                .add_filter("glTF", &["glb", "gltf"])
                .pick_file();
            if let Some(path) = picked {
                match rgltf_asset::load(&path) {
                    Ok(scene) => apply(scene),
                    Err(e) => status.set(format!("Failed to open: {e}")),
                }
            }
        }
    };

    // Registry holds a functional clone; keep our handle alive for the app's life.
    std::mem::forget(surface);

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
                    "Phase 4 · KHR_texture_basisu" }
            }

            // ── Body: scene tree · viewport · inspector ──────────────────
            div {
                style: "display:flex; flex:1; min-height:0;",

                // Scene tree (left)
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

                // 3D viewport (center)
                {viewport}

                // Inspector (right)
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

fn main() {
    let props = WindowProps {
        title: "rgltf".into(),
        width: 1280,
        height: 800,
        transparent: true, // required for zero-copy GPU-layer compositing (see PLAN.md)
        ..Default::default()
    };

    let gpu = RinchGpuConfig {
        // KTX2/Basis textures transcode to BC7, so the shared device must expose BC.
        // Universally supported on desktop GPUs (the AMD 890M here does); a non-BC
        // adapter would fail device creation — acceptable for a desktop viewer. Images
        // whose dimensions aren't 4-aligned fall back to RGBA8 (no BC needed).
        required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
        required_limits: wgpu::Limits::default(),
    };

    run_with_gpu_config(app, props, None, gpu);
}
