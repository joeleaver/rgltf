# rgltf — a native glTF viewer with full gltfpack extension support

**Goal:** a fast, native (Linux-first, cross-platform) glTF/GLB viewer that correctly
renders files produced by [gltfpack](https://github.com/zeux/meshoptimizer) — i.e. one
that handles meshopt compression, mesh quantization, KTX2/Basis textures, and GPU
instancing, on top of standard glTF 2.0 PBR. Built as a Rust app embedding a
from-scratch wgpu renderer inside the [rinch](https://github.com/joeleaver/rinch) GUI
framework.

---

## 1. Architecture

### Embedding model (verified against rinch source)

rinch's **`RenderSurface`** path is the fit for a docked-editor layout:

- rinch owns the winit window and the wgpu `Device`/`Queue`/`Adapter`. We obtain them
  via `rinch::shell::desktop::gpu_handle() -> &GpuHandle { device, queue, adapter }`
  (all `Arc`), after starting rinch with **`run_with_gpu_config(app, WindowProps, theme,
  RinchGpuConfig { required_features, required_limits })`** — the hook that lets us raise
  device features (`TEXTURE_COMPRESSION_BC`, `FLOAT32_FILTERABLE`) so the *shared* device
  can host our pipelines and compressed textures.
- The 3D viewport is a `create_render_surface_with_name("viewport")` bound to a
  `<div data-viewport="viewport" style="…; background:transparent;">` hole in a CSS flex
  layout. We render into a wgpu color texture on the shared device and hand it over
  **zero-copy**: `handle = surface.gpu_registrar(); handle.set_texture_source(tex, view,
  w, h); handle.notify_frame_ready();`. rinch composites it under the UI with a
  hole-punch + WGSL pass — no GPU→CPU readback.
- Input: `surface.set_event_handler(|SurfaceEvent| …)` delivers `MouseDown/Move/Up/Wheel/
  KeyDown` in **viewport-local logical coordinates** → camera control.
- Resize: `surface.set_resize_callback(|w, h| …)` fires on first layout and every resize,
  in **physical pixels** (HiDPI-correct) → resize render targets + camera aspect.

### Verified integration requirements (Phase 0)

A zero-copy GPU viewport in a docked layout needs three non-obvious things (each
independently causes a blank/non-interactive viewport — all confirmed by debugging):

1. **`transparent: true` window.** rinch forces the Vello UI base transparent only for
   CPU `composite_layers`, not GPU-only layers — so an opaque window's base fills the
   punched hole and the UI paints over the 3D. (`shell/desktop.rs` ~L465.)
2. **`overflow:hidden` on the opaque-background ancestor.** The hole-punch only cuts a
   node's background when it clips (`overflow: hidden|scroll|auto`). (`paint/mod.rs` ~L872.)
3. **`data-render-surface="{surface.id()}"` on the hole div** (in addition to
   `data-viewport="{name}"`). Compositing matches by name; *input* routing matches by the
   numeric surface id. (`app/click_handling.rs` ~L539.) The built-in `RenderSurface`
   component sets both; a manual hole must too.

### wgpu lockstep (critical)

The `wgpu::Texture` we hand to `set_texture_source` must be the **same crate instance**
rinch links. rinch pins `wgpu = "27"` and patches `wgpu*`/`naga` to
`joeleaver/wgpu-fork#rinch-patch` (rebased to v27.0.1), plus a `winit`/`dpi` fork. Our
workspace root **replicates that entire `[patch.crates-io]` block** (a `[patch]` is only
read from the root manifest, never from a path dependency's workspace). We depend on
`rinch` by **path** to the local checkout so we build against the working tree, and use
`wgpu = "27"` (which the patch redirects to the fork). rinch also re-exports its wgpu as
`rinch::wgpu` — used directly in the app crate for GPU-config types.

### Workspace layout

```
rgltf/
├── Cargo.toml                 # workspace + the mirrored [patch.crates-io]
├── rust-toolchain.toml        # stable
├── PLAN.md                    # this document
└── crates/
    ├── rgltf-render/          # from-scratch wgpu renderer (rinch-independent)
    │   ├── src/lib.rs         # Renderer: new(device,queue,w,h) · resize · render(&Camera) · color_target()
    │   ├── src/camera.rs      # orbit camera (glam)
    │   └── src/shader.wgsl
    ├── rgltf-asset/           # glTF parse + gltfpack extension decode (CPU side)
    │   └── src/lib.rs         # load(path) -> Scene; grows per phase
    └── rgltf-app/             # the binary `rgltf`: rinch UI shell + wiring
        └── src/main.rs
```

Rationale for a from-scratch renderer over `bevy`/`rend3`/`renderling`: each brings its
own pinned `wgpu` and (bevy) owns the window/event loop, which collides head-on with the
wgpu-fork lockstep and with rinch owning the window. A focused wgpu-27 PBR renderer is
the only clean fit and keeps us in full control of the extension decode paths.

---

## 2. Dependencies (added per phase to keep early builds lean)

| Need | Crate | Notes / build risk |
|------|-------|--------------------|
| GUI + window + device | `rinch` (path) | features `desktop, gpu, components, theme, debug` |
| GPU | `wgpu = "27"` (fork) | unified via `[patch]` |
| Math | `glam` | `to_cols_array` → uniform upload |
| POD casts | `bytemuck` | |
| glTF parse | `gltf = "1"` | features for `KHR_materials_*`, `KHR_lights_punctual`, `KHR_texture_transform`, `KHR_materials_variants`, `extensions`, `extras`, `import`. Surfaces raw JSON extensions for meshopt/basisu that we decode ourselves. |
| meshopt decode | `meshopt` | `decode_vertex_buffer`, `decode_index_buffer`, filter decoders. Vendors C meshoptimizer via `cc` → needs a C compiler (present). |
| KTX2 + Basis transcode | `libktx-rs` (Khronos KTX-Software v4) | **Chosen over `basis-universal`** — that crate can't transcode ETC1S/BasisLZ-in-KTX2 (gltfpack `-tc`'s default), only `.basis`/UASTC slices. `libktx-rs`'s `Ktx2::transcode_basis` handles both ETC1S & UASTC → BC7/RGBA directly. Bundles the C source (no submodule); pre-generated bindings (no libclang); cmake build (needs `CMAKE_POLICY_VERSION_MINIMUM=3.5`, set in `.cargo/config.toml`). |
| PNG/JPEG | `image` | non-KTX2 textures (replaces `gltf::import_images` so KTX2 + PNG/JPEG coexist) |
| base64 `data:` URIs | `base64` | decode embedded `.gltf` buffers/images |

---

## 3. Asset load pipeline (ordered)

1. **Parse** `.glb`/`.gltf` (`gltf::import` or manual `gltf-json` + buffer resolution).
2. **Resolve buffers** (embedded base64 / external `.bin` / GLB BIN chunk).
3. **meshopt decode** — for each `bufferView` carrying `EXT_meshopt_compression`, decode
   the byte stream (`decode_vertex_buffer`/`decode_index_buffer`) then apply the declared
   filter (`NONE`/`OCTAHEDRAL`/`QUATERNION`/`EXPONENTIAL`). Produces the real bufferView bytes.
4. **Dequantize accessors** — honor `KHR_mesh_quantization`: integer/normalized POSITION
   (i16/u16 + node or mesh scale/translation), oct-encoded NORMAL/TANGENT, normalized UVs
   (+ `KHR_texture_transform`). Recover float attributes.
5. **GPU upload** — interleave/prepare vertex + index buffers; build per-primitive draw data.
6. **Textures** — KTX2 (`KHR_texture_basisu`): parse container, choose a transcode target
   from adapter features (BC7 sRGB for color, BC5 for normal, BC1/BC3 otherwise; RGBA8
   fallback), upload compressed blocks. Non-KTX2: decode via `image`, upload RGBA8,
   respecting sRGB-vs-linear per slot.
7. **Materials** — metallic-roughness + emissive; pass-through extensions
   (`emissive_strength`, `transmission`, `clearcoat`, `sheen`, `specular`, `ior`, `unlit`,
   `variants`); build bind groups.
8. **Scene graph** — nodes (TRS or matrix), meshes, `EXT_mesh_gpu_instancing`, skins,
   animations, morph targets, `KHR_lights_punctual`.

**Correctness gotchas to respect:** meshopt filter application order (decode → filter);
octahedral normal decode; KTX2 transcode-target selection + sRGB flags; premultiplied
alpha; Y-flip / winding / handedness (glTF is right-handed, +Y up, CCW front, UV origin
top-left); node `matrix` vs `T/R/S`; generating tangents when absent (for normal maps).

---

## 4. Renderer design

- **PBR** metallic-roughness BRDF (Khronos reference model), normal mapping with
  tangents, emissive (+ `KHR_materials_emissive_strength`).
- **Lighting:** MVP ships a couple of analytic directional/point lights + a flat ambient;
  **IBL** (prefiltered specular + irradiance from an environment map) follows — the thing
  that makes PBR read correctly. `KHR_lights_punctual` honored.
- **Tone mapping:** Khronos PBR-neutral (or ACES) + sRGB encode in the fragment shader,
  since we render into an `Rgba8Unorm` texture the compositor samples verbatim.
- **Deferred past MVP:** transmission/volume refraction, clearcoat/sheen, MSAA, shadows.

---

## 5. UI shell (rinch)

- Docked flex layout: toolbar (File ▸ Open via `rfd`), left **scene-tree** panel
  (`Tree`), center **viewport** hole, right **inspector** (material/mesh/animation via
  sliders, selects, collapsible sections), bottom status bar.
- Reactive state via `Signal`s: loaded scene, selected node, animation clip, playback,
  camera presets, extension diagnostics ("this file uses: EXT_meshopt_compression, …").
- The `debug` feature exposes rinch's MCP bridge so the running viewer can be screenshotted
  and inspected during development.

---

## 6. Phased delivery

| Phase | Goal | Success check |
|-------|------|---------------|
| **0** ✅ | Integration proof: rinch docked shell + orbitable cube rendered zero-copy into the viewport via the shared device. | **DONE** — 3-panel layout, cube composites zero-copy, orbits on drag; verified via rinch MCP screenshots. |
| **1** ✅ | Load & render a standard glTF/GLB as lit geometry; scene-tree populated; File▸Open. | **DONE** — Khronos `Duck`/`Box` load & render (Lambert lit, `baseColorFactor`), tree + status populated, camera auto-frames. Verified via MCP. |
| **2** ✅ | Full PBR: metallic-roughness, base-color/normal/MR/emissive/occlusion textures, sRGB handling, tangents, directional light + hemisphere ambient, ACES tone map. | **DONE** — `DamagedHelmet` (emissive HUD, metal, normals) & `Avocado` render correctly. Per-material bind groups, tangents generated when absent. Verified via MCP. |
| **3** ✅ | gltfpack core: `EXT_meshopt_compression` + `KHR_mesh_quantization` (+ `KHR_texture_transform`). | **DONE** — `gltfpack -c` DamagedHelmet renders identically to the uncompressed source; meshopt decode via `meshopt::ffi` (incl. octahedral filter), quantization-aware accessor reader, fallback-buffer resolver, UV transform. Verified via MCP. |
| **4** ✅ | `KHR_texture_basisu`: KTX2 + Basis transcode with BC targets + fallback. | **DONE** — ETC1S (`gltfpack -tc`) & UASTC (`-tc -tu`) `DamagedHelmet` transcode to BC7 (full mip chain) via `libktx-rs` and render matching the uncompressed golden; non-4-aligned KTX2 falls back to RGBA8; PNG/JPEG + base64 `data:` URIs handled via `image`/`base64`. Verified via MCP (ETC1S, UASTC, golden, embedded-data-URI Duck). |
| **5** | Animation + skinning (LINEAR/STEP/CUBICSPLINE, joints), `EXT_mesh_gpu_instancing`, morph targets. | Animated/skinned sample plays; instanced sample draws all instances. |
| **6** | IBL, material extensions polish, camera presets, drag-and-drop open, perf. | Environment-lit PBR; drop-to-open; smooth on large models. |

---

## 7. Top risks & mitigations

- **wgpu type unification** across the `[patch]` boundary → verified by building the app
  against `rinch::wgpu`; if it ever drifts, switch the render crate to consume `rinch::wgpu`.
- ~~**`basis-universal` build on Linux**~~ → **resolved**: switched to `libktx-rs` (bundled C
  source, pre-generated bindings, cmake). `basis-universal` couldn't transcode ETC1S-in-KTX2
  anyway. Only remaining caveat: cmake ≥4 needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` (baked into
  `.cargo/config.toml`), and BC is a *required* device feature (a non-BC adapter fails device
  creation — acceptable for a desktop viewer; non-4-aligned KTX2 still fall back to RGBA8).
- **meshopt/quantization correctness** → golden-image compare a gltfpack file against its
  uncompressed source (Phase 3 success check is exactly this).
- **Cross-workspace path dep + patch** → the workspace root owns the full patch set; the
  first successful `cargo build` confirms resolution.
