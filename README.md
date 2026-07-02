# rgltf

A native glTF/GLB viewer for Linux that renders [gltfpack](https://github.com/zeux/meshoptimizer)-compressed models correctly.

Native glTF viewers on Linux tend to fall over on gltfpack output — meshopt-compressed buffers come through as garbage, quantized meshes render distorted, and KTX2/Basis textures don't load at all. rgltf exists to render those files the way they're meant to look, on top of ordinary glTF 2.0 PBR.

## What it handles

- **`EXT_meshopt_compression`** — meshopt-decoded vertex/index streams, including the octahedral, quaternion, and exponential filters.
- **`KHR_mesh_quantization`** — integer/normalized attributes dequantized correctly (the common accessor readers assume float and silently don't).
- **`KHR_texture_basisu`** — KTX2 textures transcoded from ETC1S *and* UASTC to BC7, with full mip chains (RGBA8 fallback for non-block-aligned sizes).
- **`KHR_texture_transform`** — UV transforms baked into the mesh.
- Standard **metallic-roughness PBR** — base-color / normal / metallic-roughness / occlusion / emissive maps, tangents generated when absent, a directional key light with a hemisphere-ambient term standing in for IBL, and ACES tone mapping.

Loads plain `.glb`/`.gltf` too, including PNG/JPEG and base64 `data:` URI textures.

Planned: animation and skinning, `EXT_mesh_gpu_instancing`, morph targets, and image-based lighting.

## How it's built

rgltf draws into an offscreen wgpu texture that's composited **zero-copy** into a docked viewport inside the [rinch](https://github.com/joeleaver/rinch) GUI framework. rinch owns the window and the GPU device and hands rgltf a shared `Device`/`Queue`.

The renderer is written from scratch instead of reaching for bevy/rend3/renderling, for one concrete reason: the wgpu that rgltf draws with must be the *same crate instance* rinch links, so the texture can be handed over without a copy. rinch pins a patched wgpu fork; a general-purpose renderer brings its own pinned wgpu (and, in some cases, owns the window and event loop), which collides head-on. A small, focused PBR renderer avoids the whole problem.

Three crates:

- `rgltf-render` — the wgpu PBR renderer, with no dependency on rinch.
- `rgltf-asset` — glTF parsing and gltfpack extension decode.
- `rgltf-app` — the rinch UI shell: scene tree, viewport, inspector, orbit camera.

## Building

```sh
git clone https://github.com/joeleaver/rgltf
cd rgltf
cargo build -p rgltf-app
```

[rinch](https://github.com/joeleaver/rinch) is pulled in as a pinned git dependency, so no extra checkout is needed. To hack on rgltf and rinch together, point the dependency at a local checkout with a patch in the workspace `Cargo.toml`:

```toml
[patch."https://github.com/joeleaver/rinch"]
rinch = { path = "../rinch/crates/rinch" }
```

KTX2/Basis support compiles Khronos [KTX-Software](https://github.com/KhronosGroup/KTX-Software) via cmake, so you'll need a C/C++ toolchain and cmake on `PATH`. The one cmake quirk (a policy-version minimum for the bundled encoder) is handled by `.cargo/config.toml`, so a plain `cargo build` works. A GPU with BC texture support is required — universal on desktop.

## Running

```sh
cargo run -p rgltf-app -- path/to/model.glb
```

Or start it with no argument and use **Open glTF…**. Drag to orbit, scroll to zoom.

To produce a gltfpack-compressed file to try it on:

```sh
gltfpack -i model.glb -o model-packed.glb -c -tc      # meshopt + ETC1S KTX2
gltfpack -i model.glb -o model-packed.glb -c -tc -tu  # meshopt + UASTC KTX2
```

Texture compression needs a native gltfpack build with Basis support (the npm build doesn't include it).

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
