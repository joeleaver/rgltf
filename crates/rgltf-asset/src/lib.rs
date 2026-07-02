//! rgltf-asset — glTF loading and gltfpack-extension decoding.
//!
//! Parses a `.glb`/`.gltf` into world-space **primitives** (position/normal/
//! tangent/uv), **materials** (metallic-roughness factors + texture refs), and
//! decoded **images** (RGBA8 + sRGB flag). Node transforms are baked in; missing
//! normals/tangents are generated.
//!
//! gltfpack extensions handled (Phase 3):
//!   - `EXT_meshopt_compression`: compressed bufferViews are decoded (meshopt) and
//!     un-filtered (octahedral/quaternion/exponential) into the fallback buffers.
//!   - `KHR_mesh_quantization`: the accessor reader is component-type/normalization
//!     aware, so integer/normalized attributes dequantize correctly.
//!   - `KHR_texture_transform`: the UV transform is baked into UV0.
//!
//! Later phases add `KHR_texture_basisu` (KTX2/Basis), animation, and instancing.

use glam::{Mat4, Quat, Vec2, Vec3};

// ── Bounds ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) }
    }
    fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    pub fn is_valid(&self) -> bool {
        self.min.cmple(self.max).all()
    }
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    pub fn radius(&self) -> f32 {
        if self.is_valid() { (self.max - self.min).length() * 0.5 } else { 1.0 }
    }
}

// ── Vertex / geometry ─────────────────────────────────────────────────────────

/// Interleaved PBR vertex (local space). `tangent.w` is the bitangent sign.
/// `joints`/`weights` are the `JOINTS_0`/`WEIGHTS_0` skinning influences (all-zero
/// for un-skinned geometry — the shader branches on the per-instance skin flag).
/// `repr(C)` + `Pod` so the renderer can upload `&[Vertex]` directly (no padding:
/// 12+12+16+8+8+16 = 72 bytes, 4-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

/// A drawable primitive: **local-space** geometry (node transforms are applied at
/// draw time, not baked) + an index into [`Scene::materials`].
#[derive(Debug)]
pub struct Primitive {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material: usize,
}

// ── Materials / textures / images ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// A reference to an image plus which UV set samples it (Phase 2 uses UV0 only).
#[derive(Clone, Copy, Debug)]
pub struct TexRef {
    pub image: usize,
    pub texcoord: u32,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub emissive_strength: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub unlit: bool,
    pub base_color_tex: Option<TexRef>,
    pub metallic_roughness_tex: Option<TexRef>,
    pub normal_tex: Option<TexRef>,
    pub occlusion_tex: Option<TexRef>,
    pub emissive_tex: Option<TexRef>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "default".into(),
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0, 0.0, 0.0],
            emissive_strength: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            unlit: false,
            base_color_tex: None,
            metallic_roughness_tex: None,
            normal_tex: None,
            occlusion_tex: None,
            emissive_tex: None,
        }
    }
}

/// The GPU-neutral pixel format of a decoded [`ImageData`]. `Rgba8` is uncompressed
/// 8-bit RGBA (PNG/JPEG); the `Bc*` variants are block-compressed, transcoded from
/// KTX2/Basis. The `srgb` flag on [`ImageData`] picks the sRGB vs linear GPU variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexFormat {
    Rgba8,
    Bc1Rgba,
    Bc3Rgba,
    Bc4R,
    Bc5Rg,
    Bc7Rgba,
}

impl TexFormat {
    /// Bytes per 4×4 block for compressed formats; bytes per pixel for `Rgba8`.
    /// (Used to compute block-aligned `bytes_per_row` on upload.)
    pub fn block_bytes(self) -> u32 {
        match self {
            TexFormat::Bc1Rgba | TexFormat::Bc4R => 8,
            TexFormat::Bc3Rgba | TexFormat::Bc5Rg | TexFormat::Bc7Rgba => 16,
            TexFormat::Rgba8 => 4,
        }
    }
    pub fn is_compressed(self) -> bool {
        self != TexFormat::Rgba8
    }
}

/// One mip level: its dimensions + tightly-packed bytes (RGBA8 pixel rows, or 4×4
/// compressed blocks in row-major order).
#[derive(Debug)]
pub struct MipLevel {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// A decoded image ready for GPU upload. `format` says how `mips` are encoded; `srgb`
/// = whether to upload as an sRGB texture format (true for base-color/emissive; false
/// for normal/metallic-roughness/occlusion). `mips[0]` is the full-resolution level —
/// KTX2 images carry the whole chain, PNG/JPEG a single level.
#[derive(Debug)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub format: TexFormat,
    pub srgb: bool,
    pub mips: Vec<MipLevel>,
}

// ── Scene graph ─────────────────────────────────────────────────────────────

/// A mesh: a bundle of local-space primitives, instantiated by one or more nodes.
#[derive(Debug)]
pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

/// A scene-graph node. The local transform is stored decomposed (TRS) so animation
/// channels can drive translation / rotation / scale independently.
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub mesh: Option<usize>,
    /// Index into [`Scene::skins`] when this node instantiates a *skinned* mesh.
    pub skin: Option<usize>,
    pub depth: u32,
}

impl Node {
    pub fn local_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
    pub fn has_mesh(&self) -> bool {
        self.mesh.is_some()
    }
}

/// A skin: the joint nodes whose animated world transforms deform a skinned mesh,
/// plus the per-joint inverse bind matrices (bind-pose mesh-space → joint-space).
/// The joint matrix palette for the vertex shader is `world[joints[j]] · inverse_bind[j]`
/// (the skinned mesh node's own transform is ignored, per the glTF spec).
#[derive(Clone, Debug)]
pub struct Skin {
    pub joints: Vec<usize>,
    pub inverse_bind: Vec<Mat4>,
    /// The common-root node hint (`skin.skeleton`); unused for joint-matrix math.
    pub skeleton: Option<usize>,
}

/// Keyframe interpolation mode of an animation sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline,
}

/// Which node property an animation channel drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimPath {
    Translation,
    Rotation,
    Scale,
    Weights,
}

/// Keyframe times (`input`) + flat output values. For `CubicSpline`, each keyframe's
/// output is `[inTangent, value, outTangent]` (so 3× `components`); otherwise one
/// `components`-tuple per keyframe.
#[derive(Clone, Debug)]
pub struct AnimSampler {
    pub interpolation: Interpolation,
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub components: usize,
}

impl AnimSampler {
    /// Values-per-keyframe stride in `output` (3× for cubic spline's tangents).
    fn stride(&self) -> usize {
        match self.interpolation {
            Interpolation::CubicSpline => self.components * 3,
            _ => self.components,
        }
    }

    /// The `components` values of keyframe `k`. For cubic spline this is the *value*
    /// slice (middle third), skipping the in/out tangents.
    fn value(&self, k: usize) -> &[f32] {
        let s = self.stride();
        let base = k * s
            + match self.interpolation {
                Interpolation::CubicSpline => self.components, // skip inTangent
                _ => 0,
            };
        &self.output[base..base + self.components]
    }

    /// Cubic-spline in/out tangents of keyframe `k`.
    fn tangents(&self, k: usize) -> (&[f32], &[f32]) {
        let s = self.stride();
        let base = k * s;
        let c = self.components;
        (&self.output[base..base + c], &self.output[base + 2 * c..base + 3 * c])
    }

    /// Locate the keyframe interval bracketing `t`: `(i0, i1, frac, dt)`.
    fn bracket(&self, t: f32) -> (usize, usize, f32, f32) {
        let times = &self.input;
        let last = times.len() - 1;
        if t <= times[0] {
            return (0, 0, 0.0, 0.0);
        }
        if t >= times[last] {
            return (last, last, 0.0, 0.0);
        }
        let mut i = 0;
        while i + 1 < times.len() && times[i + 1] <= t {
            i += 1;
        }
        let (t0, t1) = (times[i], times[i + 1]);
        let dt = t1 - t0;
        let frac = if dt > 0.0 { (t - t0) / dt } else { 0.0 };
        (i, i + 1, frac, dt)
    }

    /// Sample `components` scalars at time `t` (used for translation/scale/weights).
    fn sample(&self, t: f32) -> Vec<f32> {
        let c = self.components;
        if self.input.is_empty() {
            return vec![0.0; c];
        }
        let (i0, i1, f, dt) = self.bracket(t);
        match self.interpolation {
            Interpolation::Step => self.value(i0).to_vec(),
            Interpolation::Linear => {
                let a = self.value(i0);
                let b = self.value(i1);
                (0..c).map(|k| a[k] + (b[k] - a[k]) * f).collect()
            }
            Interpolation::CubicSpline => {
                let v0 = self.value(i0).to_vec();
                let v1 = self.value(i1).to_vec();
                let (_, out0) = self.tangents(i0);
                let (in1, _) = self.tangents(i1);
                let (f2, f3) = (f * f, f * f * f);
                let (h00, h10) = (2.0 * f3 - 3.0 * f2 + 1.0, f3 - 2.0 * f2 + f);
                let (h01, h11) = (-2.0 * f3 + 3.0 * f2, f3 - f2);
                (0..c)
                    .map(|k| h00 * v0[k] + h10 * dt * out0[k] + h01 * v1[k] + h11 * dt * in1[k])
                    .collect()
            }
        }
    }

    fn sample_vec3(&self, t: f32) -> Vec3 {
        let v = self.sample(t);
        Vec3::new(v[0], v[1], v[2])
    }

    /// Rotation sampling: slerp for `Linear` (per glTF spec), nearest key for `Step`,
    /// normalized Hermite for `CubicSpline`.
    fn sample_quat(&self, t: f32) -> Quat {
        if self.input.is_empty() {
            return Quat::IDENTITY;
        }
        let q = |k: usize| {
            let v = self.value(k);
            Quat::from_xyzw(v[0], v[1], v[2], v[3])
        };
        let (i0, i1, f, _dt) = self.bracket(t);
        let result = match self.interpolation {
            Interpolation::Step => q(i0),
            Interpolation::Linear => q(i0).slerp(q(i1), f),
            Interpolation::CubicSpline => {
                let v = self.sample(t);
                Quat::from_xyzw(v[0], v[1], v[2], v[3])
            }
        };
        result.normalize()
    }
}

/// One channel of an [`Animation`]: `sampler` drives `path` on `node`.
#[derive(Clone, Debug)]
pub struct AnimChannel {
    pub node: usize,
    pub path: AnimPath,
    pub sampler: usize,
}

#[derive(Clone, Debug)]
pub struct Animation {
    pub name: String,
    pub samplers: Vec<AnimSampler>,
    pub channels: Vec<AnimChannel>,
    pub duration: f32,
}

#[derive(Debug)]
pub struct Scene {
    pub name: String,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub images: Vec<ImageData>,
    pub nodes: Vec<Node>,
    pub roots: Vec<usize>,
    pub animations: Vec<Animation>,
    pub skins: Vec<Skin>,
    /// Rest-pose world-space bounds (for camera framing).
    pub bounds: Aabb,
    pub extensions_used: Vec<String>,
}

impl Scene {
    pub fn triangle_count(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| n.mesh)
            .map(|m| self.meshes[m].primitives.iter().map(|p| p.indices.len() / 3).sum::<usize>())
            .sum()
    }
    pub fn vertex_count(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| n.mesh)
            .map(|m| self.meshes[m].primitives.iter().map(|p| p.vertices.len()).sum::<usize>())
            .sum()
    }

    /// Per-node local TRS at animation `time` (seconds): the rest pose with any
    /// channels of animation `anim` applied on top.
    pub fn local_transforms(&self, anim: Option<usize>, time: f32) -> Vec<(Vec3, Quat, Vec3)> {
        let mut trs: Vec<(Vec3, Quat, Vec3)> =
            self.nodes.iter().map(|n| (n.translation, n.rotation, n.scale)).collect();
        if let Some(a) = anim.and_then(|i| self.animations.get(i)) {
            for ch in &a.channels {
                let s = &a.samplers[ch.sampler];
                match ch.path {
                    AnimPath::Translation => trs[ch.node].0 = s.sample_vec3(time),
                    AnimPath::Rotation => trs[ch.node].1 = s.sample_quat(time),
                    AnimPath::Scale => trs[ch.node].2 = s.sample_vec3(time),
                    AnimPath::Weights => {} // morph targets — Phase 5d
                }
            }
        }
        trs
    }

    /// World matrix per node at animation `time`, walking the hierarchy from `roots`.
    pub fn node_world_matrices(&self, anim: Option<usize>, time: f32) -> Vec<Mat4> {
        let trs = self.local_transforms(anim, time);
        let mut world = vec![Mat4::IDENTITY; self.nodes.len()];
        let mut stack: Vec<(usize, Mat4)> =
            self.roots.iter().rev().map(|&r| (r, Mat4::IDENTITY)).collect();
        while let Some((n, parent)) = stack.pop() {
            let (t, r, s) = trs[n];
            let m = parent * Mat4::from_scale_rotation_translation(s, r, t);
            world[n] = m;
            for &c in &self.nodes[n].children {
                stack.push((c, m));
            }
        }
        world
    }

    /// Joint-matrix palette for `skin` given per-node world matrices: one matrix per
    /// joint = `world[joint] · inverse_bind[joint]`, mapping bind-pose mesh-space
    /// vertices to world space. Feed the result to the skinning vertex shader.
    pub fn skin_palette(&self, skin: &Skin, world: &[Mat4]) -> Vec<Mat4> {
        skin.joints
            .iter()
            .enumerate()
            .map(|(j, &node)| {
                let ibm = skin.inverse_bind.get(j).copied().unwrap_or(Mat4::IDENTITY);
                world.get(node).copied().unwrap_or(Mat4::IDENTITY) * ibm
            })
            .collect()
    }

    /// A built-in unit cube (flat normals) shown before any file is loaded.
    pub fn demo_cube() -> Self {
        let faces: [([f32; 3], [f32; 4], [[f32; 3]; 4]); 6] = [
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
            ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]),
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]),
        ];
        let uvq = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (n, t, quad) in faces {
            let base = vertices.len() as u32;
            for (k, p) in quad.iter().enumerate() {
                vertices.push(Vertex {
                    pos: *p,
                    normal: n,
                    tangent: t,
                    uv: uvq[k],
                    joints: [0; 4],
                    weights: [0.0; 4],
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let mut bounds = Aabb::empty();
        for v in &vertices {
            bounds.expand(Vec3::from_array(v.pos));
        }
        let mut mat = Material::default();
        mat.base_color_factor = [0.78, 0.80, 0.83, 1.0];
        mat.metallic_factor = 0.0;
        mat.roughness_factor = 0.55;
        Scene {
            name: "cube".into(),
            meshes: vec![Mesh { primitives: vec![Primitive { vertices, indices, material: 0 }] }],
            materials: vec![mat],
            images: Vec::new(),
            nodes: vec![Node {
                name: "cube".into(),
                parent: None,
                children: Vec::new(),
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                mesh: Some(0),
                skin: None,
                depth: 0,
            }],
            roots: vec![0],
            animations: Vec::new(),
            skins: Vec::new(),
            bounds,
            extensions_used: Vec::new(),
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Gltf(gltf::Error),
    Decode(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "io error: {e}"),
            LoadError::Gltf(e) => write!(f, "glTF error: {e}"),
            LoadError::Decode(m) => write!(f, "decode error: {m}"),
        }
    }
}
impl std::error::Error for LoadError {}
impl From<gltf::Error> for LoadError {
    fn from(e: gltf::Error) -> Self {
        LoadError::Gltf(e)
    }
}
impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

// ── Loading ───────────────────────────────────────────────────────────────────

pub fn load(path: &std::path::Path) -> Result<Scene, LoadError> {
    let bytes = std::fs::read(path)?;
    // Skip validation: gltfpack files declare EXT_meshopt_compression /
    // KHR_mesh_quantization as *required*, which the gltf crate flags as
    // "unsupported" — but we decode those ourselves below.
    let gltf = gltf::Gltf::from_slice_without_validation(&bytes)?;
    let doc = &gltf.document;
    let base = path.parent();

    // Parse raw JSON (needed to detect meshopt fallback buffers + read the
    // EXT_meshopt_compression bufferView data, neither of which the gltf crate
    // surfaces typed).
    let json_bytes: Vec<u8> = if bytes.starts_with(b"glTF") {
        gltf::Glb::from_slice(&bytes)?.json.into_owned()
    } else {
        bytes.clone()
    };
    let root: serde_json::Value = serde_json::from_slice(&json_bytes)
        .map_err(|e| LoadError::Decode(format!("json: {e}")))?;

    // Resolve buffers ourselves: meshopt *fallback* buffers have no data in the GLB
    // (they'd trip gltf::import_buffers), so allocate them as zeros to be filled by
    // the meshopt decoder.
    let mut buffers = resolve_buffers(doc, base, gltf.blob.clone(), &root)?;

    // Decode EXT_meshopt_compression bufferViews in place (reads compressed source
    // from a snapshot, writes decoded output into the fallback regions).
    if doc.extensions_used().any(|e| e == "EXT_meshopt_compression") {
        let src: Vec<gltf::buffer::Data> =
            buffers.iter().map(|v| gltf::buffer::Data(v.clone())).collect();
        decode_meshopt_bufferviews(&root, &src, &mut buffers)?;
    }

    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());

    // Map each glTF *texture* index → image index. For KHR_texture_basisu (gltfpack
    // `-tc`) the real source lives under the extension, not `textures[].source` (which
    // the gltf crate would panic on), so resolve it from raw JSON.
    let tex_to_image = build_tex_to_image(&root);

    let default_index = doc.materials().len();
    let mut materials: Vec<Material> =
        doc.materials().map(|m| convert_material(m, &tex_to_image)).collect();
    materials.push(Material::default());

    // Meshes: primitives kept in local space (node transforms applied at draw time).
    let meshes: Vec<Mesh> = doc
        .meshes()
        .map(|m| Mesh {
            primitives: m
                .primitives()
                .filter_map(|p| load_primitive(&p, &buffers, default_index))
                .collect(),
        })
        .collect();

    // Node hierarchy + the default scene's root nodes.
    let nodes = build_nodes(doc);
    let roots: Vec<usize> = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .map(|s| s.nodes().map(|n| n.index()).collect())
        .unwrap_or_default();

    let animations = parse_animations(doc, &buffers);
    let skins = parse_skins(doc, &buffers);

    let srgb_flags = srgb_usage(&materials, doc.images().count());
    let images = load_images(doc, base, &buffers, &srgb_flags)?;

    let mut scene = Scene {
        name,
        meshes,
        materials,
        images,
        nodes,
        roots,
        animations,
        skins,
        bounds: Aabb::empty(),
        extensions_used: doc.extensions_used().map(|s| s.to_string()).collect(),
    };

    scene.bounds = compute_rest_bounds(&scene);
    if !scene.bounds.is_valid() {
        scene.bounds = Aabb { min: Vec3::splat(-0.5), max: Vec3::splat(0.5) };
    }
    Ok(scene)
}

/// Find a `KHR_texture_transform` on the material (gltfpack applies the same one to
/// all textures). Returns `(offset, rotation, scale)`.
fn find_uv_transform(m: &gltf::Material) -> Option<([f32; 2], f32, [f32; 2])> {
    let pbr = m.pbr_metallic_roughness();
    let from_info = |i: Option<gltf::texture::Info>| {
        i.and_then(|x| x.texture_transform()).map(|t| (t.offset(), t.rotation(), t.scale()))
    };
    from_info(pbr.base_color_texture())
        .or_else(|| from_info(pbr.metallic_roughness_texture()))
        .or_else(|| from_info(m.emissive_texture()))
}

/// Apply a `KHR_texture_transform` (m = translate · rotate · scale) to UVs.
fn apply_uv_transform(uvs: &[[f32; 2]], offset: [f32; 2], rotation: f32, scale: [f32; 2]) -> Vec<[f32; 2]> {
    let (s, c) = rotation.sin_cos();
    uvs.iter()
        .map(|&[u, v]| {
            let (su, sv) = (u * scale[0], v * scale[1]);
            [c * su - s * sv + offset[0], s * su + c * sv + offset[1]]
        })
        .collect()
}

fn convert_material(m: gltf::Material, tex_to_image: &[usize]) -> Material {
    let pbr = m.pbr_metallic_roughness();
    // Resolve a texture index → image index (KHR_texture_basisu-aware). Never call
    // `Texture::source()` — it panics on basisu textures whose base source is absent.
    let img_of = |tex_index: usize| tex_to_image.get(tex_index).copied().unwrap_or(tex_index);
    let tref = |info: gltf::texture::Info| TexRef {
        image: img_of(info.texture().index()),
        texcoord: info.tex_coord(),
    };
    Material {
        name: m.name().unwrap_or("material").to_string(),
        base_color_factor: pbr.base_color_factor(),
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        emissive_factor: m.emissive_factor(),
        emissive_strength: m.emissive_strength().unwrap_or(1.0),
        normal_scale: m.normal_texture().map(|t| t.scale()).unwrap_or(1.0),
        occlusion_strength: m.occlusion_texture().map(|t| t.strength()).unwrap_or(1.0),
        alpha_mode: match m.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        double_sided: m.double_sided(),
        unlit: m.unlit(),
        base_color_tex: pbr.base_color_texture().map(tref),
        metallic_roughness_tex: pbr.metallic_roughness_texture().map(tref),
        normal_tex: m.normal_texture().map(|t| TexRef {
            image: img_of(t.texture().index()),
            texcoord: t.tex_coord(),
        }),
        occlusion_tex: m.occlusion_texture().map(|t| TexRef {
            image: img_of(t.texture().index()),
            texcoord: t.tex_coord(),
        }),
        emissive_tex: m.emissive_texture().map(tref),
    }
}

/// Base-color and emissive textures are sRGB-encoded; everything else is linear.
fn srgb_usage(materials: &[Material], image_count: usize) -> Vec<bool> {
    let mut srgb = vec![false; image_count];
    for m in materials {
        for t in [m.base_color_tex, m.emissive_tex].into_iter().flatten() {
            if t.image < srgb.len() {
                srgb[t.image] = true;
            }
        }
    }
    srgb
}

/// Build the node hierarchy: decomposed TRS, parent/child links, and tree depth.
fn build_nodes(doc: &gltf::Document) -> Vec<Node> {
    let count = doc.nodes().count();
    let mut nodes: Vec<Node> = doc
        .nodes()
        .map(|n| {
            let (t, r, s) = n.transform().decomposed();
            Node {
                name: n
                    .name()
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("node_{}", n.index())),
                parent: None,
                children: n.children().map(|c| c.index()).collect(),
                translation: Vec3::from_array(t),
                rotation: Quat::from_array(r),
                scale: Vec3::from_array(s),
                mesh: n.mesh().map(|m| m.index()),
                skin: n.skin().map(|s| s.index()),
                depth: 0,
            }
        })
        .collect();
    for i in 0..count {
        for c in nodes[i].children.clone() {
            if let Some(child) = nodes.get_mut(c) {
                child.parent = Some(i);
            }
        }
    }
    // Depth via a walk from parentless roots (glTF nodes form a forest).
    let mut stack: Vec<(usize, u32)> =
        (0..count).filter(|&i| nodes[i].parent.is_none()).map(|i| (i, 0)).collect();
    while let Some((n, d)) = stack.pop() {
        nodes[n].depth = d;
        for c in nodes[n].children.clone() {
            stack.push((c, d + 1));
        }
    }
    nodes
}

/// Rest-pose (default transforms) world-space AABB, for camera framing. Skinned
/// meshes are posed by their rest joint palette (the mesh node transform is ignored);
/// static meshes by their node's world matrix.
fn compute_rest_bounds(scene: &Scene) -> Aabb {
    let world = scene.node_world_matrices(None, 0.0);
    let mut b = Aabb::empty();
    for (i, node) in scene.nodes.iter().enumerate() {
        let Some(mi) = node.mesh else { continue };
        let palette = node.skin.and_then(|s| scene.skins.get(s)).map(|s| scene.skin_palette(s, &world));
        let m = world[i];
        for prim in &scene.meshes[mi].primitives {
            for v in &prim.vertices {
                let p = Vec3::from_array(v.pos);
                let world_p = match &palette {
                    Some(pal) => skin_point(p, v.joints, v.weights, pal),
                    None => m.transform_point3(p),
                };
                b.expand(world_p);
            }
        }
    }
    b
}

/// CPU mirror of the vertex-shader skinning: blend the joint matrices by weight and
/// transform the point. Used for rest-pose bounds; the GPU does this per-frame.
fn skin_point(p: Vec3, joints: [u16; 4], weights: [f32; 4], palette: &[Mat4]) -> Vec3 {
    let mut m = Mat4::ZERO;
    for k in 0..4 {
        if weights[k] == 0.0 {
            continue;
        }
        if let Some(j) = palette.get(joints[k] as usize) {
            m += *j * weights[k];
        }
    }
    // A degenerate (all-zero) blend collapses to identity so bounds stay finite.
    if m == Mat4::ZERO {
        return p;
    }
    m.transform_point3(p)
}

/// Parse all glTF skins: joint node lists + inverse bind matrices (defaulting to
/// identity when the `inverseBindMatrices` accessor is absent, per the spec).
fn parse_skins(doc: &gltf::Document, buffers: &[Vec<u8>]) -> Vec<Skin> {
    doc.skins()
        .map(|s| {
            let joints: Vec<usize> = s.joints().map(|j| j.index()).collect();
            let inverse_bind: Vec<Mat4> = match s.inverse_bind_matrices() {
                Some(acc) => read_attr(&acc, buffers, 16)
                    .chunks_exact(16)
                    .map(|c| Mat4::from_cols_array(c.try_into().unwrap()))
                    .collect(),
                None => vec![Mat4::IDENTITY; joints.len()],
            };
            Skin { joints, inverse_bind, skeleton: s.skeleton().map(|n| n.index()) }
        })
        .collect()
}

/// Parse all glTF animations into keyframe samplers + channels.
fn parse_animations(doc: &gltf::Document, buffers: &[Vec<u8>]) -> Vec<Animation> {
    use gltf::animation::{Interpolation as GInterp, Property};
    doc.animations()
        .map(|a| {
            let samplers: Vec<AnimSampler> = a
                .samplers()
                .map(|s| {
                    let out = s.output();
                    let components = match out.dimensions() {
                        gltf::accessor::Dimensions::Scalar => 1,
                        gltf::accessor::Dimensions::Vec2 => 2,
                        gltf::accessor::Dimensions::Vec3 => 3,
                        gltf::accessor::Dimensions::Vec4 => 4,
                        _ => 4,
                    };
                    AnimSampler {
                        interpolation: match s.interpolation() {
                            GInterp::Linear => Interpolation::Linear,
                            GInterp::Step => Interpolation::Step,
                            GInterp::CubicSpline => Interpolation::CubicSpline,
                        },
                        input: read_attr(&s.input(), buffers, 1),
                        output: read_attr(&out, buffers, components),
                        components,
                    }
                })
                .collect();
            let channels: Vec<AnimChannel> = a
                .channels()
                .map(|c| AnimChannel {
                    node: c.target().node().index(),
                    path: match c.target().property() {
                        Property::Translation => AnimPath::Translation,
                        Property::Rotation => AnimPath::Rotation,
                        Property::Scale => AnimPath::Scale,
                        Property::MorphTargetWeights => AnimPath::Weights,
                    },
                    sampler: c.sampler().index(),
                })
                .collect();
            let duration = samplers
                .iter()
                .map(|s| s.input.last().copied().unwrap_or(0.0))
                .fold(0.0f32, f32::max);
            Animation {
                name: a.name().unwrap_or("animation").to_string(),
                samplers,
                channels,
                duration,
            }
        })
        .collect()
}

/// Load one primitive in **local (mesh) space**. Node transforms are applied at draw
/// time, so no world/normal matrices are baked here.
fn load_primitive(
    prim: &gltf::Primitive,
    buffers: &[Vec<u8>],
    default_material: usize,
) -> Option<Primitive> {
    use gltf::Semantic;
    if prim.mode() != gltf::mesh::Mode::Triangles {
        return None;
    }

    // Positions (component-type aware → supports KHR_mesh_quantization).
    let pos_acc = prim.get(&Semantic::Positions)?;
    let positions = read_vec3(&pos_acc, buffers);
    if positions.is_empty() {
        return None;
    }
    let count = positions.len();

    let indices: Vec<u32> = match prim.indices() {
        Some(acc) => read_indices(&acc, buffers),
        None => (0..count as u32).collect(),
    };

    let normals: Vec<[f32; 3]> = match prim.get(&Semantic::Normals) {
        Some(acc) => read_vec3(&acc, buffers)
            .iter()
            .map(|n| Vec3::from_array(*n).normalize_or(Vec3::Y).to_array())
            .collect(),
        None => compute_smooth_normals(&positions, &indices),
    };

    let uvs: Vec<[f32; 2]> = match prim.get(&Semantic::TexCoords(0)) {
        Some(acc) => read_vec2(&acc, buffers),
        None => vec![[0.0, 0.0]; count],
    };
    // KHR_texture_transform: gltfpack remaps quantized UVs back via a per-texture
    // transform. Baking it recovers the original UVs (which provided tangents match).
    let uvs = match find_uv_transform(&prim.material()) {
        Some((offset, rotation, scale)) => apply_uv_transform(&uvs, offset, rotation, scale),
        None => uvs,
    };

    let tangents: Vec<[f32; 4]> = match prim.get(&Semantic::Tangents) {
        Some(acc) => read_vec4(&acc, buffers)
            .iter()
            .map(|t| {
                let v = Vec3::new(t[0], t[1], t[2]).normalize_or(Vec3::X);
                [v.x, v.y, v.z, if t[3] < 0.0 { -1.0 } else { 1.0 }]
            })
            .collect(),
        None => generate_tangents(&positions, &normals, &uvs, &indices),
    };

    let material = prim.material().index().unwrap_or(default_material);

    // Skinning influences (JOINTS_0 = integer indices; WEIGHTS_0 = float/normalized).
    // `read_attr` returns joint indices verbatim (non-normalized), so `as u16` is exact.
    let joints: Vec<[u16; 4]> = match prim.get(&Semantic::Joints(0)) {
        Some(acc) => read_attr(&acc, buffers, 4)
            .chunks_exact(4)
            .map(|c| [c[0] as u16, c[1] as u16, c[2] as u16, c[3] as u16])
            .collect(),
        None => vec![[0; 4]; count],
    };
    let weights: Vec<[f32; 4]> = match prim.get(&Semantic::Weights(0)) {
        Some(acc) => read_vec4(&acc, buffers).iter().map(|w| normalize_weights(*w)).collect(),
        None => vec![[0.0; 4]; count],
    };

    let vertices: Vec<Vertex> = (0..count)
        .map(|i| Vertex {
            pos: positions[i],
            normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            tangent: tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
            uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
            joints: joints.get(i).copied().unwrap_or([0; 4]),
            weights: weights.get(i).copied().unwrap_or([0.0; 4]),
        })
        .collect();

    Some(Primitive { vertices, indices, material })
}

// ── Buffer resolution ─────────────────────────────────────────────────────────

/// Load each glTF buffer's bytes. Unlike `gltf::import_buffers`, meshopt *fallback*
/// buffers (no data in the GLB, marked `EXT_meshopt_compression.fallback`) are
/// allocated as zeros rather than erroring.
fn resolve_buffers(
    doc: &gltf::Document,
    base: Option<&std::path::Path>,
    mut blob: Option<Vec<u8>>,
    root: &serde_json::Value,
) -> Result<Vec<Vec<u8>>, LoadError> {
    let mut out = Vec::new();
    for buffer in doc.buffers() {
        let i = buffer.index();
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                let fallback = root
                    .pointer(&format!("/buffers/{i}/extensions/EXT_meshopt_compression/fallback"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if fallback {
                    out.push(vec![0u8; buffer.length()]);
                } else {
                    out.push(
                        blob.take()
                            .ok_or_else(|| LoadError::Decode("missing GLB binary chunk".into()))?,
                    );
                }
            }
            gltf::buffer::Source::Uri(uri) => out.push(load_uri(uri, base)?),
        }
    }
    Ok(out)
}

/// Load a URI referenced by a buffer or image: a base64 `data:` URI (decoded inline,
/// as embedded `.gltf` files use) or a relative external file. Non-base64 data URIs
/// are not supported.
fn load_uri(uri: &str, base: Option<&std::path::Path>) -> Result<Vec<u8>, LoadError> {
    if let Some(rest) = uri.strip_prefix("data:") {
        // data:[<mediatype>][;base64],<payload>
        let comma = rest.find(',').ok_or_else(|| LoadError::Decode("malformed data URI".into()))?;
        if !rest[..comma].contains(";base64") {
            return Err(LoadError::Decode("non-base64 data URI not supported".into()));
        }
        use base64::Engine;
        return base64::engine::general_purpose::STANDARD
            .decode(rest[comma + 1..].as_bytes())
            .map_err(|e| LoadError::Decode(format!("data URI base64: {e}")));
    }
    let base = base.ok_or_else(|| LoadError::Decode("external URI with no base dir".into()))?;
    Ok(std::fs::read(base.join(uri))?)
}

// ── EXT_meshopt_compression + accessor reading ────────────────────────────────

/// Decode every bufferView carrying `EXT_meshopt_compression` (meshopt decode +
/// optional filter) into the reconstructed `buffers` at the bufferView's own
/// (fallback) offset. Reads the compressed source from `src` (raw buffer data).
fn decode_meshopt_bufferviews(
    root: &serde_json::Value,
    src: &[gltf::buffer::Data],
    buffers: &mut [Vec<u8>],
) -> Result<(), LoadError> {
    let Some(views) = root.get("bufferViews").and_then(|v| v.as_array()) else {
        return Ok(());
    };
    for bv in views {
        let Some(ext) = bv.pointer("/extensions/EXT_meshopt_compression") else {
            continue;
        };
        let u = |v: &serde_json::Value, k: &str, d: u64| v.get(k).and_then(|x| x.as_u64()).unwrap_or(d);
        let src_buf = u(ext, "buffer", 0) as usize;
        let src_off = u(ext, "byteOffset", 0) as usize;
        let src_len = u(ext, "byteLength", 0) as usize;
        let stride = u(ext, "byteStride", 0) as usize;
        let count = u(ext, "count", 0) as usize;
        let mode = ext.get("mode").and_then(|v| v.as_str()).unwrap_or("ATTRIBUTES");
        let filter = ext.get("filter").and_then(|v| v.as_str()).unwrap_or("NONE");
        let dst_buf = u(bv, "buffer", 0) as usize;
        let dst_off = u(bv, "byteOffset", 0) as usize;

        if stride == 0 || count == 0 {
            continue;
        }
        let src_bytes = src
            .get(src_buf)
            .and_then(|d| d.0.get(src_off..src_off + src_len))
            .ok_or_else(|| LoadError::Decode("meshopt source out of range".into()))?;
        let decoded = meshopt_decode(mode, filter, count, stride, src_bytes)
            .ok_or_else(|| LoadError::Decode(format!("meshopt decode failed (mode {mode})")))?;

        let dst = buffers
            .get_mut(dst_buf)
            .ok_or_else(|| LoadError::Decode("meshopt target buffer missing".into()))?;
        let end = dst_off + decoded.len();
        if dst.len() < end {
            dst.resize(end, 0);
        }
        dst[dst_off..end].copy_from_slice(&decoded);
    }
    Ok(())
}

fn meshopt_decode(mode: &str, filter: &str, count: usize, stride: usize, src: &[u8]) -> Option<Vec<u8>> {
    use meshopt::ffi;
    let out_len = count.checked_mul(stride)?;
    let mut out = vec![0u8; out_len];
    let rc = unsafe {
        match mode {
            "ATTRIBUTES" => ffi::meshopt_decodeVertexBuffer(out.as_mut_ptr().cast(), count, stride, src.as_ptr(), src.len()),
            "TRIANGLES" => ffi::meshopt_decodeIndexBuffer(out.as_mut_ptr().cast(), count, stride, src.as_ptr(), src.len()),
            "INDICES" => ffi::meshopt_decodeIndexSequence(out.as_mut_ptr().cast(), count, stride, src.as_ptr(), src.len()),
            _ => return None,
        }
    };
    if rc != 0 {
        return None;
    }
    if mode == "ATTRIBUTES" {
        unsafe {
            match filter {
                "OCTAHEDRAL" => ffi::meshopt_decodeFilterOct(out.as_mut_ptr().cast(), count, stride),
                "QUATERNION" => ffi::meshopt_decodeFilterQuat(out.as_mut_ptr().cast(), count, stride),
                "EXPONENTIAL" => ffi::meshopt_decodeFilterExp(out.as_mut_ptr().cast(), count, stride),
                _ => {}
            }
        }
    }
    Some(out)
}

fn comp_size(dt: gltf::accessor::DataType) -> usize {
    use gltf::accessor::DataType::*;
    match dt {
        I8 | U8 => 1,
        I16 | U16 => 2,
        U32 | F32 => 4,
    }
}

/// Read one component at `off` as f32, honoring type + normalization
/// (the core of `KHR_mesh_quantization` support).
fn read_component(buf: &[u8], off: usize, dt: gltf::accessor::DataType, normalized: bool) -> f32 {
    use gltf::accessor::DataType::*;
    let g1 = |i: usize| buf.get(off + i).copied().unwrap_or(0);
    match dt {
        F32 => f32::from_le_bytes([g1(0), g1(1), g1(2), g1(3)]),
        U32 => u32::from_le_bytes([g1(0), g1(1), g1(2), g1(3)]) as f32,
        U16 => {
            let v = u16::from_le_bytes([g1(0), g1(1)]) as f32;
            if normalized { v / 65535.0 } else { v }
        }
        I16 => {
            let v = i16::from_le_bytes([g1(0), g1(1)]) as f32;
            if normalized { (v / 32767.0).max(-1.0) } else { v }
        }
        U8 => {
            let v = g1(0) as f32;
            if normalized { v / 255.0 } else { v }
        }
        I8 => {
            let v = g1(0) as i8 as f32;
            if normalized { (v / 127.0).max(-1.0) } else { v }
        }
    }
}

/// Read a vertex attribute accessor as `count` × `comps` f32 values (flat).
fn read_attr(acc: &gltf::Accessor, buffers: &[Vec<u8>], comps: usize) -> Vec<f32> {
    let Some(view) = acc.view() else {
        return vec![0.0; acc.count() * comps];
    };
    let Some(buf) = buffers.get(view.buffer().index()) else {
        return vec![0.0; acc.count() * comps];
    };
    let dt = acc.data_type();
    let cs = comp_size(dt);
    let stride = view.stride().unwrap_or(cs * comps);
    let base = view.offset() + acc.offset();
    let norm = acc.normalized();
    let count = acc.count();
    let mut out = Vec::with_capacity(count * comps);
    for i in 0..count {
        let elem = base + i * stride;
        for c in 0..comps {
            out.push(read_component(buf, elem + c * cs, dt, norm));
        }
    }
    out
}

fn read_vec2(acc: &gltf::Accessor, buffers: &[Vec<u8>]) -> Vec<[f32; 2]> {
    read_attr(acc, buffers, 2).chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}
fn read_vec3(acc: &gltf::Accessor, buffers: &[Vec<u8>]) -> Vec<[f32; 3]> {
    read_attr(acc, buffers, 3).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
fn read_vec4(acc: &gltf::Accessor, buffers: &[Vec<u8>]) -> Vec<[f32; 4]> {
    read_attr(acc, buffers, 4).chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

/// Read an index accessor (U8/U16/U32) as u32.
fn read_indices(acc: &gltf::Accessor, buffers: &[Vec<u8>]) -> Vec<u32> {
    use gltf::accessor::DataType::*;
    let Some(view) = acc.view() else { return Vec::new() };
    let Some(buf) = buffers.get(view.buffer().index()) else { return Vec::new() };
    let dt = acc.data_type();
    let cs = comp_size(dt);
    let stride = view.stride().unwrap_or(cs);
    let base = view.offset() + acc.offset();
    let count = acc.count();
    let g = |i: usize| buf.get(i).copied().unwrap_or(0);
    (0..count)
        .map(|i| {
            let o = base + i * stride;
            match dt {
                U8 => g(o) as u32,
                U16 => u16::from_le_bytes([g(o), g(o + 1)]) as u32,
                U32 => u32::from_le_bytes([g(o), g(o + 1), g(o + 2), g(o + 3)]),
                _ => 0,
            }
        })
        .collect()
}

fn compute_smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from_array(positions[a]);
        let p1 = Vec3::from_array(positions[b]);
        let p2 = Vec3::from_array(positions[c]);
        let face = (p1 - p0).cross(p2 - p0);
        acc[a] += face;
        acc[b] += face;
        acc[c] += face;
    }
    acc.iter().map(|n| n.normalize_or(Vec3::Y).to_array()).collect()
}

/// Lengyel's method: per-vertex tangents from UV/position gradients, orthonormalized
/// against the normal, with a handedness sign in `w`.
fn generate_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let n = positions.len();
    let mut tan = vec![Vec3::ZERO; n];
    let mut bit = vec![Vec3::ZERO; n];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        let w0 = Vec2::from_array(uvs[i0]);
        let w1 = Vec2::from_array(uvs[i1]);
        let w2 = Vec2::from_array(uvs[i2]);
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let d1 = w1 - w0;
        let d2 = w2 - w0;
        let denom = d1.x * d2.y - d2.x * d1.y;
        if denom.abs() < 1e-12 {
            continue;
        }
        let r = 1.0 / denom;
        let t = (e1 * d2.y - e2 * d1.y) * r;
        let b = (e2 * d1.x - e1 * d2.x) * r;
        for &i in &[i0, i1, i2] {
            tan[i] += t;
            bit[i] += b;
        }
    }
    (0..n)
        .map(|i| {
            let normal = Vec3::from_array(normals[i]);
            let t = tan[i];
            // Gram-Schmidt orthogonalize against the normal.
            let t_ortho = (t - normal * normal.dot(t)).normalize_or(any_perpendicular(normal));
            let w = if normal.cross(t_ortho).dot(bit[i]) < 0.0 { -1.0 } else { 1.0 };
            [t_ortho.x, t_ortho.y, t_ortho.z, w]
        })
        .collect()
}

/// Renormalize the four skin weights to sum to 1 (quantized weights don't exactly),
/// falling back to full influence from the first joint when they're all ~zero.
fn normalize_weights(w: [f32; 4]) -> [f32; 4] {
    let sum = w[0] + w[1] + w[2] + w[3];
    if sum > 1e-6 {
        [w[0] / sum, w[1] / sum, w[2] / sum, w[3] / sum]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn any_perpendicular(n: Vec3) -> Vec3 {
    if n.x.abs() > 0.9 { Vec3::Y.cross(n).normalize_or(Vec3::Z) } else { Vec3::X.cross(n).normalize_or(Vec3::Z) }
}

// ── Images: KTX2/Basis transcode + PNG/JPEG decode ─────────────────────────────

/// KTX2 file identifier (first 12 bytes) — how we tell KTX2 images from PNG/JPEG.
const KTX2_MAGIC: [u8; 12] =
    [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];

/// Map glTF texture index → image index, honoring `KHR_texture_basisu` (whose
/// `.source` overrides the usually-absent `textures[].source`).
fn build_tex_to_image(root: &serde_json::Value) -> Vec<usize> {
    root.get("textures")
        .and_then(|v| v.as_array())
        .map(|txs| {
            txs.iter()
                .map(|t| {
                    t.pointer("/extensions/KHR_texture_basisu/source")
                        .and_then(|v| v.as_u64())
                        .or_else(|| t.get("source").and_then(|v| v.as_u64()))
                        .unwrap_or(0) as usize
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Decode every glTF image to a GPU-ready [`ImageData`]. KTX2 images (KHR_texture_basisu)
/// are Basis-transcoded (BC7, or RGBA8 for non-block-aligned sizes); PNG/JPEG decode to
/// RGBA8. `srgb_flags[i]` selects sRGB for PNG/JPEG image `i`; KTX2 takes its color space
/// from the container.
fn load_images(
    doc: &gltf::Document,
    base: Option<&std::path::Path>,
    buffers: &[Vec<u8>],
    srgb_flags: &[bool],
) -> Result<Vec<ImageData>, LoadError> {
    let mut out = Vec::with_capacity(doc.images().count());
    for image in doc.images() {
        let i = image.index();
        let bytes = image_source_bytes(&image, buffers, base)?;
        let img = if bytes.len() >= 12 && bytes[..12] == KTX2_MAGIC {
            transcode_ktx2(&bytes)?
        } else {
            decode_rgba8(&bytes, srgb_flags.get(i).copied().unwrap_or(false))?
        };
        out.push(img);
    }
    Ok(out)
}

/// Read an image's raw source bytes (GLB bufferView or external file URI).
fn image_source_bytes(
    image: &gltf::Image,
    buffers: &[Vec<u8>],
    base: Option<&std::path::Path>,
) -> Result<Vec<u8>, LoadError> {
    match image.source() {
        gltf::image::Source::View { view, .. } => {
            let buf = buffers
                .get(view.buffer().index())
                .ok_or_else(|| LoadError::Decode("image bufferView buffer out of range".into()))?;
            let start = view.offset();
            let end = start + view.length();
            buf.get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| LoadError::Decode("image bufferView slice out of range".into()))
        }
        gltf::image::Source::Uri { uri, .. } => load_uri(uri, base),
    }
}

/// Decode a PNG/JPEG image to tightly-packed RGBA8 (single mip).
fn decode_rgba8(bytes: &[u8], srgb: bool) -> Result<ImageData, LoadError> {
    let rgba = image::load_from_memory(bytes)
        .map_err(|e| LoadError::Decode(format!("image decode: {e}")))?
        .to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok(ImageData {
        width: w,
        height: h,
        format: TexFormat::Rgba8,
        srgb,
        mips: vec![MipLevel { width: w, height: h, data: rgba.into_raw() }],
    })
}

/// Transcode a KTX2/Basis image (ETC1S or UASTC) via libktx. Targets BC7, except for
/// images whose base dimensions aren't multiples of 4 — wgpu rejects such compressed
/// textures — which fall back to RGBA8. (BC availability is guaranteed by the required
/// device feature; see rgltf-app.) libktx sets the sRGB/linear color space from the
/// container's transfer function, which we read back from the resulting vkFormat.
fn transcode_ktx2(bytes: &[u8]) -> Result<ImageData, LoadError> {
    use libktx_rs::sources::StreamSource;
    use libktx_rs::stream::RustKtxStream;
    use libktx_rs::{Texture, TextureCreateFlags, TranscodeFlags, TranscodeFormat};
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    // KTX2 header: pixelWidth @ byte 20, pixelHeight @ 24 (LE u32). BC blocks are 4×4,
    // so only block-aligned base sizes can use BC7.
    if bytes.len() < 28 {
        return Err(LoadError::Decode("truncated KTX2 header".into()));
    }
    let base_w = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
    let base_h = u32::from_le_bytes(bytes[24..28].try_into().unwrap()).max(1);
    let use_bc = base_w % 4 == 0 && base_h % 4 == 0;

    let stream = RustKtxStream::new(Box::new(Cursor::new(bytes.to_vec())))
        .map_err(|e| LoadError::Decode(format!("ktx2 stream: {e:?}")))?;
    let src = StreamSource::new(Arc::new(Mutex::new(stream)), TextureCreateFlags::LOAD_IMAGE_DATA);
    let mut tex = Texture::new(src).map_err(|e| LoadError::Decode(format!("ktx2 open: {e:?}")))?;

    let (format, srgb) = {
        let mut k2 = tex
            .ktx2()
            .ok_or_else(|| LoadError::Decode("KHR_texture_basisu image is not KTX2".into()))?;
        if k2.needs_transcoding() {
            let target = if use_bc { TranscodeFormat::Bc7Rgba } else { TranscodeFormat::Rgba32 };
            k2.transcode_basis(target, TranscodeFlags::empty())
                .map_err(|e| LoadError::Decode(format!("ktx2 transcode: {e:?}")))?;
        }
        vk_to_texformat(k2.vk_format())?
    };

    // iterate_levels fires in descending level order — index by `level` so mips[0]=base.
    let mut levels: Vec<(u32, MipLevel)> = Vec::new();
    tex.iterate_levels(|level, _face, w, h, _d, data| {
        levels.push((
            level as u32,
            MipLevel { width: (w as u32).max(1), height: (h as u32).max(1), data: data.to_vec() },
        ));
        Ok(())
    })
    .map_err(|e| LoadError::Decode(format!("ktx2 iterate levels: {e:?}")))?;
    levels.sort_by_key(|(l, _)| *l);
    let mips: Vec<MipLevel> = levels.into_iter().map(|(_, m)| m).collect();

    let (width, height) = mips.first().map(|m| (m.width, m.height)).unwrap_or((1, 1));
    if mips.is_empty() {
        return Err(LoadError::Decode("ktx2 produced no mip levels".into()));
    }
    Ok(ImageData { width, height, format, srgb, mips })
}

/// Map a Vulkan format (as reported by libktx after transcoding) to our neutral
/// [`TexFormat`] + sRGB flag.
fn vk_to_texformat(vk: u32) -> Result<(TexFormat, bool), LoadError> {
    let r = match vk {
        37 => (TexFormat::Rgba8, false),          // R8G8B8A8_UNORM
        43 => (TexFormat::Rgba8, true),           // R8G8B8A8_SRGB
        131 | 133 => (TexFormat::Bc1Rgba, false), // BC1_RGB/RGBA_UNORM_BLOCK
        132 | 134 => (TexFormat::Bc1Rgba, true),  // BC1_RGB/RGBA_SRGB_BLOCK
        137 => (TexFormat::Bc3Rgba, false),       // BC3_UNORM_BLOCK
        138 => (TexFormat::Bc3Rgba, true),        // BC3_SRGB_BLOCK
        139 => (TexFormat::Bc4R, false),          // BC4_UNORM_BLOCK
        141 => (TexFormat::Bc5Rg, false),         // BC5_UNORM_BLOCK
        145 => (TexFormat::Bc7Rgba, false),       // BC7_UNORM_BLOCK
        146 => (TexFormat::Bc7Rgba, true),        // BC7_SRGB_BLOCK
        other => {
            return Err(LoadError::Decode(format!("unsupported transcoded vkFormat {other}")))
        }
    };
    Ok(r)
}
