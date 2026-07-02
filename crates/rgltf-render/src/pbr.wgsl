// Phase 2 — metallic-roughness PBR with normal mapping, a single directional
// key light, and a cheap hemisphere-ambient term standing in for IBL (so metals
// aren't black). ACES tone map + sRGB encode into the Rgba8Unorm target.

const PI: f32 = 3.14159265359;

struct Frame {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    light_dir: vec4<f32>,   // xyz = direction TO the light
    light_color: vec4<f32>, // rgb * intensity
    ambient_sky: vec4<f32>,
    ambient_ground: vec4<f32>,
};

struct Material {
    base_color_factor: vec4<f32>,
    emissive: vec4<f32>,   // rgb = factor, w = emissive strength
    params: vec4<f32>,     // metallic, roughness, normal_scale, occlusion_strength
    flags: vec4<f32>,      // has: base, metallic_roughness, normal, occlusion
    flags2: vec4<f32>,     // has_emissive, alpha_cutoff, alpha_mode(0 opaque,1 mask,2 blend), unlit
};

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var<uniform> mat: Material;
@group(1) @binding(1) var t_base: texture_2d<f32>;
@group(1) @binding(2) var t_mr: texture_2d<f32>;
@group(1) @binding(3) var t_normal: texture_2d<f32>;
@group(1) @binding(4) var t_occlusion: texture_2d<f32>;
@group(1) @binding(5) var t_emissive: texture_2d<f32>;
@group(1) @binding(6) var samp: sampler;

// Skinning: the joint-matrix palette (all skins concatenated; each draw's slice starts
// at its per-instance `skin.y` base). Already world-space (world[joint]·inverseBind).
@group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    // Per-vertex skin influences (JOINTS_0 as u32, WEIGHTS_0).
    @location(11) joints: vec4<u32>,
    @location(12) weights: vec4<f32>,
    // Per-instance model matrix (columns) + normal matrix (columns; xyz used).
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) n0: vec4<f32>,
    @location(9) n1: vec4<f32>,
    @location(10) n2: vec4<f32>,
    // Per-instance skin selector: x = skinned flag, y = palette base offset.
    @location(13) skin: vec4<u32>,
) -> VsOut {
    var model = mat4x4<f32>(m0, m1, m2, m3);
    var normal_mat = mat3x3<f32>(n0.xyz, n1.xyz, n2.xyz);

    // Skinned: replace the model transform with the weighted joint-palette blend. The
    // joint matrices are already world-space, so the mesh node transform is ignored
    // (per the glTF spec). Normals use the blend's 3x3 (rigid/near-uniform → inverse-
    // transpose ≈ itself), which is standard practice for real-time skinning.
    if (skin.x == 1u) {
        let b = skin.y;
        model = weights.x * joint_matrices[b + joints.x]
              + weights.y * joint_matrices[b + joints.y]
              + weights.z * joint_matrices[b + joints.z]
              + weights.w * joint_matrices[b + joints.w];
        normal_mat = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    }

    let world = model * vec4<f32>(pos, 1.0);

    var out: VsOut;
    out.clip_pos = frame.view_proj * world;
    out.world_pos = world.xyz;
    let wn = normalize(normal_mat * normal);
    let wt = normalize((model * vec4<f32>(tangent.xyz, 0.0)).xyz);
    out.normal = wn;
    out.tangent = wt;
    out.bitangent = cross(wn, wt) * tangent.w;
    out.uv = uv;
    return out;
}

fn d_ggx(n_dot_h: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let d = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn v_smith_ggx(n_dot_v: f32, n_dot_l: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let k = a * 0.5;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn f_schlick(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

fn f_schlick_rough(cos_t: f32, f0: vec3<f32>, rough: f32) -> vec3<f32> {
    let fmax = max(vec3<f32>(1.0 - rough), f0);
    return f0 + (fmax - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    // Base color (texture is sRGB → auto-linearized on sample).
    var base = mat.base_color_factor;
    if (mat.flags.x > 0.5) {
        base = base * textureSample(t_base, samp, in.uv);
    }

    // Alpha mask.
    let alpha_mode = mat.flags2.z;
    if (alpha_mode == 1.0 && base.a < mat.flags2.y) {
        discard;
    }

    // Unlit shortcut.
    if (mat.flags2.w > 0.5) {
        return vec4<f32>(to_srgb(base.rgb), base.a);
    }

    // Geometric normal (two-sided).
    var ng = normalize(in.normal);
    if (!front) {
        ng = -ng;
    }

    // Normal mapping.
    var n = ng;
    if (mat.flags.z > 0.5) {
        let ts = textureSample(t_normal, samp, in.uv).xyz * 2.0 - 1.0;
        let scaled = vec3<f32>(ts.xy * mat.params.z, ts.z);
        let t = normalize(in.tangent - ng * dot(ng, in.tangent));
        let b = normalize(in.bitangent);
        let tbn = mat3x3<f32>(t, b, ng);
        n = normalize(tbn * scaled);
    }

    // Metallic / roughness.
    var metallic = mat.params.x;
    var rough = mat.params.y;
    if (mat.flags.y > 0.5) {
        let mr = textureSample(t_mr, samp, in.uv);
        rough = rough * mr.g;
        metallic = metallic * mr.b;
    }
    rough = clamp(rough, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    // Occlusion.
    var ao = 1.0;
    if (mat.flags.w > 0.5) {
        let occ = textureSample(t_occlusion, samp, in.uv).r;
        ao = 1.0 + mat.params.w * (occ - 1.0);
    }

    let albedo = base.rgb;
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let diffuse_color = albedo * (1.0 - metallic);

    let v = normalize(frame.cam_pos.xyz - in.world_pos);
    let n_dot_v = max(dot(n, v), 1e-4);

    // Directional key light.
    var lo = vec3<f32>(0.0);
    {
        let l = normalize(frame.light_dir.xyz);
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l > 0.0) {
            let n_dot_h = max(dot(n, h), 0.0);
            let v_dot_h = max(dot(v, h), 0.0);
            let d = d_ggx(n_dot_h, rough);
            let vis = v_smith_ggx(n_dot_v, n_dot_l, rough);
            let f = f_schlick(v_dot_h, f0);
            let spec = d * vis * f / max(4.0 * n_dot_v * n_dot_l, 1e-5);
            let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
            lo = (kd * diffuse_color / PI + spec) * frame.light_color.rgb * n_dot_l;
        }
    }

    // Hemisphere ambient (cheap IBL stand-in).
    let sky = frame.ambient_sky.rgb;
    let ground = frame.ambient_ground.rgb;
    let diff_irr = mix(ground, sky, clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let r = reflect(-v, n);
    let spec_irr = mix(ground, sky, clamp(r.y * 0.5 + 0.5, 0.0, 1.0));
    let f_amb = f_schlick_rough(n_dot_v, f0, rough);
    let ambient = (diff_irr * diffuse_color * (vec3<f32>(1.0) - f_amb) + spec_irr * f_amb) * ao;

    // Emissive.
    var emissive = mat.emissive.rgb * mat.emissive.w;
    if (mat.flags2.x > 0.5) {
        emissive = emissive * textureSample(t_emissive, samp, in.uv).rgb;
    }

    let color = lo + ambient + emissive;
    return vec4<f32>(to_srgb(aces(color)), base.a);
}
