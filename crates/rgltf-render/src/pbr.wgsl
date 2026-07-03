// Metallic-roughness PBR with normal mapping, a directional key light (the "sun"),
// and split-sum image-based lighting (irradiance + prefiltered specular + BRDF LUT,
// group 3) for the ambient/reflection term. ACES tone map + sRGB encode into the
// Rgba8Unorm target. Also hosts the analytic skybox (vs_sky/fs_sky).

const PI: f32 = 3.14159265359;

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    light_dir: vec4<f32>,   // xyz = direction TO the light
    light_color: vec4<f32>, // rgb * intensity
    env_sky: vec4<f32>,     // environment zenith radiance (linear)
    env_horizon: vec4<f32>, // environment horizon-band radiance
    env_ground: vec4<f32>,  // environment nadir radiance
    flags: vec4<f32>,       // x = use material textures, y = prefilter mip count, z = IBL intensity
};

struct Material {
    base_color_factor: vec4<f32>,
    emissive: vec4<f32>,   // rgb = factor, w = emissive strength
    params: vec4<f32>,     // metallic, roughness, normal_scale, occlusion_strength
    flags: vec4<f32>,      // has: base, metallic_roughness, normal, occlusion
    flags2: vec4<f32>,     // has_emissive, alpha_cutoff, alpha_mode(0 opaque,1 mask,2 blend), unlit
    // Extensions:
    specular: vec4<f32>,       // KHR_materials_specular factor, KHR_materials_ior, _, _
    specular_color: vec4<f32>, // specular colour factor rgb, _
    clearcoat: vec4<f32>,      // KHR_materials_clearcoat factor, roughness, normal_scale, _
    sheen: vec4<f32>,          // KHR_materials_sheen colour rgb, w = roughness
    ext_flags: vec4<f32>,      // has: specular, specular_color, clearcoat, clearcoat_roughness
    ext_flags2: vec4<f32>,     // has: clearcoat_normal, sheen_color, sheen_roughness, _
};

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var<uniform> mat: Material;
@group(1) @binding(1) var t_base: texture_2d<f32>;
@group(1) @binding(2) var t_mr: texture_2d<f32>;
@group(1) @binding(3) var t_normal: texture_2d<f32>;
@group(1) @binding(4) var t_occlusion: texture_2d<f32>;
@group(1) @binding(5) var t_emissive: texture_2d<f32>;
@group(1) @binding(6) var samp: sampler;
@group(1) @binding(7) var t_specular: texture_2d<f32>;
@group(1) @binding(8) var t_specular_color: texture_2d<f32>;
@group(1) @binding(9) var t_clearcoat: texture_2d<f32>;
@group(1) @binding(10) var t_clearcoat_rough: texture_2d<f32>;
@group(1) @binding(11) var t_clearcoat_normal: texture_2d<f32>;
@group(1) @binding(12) var t_sheen_color: texture_2d<f32>;
@group(1) @binding(13) var t_sheen_rough: texture_2d<f32>;

// Skinning: the joint-matrix palette (all skins concatenated; each draw's slice starts
// at its per-instance `skin.y` base). Already world-space (world[joint]·inverseBind).
@group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;

// Image-based lighting (baked from the current environment; see ibl.rs).
@group(3) @binding(0) var irr_cube: texture_cube<f32>;   // diffuse irradiance
@group(3) @binding(1) var pre_cube: texture_cube<f32>;   // prefiltered specular (roughness mips)
@group(3) @binding(2) var brdf_lut: texture_2d<f32>;     // split-sum env-BRDF (scale, bias)
@group(3) @binding(3) var cube_samp: sampler;
@group(3) @binding(4) var lut_samp: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) uv: vec2<f32>,
    // Bone-weight debug colour (Σ weightₖ · colour(jointₖ)); grey when un-skinned.
    @location(5) weight_color: vec3<f32>,
};

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let p = abs(fract(vec3<f32>(h) + vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

// A distinct colour per (local) joint index, spaced by the golden ratio for good spread.
fn joint_color(j: u32) -> vec3<f32> {
    return hsv2rgb(fract(f32(j) * 0.6180339887), 0.65, 0.95);
}

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

    // Bone-weight visualisation colour: blend the influencing joints' colours by weight.
    let wsum = weights.x + weights.y + weights.z + weights.w;
    if (skin.x == 1u && wsum > 0.001) {
        out.weight_color = weights.x * joint_color(joints.x)
                         + weights.y * joint_color(joints.y)
                         + weights.z * joint_color(joints.z)
                         + weights.w * joint_color(joints.w);
    } else {
        out.weight_color = vec3<f32>(0.12); // un-skinned → dark grey
    }
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

// Scalar Schlick for the clearcoat coat (fixed F0 = 0.04, IOR 1.5 lacquer).
fn f_schlick_scalar(cos_t: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

// KHR_materials_sheen: the "Charlie" sheen NDF + Ashikhmin visibility (retroreflective
// cloth lobe).
fn d_charlie(n_dot_h: f32, rough: f32) -> f32 {
    let a = max(rough, 0.07);
    let inv = 1.0 / a;
    let sin2 = max(1.0 - n_dot_h * n_dot_h, 0.0);
    return (2.0 + inv) * pow(sin2, inv * 0.5) / (2.0 * PI);
}

fn v_ashikhmin(n_dot_l: f32, n_dot_v: f32) -> f32 {
    return 1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), 1e-4);
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
    // Bone-weight debug view (flat, unlit): short-circuit before any shading.
    if (frame.flags.w > 0.5) {
        return vec4<f32>(to_srgb(in.weight_color), 1.0);
    }

    // Global toggle: when off, ignore all material textures (show factors only).
    let use_tex = frame.flags.x > 0.5;

    // Base color (texture is sRGB → auto-linearized on sample).
    var base = mat.base_color_factor;
    if (mat.flags.x > 0.5 && use_tex) {
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
    if (mat.flags.z > 0.5 && use_tex) {
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
    if (mat.flags.y > 0.5 && use_tex) {
        let mr = textureSample(t_mr, samp, in.uv);
        rough = rough * mr.g;
        metallic = metallic * mr.b;
    }
    rough = clamp(rough, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    // Occlusion.
    var ao = 1.0;
    if (mat.flags.w > 0.5 && use_tex) {
        let occ = textureSample(t_occlusion, samp, in.uv).r;
        ao = 1.0 + mat.params.w * (occ - 1.0);
    }

    let albedo = base.rgb;

    // KHR_materials_ior + KHR_materials_specular: dielectric F0 from the IOR, tinted by
    // the specular colour; the specular weight scales the whole dielectric specular lobe
    // (applied to both direct + IBL below — metals are unaffected). Defaults (ior 1.5,
    // colour 1, weight 1) reproduce plain F0 = 0.04.
    let ior = mat.specular.y;
    let f0_ior = pow((ior - 1.0) / (ior + 1.0), 2.0);
    var spec_color = mat.specular_color.rgb;
    if (mat.ext_flags.y > 0.5 && use_tex) {
        spec_color = spec_color * textureSample(t_specular_color, samp, in.uv).rgb;
    }
    var spec_weight = mat.specular.x;
    if (mat.ext_flags.x > 0.5 && use_tex) {
        spec_weight = spec_weight * textureSample(t_specular, samp, in.uv).a;
    }
    let dielectric_f0 = min(vec3<f32>(f0_ior) * spec_color, vec3<f32>(1.0));
    let f0 = mix(dielectric_f0, albedo, metallic);
    // Weight applied to the specular lobe: dielectrics scaled by spec_weight, metals by 1.
    let spec_lobe_weight = mix(spec_weight, 1.0, metallic);
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
            // KHR_materials_specular weight scales the whole dielectric specular lobe.
            let spec = d * vis * f / max(4.0 * n_dot_v * n_dot_l, 1e-5) * spec_lobe_weight;
            let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
            lo = (kd * diffuse_color / PI + spec) * frame.light_color.rgb * n_dot_l;
        }
    }

    // Image-based lighting (split-sum): diffuse irradiance + prefiltered specular,
    // combined with the precomputed environment-BRDF LUT.
    let f_amb = f_schlick_rough(n_dot_v, f0, rough);
    let kd = (vec3<f32>(1.0) - f_amb) * (1.0 - metallic);
    let irradiance = textureSample(irr_cube, cube_samp, n).rgb;
    let diffuse_ibl = irradiance * albedo;
    let refl = reflect(-v, n);
    let max_mip = max(frame.flags.y - 1.0, 0.0);
    let prefiltered = textureSampleLevel(pre_cube, cube_samp, refl, rough * max_mip).rgb;
    let env_brdf = textureSample(brdf_lut, lut_samp, vec2<f32>(n_dot_v, rough)).rg;
    // KHR_materials_specular weight also scales the env specular (metals keep 1).
    let specular_ibl = prefiltered * (f0 * env_brdf.x + env_brdf.y) * spec_lobe_weight;
    let ambient = (kd * diffuse_ibl + specular_ibl) * ao * frame.flags.z;

    // Base material lit result (direct key light + IBL). Sheen and clearcoat layer over it.
    var lit = lo + ambient;

    // ── KHR_materials_sheen: a retroreflective cloth lobe added over the base. Gated on
    // the (uniform) sheen colour/flag so textureSample stays in uniform control flow.
    if (max(mat.sheen.r, max(mat.sheen.g, mat.sheen.b)) > 0.0 || mat.ext_flags2.y > 0.5) {
        var sheen_color = mat.sheen.rgb;
        if (mat.ext_flags2.y > 0.5 && use_tex) {
            sheen_color = sheen_color * textureSample(t_sheen_color, samp, in.uv).rgb;
        }
        var sheen_rough = mat.sheen.w;
        if (mat.ext_flags2.z > 0.5 && use_tex) {
            sheen_rough = sheen_rough * textureSample(t_sheen_rough, samp, in.uv).a;
        }
        sheen_rough = clamp(sheen_rough, 0.07, 1.0);
        let l = normalize(frame.light_dir.xyz);
        let h = normalize(v + l);
        let ndl = max(dot(n, l), 0.0);
        var sheen = vec3<f32>(0.0);
        if (ndl > 0.0) {
            let ndh = max(dot(n, h), 0.0);
            sheen = sheen_color * d_charlie(ndh, sheen_rough) * v_ashikhmin(ndl, n_dot_v)
                  * frame.light_color.rgb * ndl;
        }
        // Cheap IBL sheen: the prefiltered environment tinted by the sheen colour.
        let sheen_env = textureSampleLevel(pre_cube, cube_samp, refl, sheen_rough * max_mip).rgb;
        lit = lit + sheen + sheen_color * sheen_env * ao * frame.flags.z;
    }

    // ── KHR_materials_clearcoat: a clear-lacquer specular layer over everything (base +
    // sheen). Its reflectance attenuates the layers beneath. Gated on the uniform factor.
    if (mat.clearcoat.x > 0.0) {
        var cc = mat.clearcoat.x;
        if (mat.ext_flags.z > 0.5 && use_tex) {
            cc = cc * textureSample(t_clearcoat, samp, in.uv).r;
        }
        var cc_rough = mat.clearcoat.y;
        if (mat.ext_flags.w > 0.5 && use_tex) {
            cc_rough = cc_rough * textureSample(t_clearcoat_rough, samp, in.uv).g;
        }
        cc_rough = clamp(cc_rough, 0.04, 1.0);
        // The coat has its own normal (its map, else the geometric normal — the coat is
        // smooth over the geometry and ignores the base normal map).
        var cc_n = ng;
        if (mat.ext_flags2.x > 0.5 && use_tex) {
            let ts = textureSample(t_clearcoat_normal, samp, in.uv).xyz * 2.0 - 1.0;
            let scaled = vec3<f32>(ts.xy * mat.clearcoat.z, ts.z);
            let t = normalize(in.tangent - ng * dot(ng, in.tangent));
            let b = normalize(in.bitangent);
            cc_n = normalize(mat3x3<f32>(t, b, ng) * scaled);
        }
        let cc_ndv = max(dot(cc_n, v), 1e-4);
        let l = normalize(frame.light_dir.xyz);
        let h = normalize(v + l);
        let cc_ndl = max(dot(cc_n, l), 0.0);
        var cc_direct = vec3<f32>(0.0);
        if (cc_ndl > 0.0) {
            let cc_ndh = max(dot(cc_n, h), 0.0);
            let cc_vdh = max(dot(v, h), 0.0);
            let dcc = d_ggx(cc_ndh, cc_rough);
            let vcc = v_smith_ggx(cc_ndv, cc_ndl, cc_rough);
            let fcc = f_schlick_scalar(cc_vdh, 0.04);
            cc_direct = vec3<f32>(dcc * vcc * fcc / max(4.0 * cc_ndv * cc_ndl, 1e-5))
                      * frame.light_color.rgb * cc_ndl;
        }
        // Coat IBL reflection (F0 = 0.04).
        let cc_refl = reflect(-v, cc_n);
        let cc_env = textureSampleLevel(pre_cube, cube_samp, cc_refl, cc_rough * max_mip).rgb;
        let cc_ibl = cc_env * f_schlick_scalar(cc_ndv, 0.04) * ao * frame.flags.z;
        // Attenuate the layers beneath by the coat's reflectance at its own view angle.
        let cc_atten = cc * f_schlick_scalar(cc_ndv, 0.04);
        lit = lit * (1.0 - cc_atten) + cc * (cc_direct + cc_ibl);
    }

    // Emissive (over the coat — emitters aren't dimmed by the clearcoat).
    var emissive = mat.emissive.rgb * mat.emissive.w;
    if (mat.flags2.x > 0.5 && use_tex) {
        emissive = emissive * textureSample(t_emissive, samp, in.uv).rgb;
    }

    let color = lit + emissive;
    return vec4<f32>(to_srgb(aces(color)), base.a);
}

// ── Skybox ────────────────────────────────────────────────────────────────────
// The analytic environment behind the model. Drawn first each frame with depth
// writes off, so meshes overwrite it. The direction is reconstructed by unprojecting
// the far-plane NDC through `inv_view_proj`. Uses the same gradient as the baked env
// cube (ibl.wgsl `sky_radiance`) so the background matches the reflections/lighting.

struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    let x = f32(i32(i) / 2) * 4.0 - 1.0;
    let y = f32(i32(i) % 2) * 4.0 - 1.0;
    var o: SkyOut;
    o.pos = vec4<f32>(x, y, 1.0, 1.0);
    o.ndc = vec2<f32>(x, y);
    return o;
}

fn sky_gradient(d: vec3<f32>) -> vec3<f32> {
    let t = clamp(d.y, -1.0, 1.0);
    if (t >= 0.0) {
        return mix(frame.env_horizon.rgb, frame.env_sky.rgb, pow(t, 0.45));
    }
    return mix(frame.env_horizon.rgb, frame.env_ground.rgb, pow(-t, 0.45));
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let clip = vec4<f32>(in.ndc, 1.0, 1.0);
    let world = frame.inv_view_proj * clip;
    let dir = normalize(world.xyz / world.w - frame.cam_pos.xyz);
    return vec4<f32>(to_srgb(aces(sky_gradient(dir))), 1.0);
}

// ── Skeleton overlay ──────────────────────────────────────────────────────────
// Bone segments (joint→parent) as world-space lines. Positions come from a per-frame
// vertex buffer built from the animated node world matrices (see lib.rs). Drawn with
// depth test disabled so the whole skeleton is visible through the mesh.

@vertex
fn vs_bone(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return frame.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_bone() -> @location(0) vec4<f32> {
    return vec4<f32>(0.35, 1.0, 0.55, 1.0); // bright green (offscreen sRGB space)
}
