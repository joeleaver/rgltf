// IBL precompute passes (split-sum, à la Karis 2013 / LearnOpenGL), run rarely
// (startup + lighting-preset change), not per frame:
//   • fs_sky       — procedural gradient environment radiance into each cube face
//   • fs_irradiance — cosine-convolved diffuse irradiance cube
//   • fs_prefilter  — GGX-prefiltered specular cube (roughness per mip)
//   • fs_brdf       — split-sum environment-BRDF LUT (scale/bias), env-independent
//
// Every pass renders a fullscreen triangle to one face (or the LUT). `uv` is the
// clip-space xy in [-1,1]; for cube faces the world direction is reconstructed from
// a per-face basis (`fwd + uv.x*right + uv.y*up`) chosen so the framebuffer texel
// (s,t) maps to exactly the direction the hardware cube lookup will later sample —
// which makes the gradient seamless across face edges.

const PI: f32 = 3.14159265359;

struct Gen {
    fwd: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    params: vec4<f32>,  // x = roughness (prefilter)
    sky: vec4<f32>,     // zenith radiance (linear)
    horizon: vec4<f32>, // horizon band radiance
    ground: vec4<f32>,  // nadir radiance
};

@group(0) @binding(0) var<uniform> gen: Gen;
// Source environment cube (irradiance + prefilter passes; unused by sky/brdf).
@group(1) @binding(0) var src_cube: texture_cube<f32>;
@group(1) @binding(1) var src_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_gen(@builtin(vertex_index) i: u32) -> VsOut {
    // Fullscreen triangle covering [-1,1]²; uv = clip xy.
    let x = f32(i32(i) / 2) * 4.0 - 1.0;
    let y = f32(i32(i) % 2) * 4.0 - 1.0;
    var o: VsOut;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

fn face_dir(uv: vec2<f32>) -> vec3<f32> {
    return normalize(gen.fwd.xyz + uv.x * gen.right.xyz + uv.y * gen.up.xyz);
}

// Analytic sky radiance for a direction: horizon band blended to a zenith colour
// above and a ground colour below. Linear HDR (env cube is Rgba16Float).
fn sky_radiance(d: vec3<f32>) -> vec3<f32> {
    let t = clamp(d.y, -1.0, 1.0);
    if (t >= 0.0) {
        return mix(gen.horizon.rgb, gen.sky.rgb, pow(t, 0.45));
    }
    return mix(gen.horizon.rgb, gen.ground.rgb, pow(-t, 0.45));
}

@fragment
fn fs_sky(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(sky_radiance(face_dir(in.uv)), 1.0);
}

// ── Diffuse irradiance: cosine-weighted convolution of the env over the hemisphere.
// Stores π·mean(L·cosθ·sinθ) so the shader can use `diffuse = irradiance * albedo`.
@fragment
fn fs_irradiance(in: VsOut) -> @location(0) vec4<f32> {
    let n = face_dir(in.uv);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let right = normalize(cross(up, n));
    let upv = cross(n, right);

    var irradiance = vec3<f32>(0.0);
    var samples = 0.0;
    let d = 0.045; // angular step (radians)
    var phi = 0.0;
    loop {
        if (phi >= 2.0 * PI) { break; }
        var theta = 0.0;
        loop {
            if (theta >= 0.5 * PI) { break; }
            // Tangent-space sample → world.
            let ts = vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            let dir = ts.x * right + ts.y * upv + ts.z * n;
            irradiance += textureSampleLevel(src_cube, src_samp, dir, 0.0).rgb
                        * cos(theta) * sin(theta);
            samples += 1.0;
            theta += d;
        }
        phi += d;
    }
    irradiance = PI * irradiance / max(samples, 1.0);
    return vec4<f32>(irradiance, 1.0);
}

// ── GGX-prefiltered specular. Hammersley + importance-sample GGX (Karis).
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 2^32
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let h = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.z) < 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let n = face_dir(in.uv);
    let v = n; // assume V = R = N
    let roughness = gen.params.x;

    let SAMPLES = 128u;
    var prefiltered = vec3<f32>(0.0);
    var total_weight = 0.0;
    for (var i = 0u; i < SAMPLES; i = i + 1u) {
        let xi = hammersley(i, SAMPLES);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let n_dot_l = dot(n, l);
        if (n_dot_l > 0.0) {
            // Env is a smooth analytic gradient (no fireflies) → sample base level.
            prefiltered += textureSampleLevel(src_cube, src_samp, l, 0.0).rgb * n_dot_l;
            total_weight += n_dot_l;
        }
    }
    return vec4<f32>(prefiltered / max(total_weight, 1e-4), 1.0);
}

// ── Split-sum environment BRDF LUT: (scale, bias) over (NdotV, roughness).
fn geometry_smith_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

@fragment
fn fs_brdf(in: VsOut) -> @location(0) vec2<f32> {
    // uv.x → NdotV (0..1 left→right); uv.y → roughness (0 top .. 1 bottom), matching
    // the fragment shader's textureSample(brdf, vec2(NdotV, roughness)).
    let n_dot_v = max(in.uv.x * 0.5 + 0.5, 1e-3);
    let roughness = 0.5 - in.uv.y * 0.5;

    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    var a = 0.0;
    var b = 0.0;
    let SAMPLES = 512u;
    for (var i = 0u; i < SAMPLES; i = i + 1u) {
        let xi = hammersley(i, SAMPLES);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let n_dot_l = max(l.z, 0.0);
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);
        if (n_dot_l > 0.0) {
            let g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / max(n_dot_h * n_dot_v, 1e-5);
            let fc = pow(1.0 - v_dot_h, 5.0);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }
    return vec2<f32>(a, b) / f32(SAMPLES);
}
