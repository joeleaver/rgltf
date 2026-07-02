// Phase 1 shading: transform world-space geometry by the view-projection and
// apply simple hemispheric Lambert lighting so form is readable. A physically
// based pipeline replaces this in Phase 2.

struct Uniforms {
    view_proj: mat4x4<f32>,
    // xyz = world light direction (toward the light), w unused
    light_dir: vec4<f32>,
    // x = ambient strength
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(pos, 1.0);
    out.normal = normal;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(u.light_dir.xyz);
    // Two-sided so back-facing tris (no culling yet) still read.
    let ndotl = abs(dot(n, l));
    let ambient = u.params.x;
    let lit = ambient + (1.0 - ambient) * ndotl;
    return vec4<f32>(in.color * lit, 1.0);
}
