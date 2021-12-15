[[block]]
struct Uniforms {
    perspective: mat4x4<f32>;
    orthographic: mat4x4<f32>;
    total: f32;
    delta: f32;
};

[[group(0), binding(0)]]
var<uniform> r_uniforms: Uniforms;

[[group(0), binding(2)]]
var r_linear_sampler: sampler;

[[group(1), binding(0)]]
var r_back_buffer: texture_2d<f32>;

[[group(1), binding(1)]]
var r_beam_buffer: texture_2d<f32>;

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] uv: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] vertex_index: u32) -> VertexOutput {
    let x: f32 = f32(i32(vertex_index & 1u) << 2u) - 1.0;
    let y: f32 = f32(i32(vertex_index & 2u) << 1u) - 1.0;
    var output: VertexOutput;
    output.position = vec4<f32>(x, -y, 0.0, 1.0);
    output.uv = vec2<f32>(x + 1.0, y + 1.0) * 0.5;
    return output;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let back = textureSample(r_back_buffer, r_linear_sampler, in.uv);

    // Unpack beam fragment
    let delta_intensity = back.r;
    let delta_delta = back.g;
    let gradient = back.b;
    let intensity = back.a;

    // Integrate intensity
    let prev_intensity = intensity;
    let intensity = clamp(intensity + delta_intensity * r_uniforms.delta, 0.0, 8.0);
    let intensity_changed = abs(sign(sign(intensity) + sign(prev_intensity)));

    // Integrate delta intensity
    let prev_delta = delta_intensity;
    let delta_intensity = delta_intensity + delta_delta * r_uniforms.delta;
    let delta_changed = abs(sign(sign(delta_intensity) + sign(prev_delta)));

    // Zero out intensity if it changes sign
    let intensity = intensity * intensity_changed;

    // Zero out delta intensity if intensity changes sign
    let delta_intensity = delta_intensity * intensity_changed;

    // Zero out delta intensity and delta delta if delta intensity changes sign
    let delta_intensity = delta_intensity * delta_changed;
    let delta_delta = delta_delta * delta_changed;

    let beam = textureSample(r_beam_buffer, r_linear_sampler, in.uv);

    let delta_intensity = beam.r;
    let delta_delta = beam.g;
    let gradient = beam.b;
    let intensity = max(intensity, beam.a);

    return vec4<f32>(delta_intensity, delta_delta, gradient, intensity);
}
