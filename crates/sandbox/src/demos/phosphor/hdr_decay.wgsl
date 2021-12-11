[[block]]
struct Time {
    total: f32;
    delta: f32;
};

[[group(0), binding(0)]]
var<uniform> r_time: Time;

[[group(0), binding(2)]]
var r_sampler: sampler;

[[group(1), binding(0)]]
var r_hdr: texture_2d<f32>;

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
    let hdr = textureSample(r_hdr, r_sampler, in.uv);

    // Unpack HDR fragment
    let intensity = hdr.r;
    let delta_intensity = hdr.g;
    let delta_delta = hdr.b;
    let gradient = hdr.a;

    // Integrate intensity
    let prev_intensity = intensity;
    let intensity = intensity + delta_intensity * r_time.delta;
    let intensity_changed = abs(sign(sign(intensity) + sign(prev_intensity)));

    // Integrate delta intensity
    let prev_delta = delta_intensity;
    let delta_intensity = delta_intensity + delta_delta * r_time.delta;
    let delta_changed = abs(sign(sign(delta_intensity) + sign(prev_delta)));

    // Zero out intensity if it changes sign
    let intensity = intensity * intensity_changed;

    // Zero out delta intensity if intensity changes sign
    let delta_intensity = delta_intensity * intensity_changed;

    // Zero out delta intensity and delta delta if delta intensity changes sign
    let delta_intensity = delta_intensity * delta_changed;
    let delta_delta = delta_delta * delta_changed;

    return vec4<f32>(intensity, delta_intensity, delta_delta, gradient);
}
