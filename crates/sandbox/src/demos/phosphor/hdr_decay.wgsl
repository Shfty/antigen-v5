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
    let intensity = hdr.r;
    let delta_intensity = hdr.g;
    let intensity = max(intensity + delta_intensity * r_time.delta, 0.0);
    let gradient = hdr.b;
    return vec4<f32>(intensity, delta_intensity, gradient, 1.0);
}
