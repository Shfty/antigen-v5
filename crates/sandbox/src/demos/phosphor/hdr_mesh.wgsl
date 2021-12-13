let PI: f32 = 3.14159265359;

[[block]]
struct Uniforms {
    total_time: f32;
    delta_time: f32;
    projection: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> r_uniforms: Uniforms;

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] intensity: f32;
    [[location(1)]] delta_intensity: f32;
    [[location(2)]] delta_delta: f32;
    [[location(3)]] gradient: f32;
};

[[stage(vertex)]]
fn vs_main(
    [[builtin(vertex_index)]] v_index: u32,
    [[location(0)]] position: vec4<f32>,
    [[location(1)]] intensity: f32,
    [[location(2)]] delta_intensity: f32,
    [[location(3)]] delta_delta: f32,
    [[location(4)]] gradient: f32,
) -> VertexOutput {
    var output: VertexOutput;
    output.position = r_uniforms.projection * position;
    output.intensity = intensity;
    output.delta_intensity = delta_intensity;
    output.delta_delta = delta_delta;
    output.gradient = gradient;
    return output;
}

[[stage(fragment)]]
fn fs_main(
    in: VertexOutput,
) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(in.intensity, in.delta_intensity, in.delta_delta, in.gradient);
}
