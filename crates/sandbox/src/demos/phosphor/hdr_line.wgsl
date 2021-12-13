let PI: f32 = 3.14159265359;

[[block]]
struct Uniforms {
    total_time: f32;
    delta_time: f32;
    projection: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> r_uniforms: Uniforms;

struct VertexInput {
    [[builtin(vertex_index)]] v_index: u32;
    [[location(0)]] position: vec4<f32>;
    [[location(1)]] end: f32;
    [[location(2)]] v0: vec4<f32>;
    [[location(3)]] v0_intensity: f32;
    [[location(4)]] v0_delta_intensity: f32;
    [[location(5)]] v0_delta_delta: f32;
    [[location(6)]] v0_gradient: f32;
    [[location(7)]] v1: vec4<f32>;
    [[location(8)]] v1_intensity: f32;
    [[location(9)]] v1_delta_intensity: f32;
    [[location(10)]] v1_delta_delta: f32;
    [[location(11)]] v1_gradient: f32;
};

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] intensity: f32;
    [[location(1)]] delta_intensity: f32;
    [[location(2)]] delta_delta: f32;
    [[location(3)]] gradient: f32;
};

fn rotate(vec: vec2<f32>, angle: f32) -> vec2<f32> {
    let cs = cos(angle);
    let sn = sin(angle);
    return vec2<f32>(
        vec.x * cs - vec.y * sn,
        vec.x * sn + vec.y * cs,
    );
}

[[stage(vertex)]]
fn vs_main(
    in: VertexInput
) -> VertexOutput {
    var delta = in.v1.xy - in.v0.xy;
    let delta_norm = normalize(delta);

    var angle = 0.0;
    if(length(delta_norm) > 0.0) {
        angle = atan2(delta_norm.y, delta_norm.x);
    }

    let pos = in.position;
    let ofs = max(sign(in.end), 0.0) * length(delta);
    let pos = vec4<f32>(pos.x + ofs, pos.y, pos.z, pos.w);
    let pos = vec4<f32>(rotate(pos.xy, angle), pos.z, pos.w);
    let pos = vec4<f32>(pos.xyz + in.v0.xyz, pos.w);

    let pos = r_uniforms.projection * pos;

    var output: VertexOutput;
    output.position = pos;
    output.intensity = mix(in.v0_intensity, in.v1_intensity, in.end);
    output.delta_intensity = mix(in.v0_delta_intensity, in.v1_delta_intensity, in.end);
    output.delta_delta = mix(in.v0_delta_delta, in.v1_delta_delta, in.end);
    output.gradient = mix(in.v0_gradient, in.v1_gradient, in.end);
    return output;
}

[[stage(fragment)]]
fn fs_main(
    in: VertexOutput,
) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(in.intensity, in.delta_intensity, in.delta_delta, in.gradient);
}
