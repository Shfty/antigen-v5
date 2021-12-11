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
    [[location(0)]] local_pos: vec2<f32>;
    [[location(1)]] intensity: f32;
    [[location(2)]] delta_intensity: f32;
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
    [[builtin(vertex_index)]] v_index: u32,
    [[location(0)]] position: vec4<f32>,
    [[location(1)]] intensity: f32,
    [[location(2)]] delta_intensity: f32,
    [[location(3)]] gradient: f32,
    [[location(4)]] instance_pos: vec4<f32>,
    [[location(5)]] instance_prev_pos: vec4<f32>,
) -> VertexOutput {
    let delta = instance_prev_pos.xy - instance_pos.xy;
    let delta_norm = normalize(delta);
    let angle = atan2(delta_norm.y, delta_norm.x);

    let pos = position.xy;
    let ofs = max(sign(f32(v_index) - 3.5), 0.0) * length(delta);
    let pos = vec2<f32>(pos.x + ofs, pos.y);
    let pos = rotate(pos, angle);
    let pos = pos + instance_pos.xy;

    let pos = vec4<f32>(pos, 0.0, 1.0) * r_uniforms.projection;

    var output: VertexOutput;
    output.position = pos;
    output.local_pos = position.xy;
    output.intensity = intensity;
    output.delta_intensity = delta_intensity;
    output.gradient = gradient;
    return output;
}

[[stage(fragment)]]
fn fs_main(
    in: VertexOutput,
) -> [[location(0)]] vec4<f32> {
    if(length(in.local_pos) > 1.0) {
        discard;
    }

    return vec4<f32>(in.intensity, in.delta_intensity, in.gradient, 1.0);
}
