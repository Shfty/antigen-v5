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
    [[location(0)]] end: f32;
    [[location(1)]] intensity: f32;
    [[location(2)]] delta_intensity: f32;
    [[location(3)]] delta_delta: f32;
    [[location(4)]] gradient: f32;
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
    [[location(1)]] end: f32,
    [[location(2)]] instance_pos: vec4<f32>,
    [[location(3)]] instance_prev_pos: vec4<f32>,
    [[location(4)]] intensity: f32,
    [[location(5)]] delta_intensity: f32,
    [[location(6)]] delta_delta: f32,
    [[location(7)]] gradient: f32,
) -> VertexOutput {
    var delta = instance_prev_pos.xy - instance_pos.xy;
    let delta_norm = normalize(delta);

    var angle = 0.0;
    if(length(delta_norm) > 0.0) {
        angle = atan2(delta_norm.y, delta_norm.x);
    }

    let pos = position.xy;
    let ofs = max(sign(end), 0.0) * length(delta);
    let pos = vec2<f32>(pos.x + ofs, pos.y);
    let pos = rotate(pos, angle);
    let pos = pos + instance_pos.xy;

    let pos = vec4<f32>(pos, 0.0, 1.0) * r_uniforms.projection;

    var output: VertexOutput;
    output.position = pos;
    output.end = end;
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
    // Lerp between intensity and integrated intensity based on position along line
    let intensity = mix(in.intensity, in.intensity + in.delta_intensity * r_uniforms.delta_time, in.end);

    return vec4<f32>(intensity, in.delta_intensity, in.delta_delta, in.gradient);
}
