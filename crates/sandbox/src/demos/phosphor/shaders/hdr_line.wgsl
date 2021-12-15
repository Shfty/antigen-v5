let PI: f32 = 3.14159265359;

[[block]]
struct Uniforms {
    perspective: mat4x4<f32>;
    orthographic: mat4x4<f32>;
    total_time: f32;
    delta_time: f32;
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
    let v0 = r_uniforms.perspective * in.v0;
    let v1 = r_uniforms.perspective * in.v1;

    let v0 = v0.xyz / v0.w;
    let v1 = v1.xyz / v1.w;

    var delta = v1 - v0;

    let delta_norm = normalize(delta);

    var angle = 0.0;
    if(length(delta_norm) > 0.0) {
        angle = atan2(delta_norm.y, delta_norm.x);
    }

    let vert = in.position.xy;
    let vert = rotate(vert.xy, angle);
    let vert = (r_uniforms.orthographic * vec4<f32>(vert.xy, 0.0, 1.0)).xy;

    let pos = vec3<f32>(vert, 0.0) + mix(v0, v1, in.end);

    var output: VertexOutput;
    output.position = vec4<f32>(pos, 1.0);
    output.intensity = mix(in.v0_intensity, in.v1_intensity, in.end);
    output.delta_intensity = mix(in.v0_delta_intensity, in.v1_delta_intensity, in.end);
    output.delta_delta = mix(in.v0_delta_delta, in.v1_delta_delta, in.end);
    output.gradient = mix(in.v0_gradient, in.v1_gradient, in.end);
    return output;
}

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

[[stage(fragment)]]
fn fs_main(
    in: VertexOutput,
) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = vec4<f32>(in.delta_intensity, in.delta_delta, in.gradient, in.intensity);
    return out;
}
