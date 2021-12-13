[[group(0), binding(1)]]
var r_gradients: texture_2d<f32>;

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

    let intensity = hdr.r + 0.5;
    let gradient = hdr.a + 0.5;

    let grad_size = vec2<f32>(textureDimensions(r_gradients));
    let u = intensity / grad_size.x;
    let v = gradient / grad_size.y;

    let color = textureSample(r_gradients, r_sampler, vec2<f32>(u, v));
    return color;
}
