use std::num::NonZeroU32;

use antigen_core::Construct;
use antigen_wgpu::{AssembleWgpu, buffer_size_of, wgpu::{BufferAddress, Extent3d, ImageCopyTextureBase, ImageDataLayout, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor}};
use legion::Entity;

use crate::phosphor::{GradientDataComponent, Gradients, LineInstanceData, LineInstanceDataComponent, OriginComponent};

use super::{
    LineInstance, MeshIndex, MeshIndexDataComponent, MeshVertex, MeshVertexData,
    MeshVertexDataComponent, Oscilloscope,
};

pub fn assemble_oscilloscope(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    buffer_index: usize,
    origin: (f32, f32, f32),
    osc: Oscilloscope,
    intensity: f32,
    delta_intensity: f32,
    delta_delta: f32,
    gradient: f32,
) {
    let entity = cmd.push(());
    cmd.add_component(entity, OriginComponent::construct(origin));
    cmd.add_component(entity, osc);
    cmd.assemble_wgpu_buffer_data_with_usage::<LineInstance, _>(
        entity,
        LineInstanceDataComponent::construct(vec![LineInstanceData {
            v0: MeshVertexData {
                position: [0.0, 0.0, 0.0, 1.0],
                intensity,
                delta_intensity,
                delta_delta,
                gradient,
            },
            v1: MeshVertexData {
                position: [0.0, 0.0, 0.0, 1.0],
                intensity,
                delta_intensity,
                delta_delta,
                gradient,
            },
        }]),
        buffer_size_of::<LineInstanceData>() * buffer_index as BufferAddress,
        Some(buffer_target),
    );
}

pub fn assemble_lines(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    buffer_index: usize,
    lines: Vec<LineInstanceData>
) {
    let entity = cmd.push(());
    cmd.assemble_wgpu_buffer_data_with_usage::<LineInstance, _>(
        entity,
        LineInstanceDataComponent::construct(lines),
        buffer_size_of::<LineInstanceData>() * buffer_index as BufferAddress,
        Some(buffer_target),
    );
}

pub fn assemble_line_list(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    buffer_index: usize,
    vertices: Vec<MeshVertexData>,
) {
    let lines = vertices
        .chunks(2)
        .map(|vs| LineInstanceData {
            v0: vs[0],
            v1: vs[1],
        })
        .collect::<Vec<_>>();

    assemble_lines(cmd, buffer_target, buffer_index, lines)
}

pub fn assemble_line_strip(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    buffer_index: usize,
    mut vertices: Vec<MeshVertexData>,
) {
    let first = vertices.remove(0);
    let last = vertices.pop().unwrap();
    let inter = vertices.into_iter().flat_map(|v| [v, v]);
    let vertices = std::iter::once(first)
        .chain(inter)
        .chain(std::iter::once(last))
        .collect();

    assemble_line_list(cmd, buffer_target, buffer_index, vertices)
}

pub fn assemble_mesh(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_buffer_index: BufferAddress,
    index_buffer_index: BufferAddress,
    vertices: Vec<MeshVertexData>,
    indices: Vec<u16>,
) {
    let entity = cmd.push(());
    cmd.assemble_wgpu_buffer_data_with_usage::<MeshVertex, _>(
        entity,
        MeshVertexDataComponent::construct(vertices),
        buffer_size_of::<MeshVertexData>() * vertex_buffer_index,
        Some(buffer_target),
    );
    cmd.assemble_wgpu_buffer_data_with_usage::<MeshIndex, _>(
        entity,
        MeshIndexDataComponent::construct(indices),
        buffer_size_of::<u16>() * index_buffer_index,
        Some(buffer_target),
    );
}

pub fn assemble_triangle_list(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_buffer_index: BufferAddress,
    index_buffer_index: BufferAddress,
    mut base_index: u16,
    vertices: Vec<MeshVertexData>,
) {
    let indices = vertices
        .chunks(3)
        .flat_map(|_| {
            let is = [base_index, base_index + 1, base_index + 2];
            base_index += 3;
            is
        })
        .collect::<Vec<_>>();

    assemble_mesh(
        cmd,
        buffer_target,
        vertex_buffer_index,
        index_buffer_index,
        vertices,
        indices,
    );
}

pub fn assemble_triangle_fan(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_buffer_index: BufferAddress,
    index_buffer_index: BufferAddress,
    base_index: u16,
    vertices: Vec<MeshVertexData>,
) {
    let mut current_index = base_index;
    let indices = (0..vertices.len() - 2)
        .flat_map(|_| {
            let is = [base_index, current_index + 1, current_index + 2];
            current_index += 1;
            is
        })
        .collect::<Vec<_>>();

    assemble_mesh(
        cmd,
        buffer_target,
        vertex_buffer_index,
        index_buffer_index,
        vertices,
        indices,
    );
}

pub fn assemble_png_texture(
    cmd: &mut legion::systems::CommandBuffer,
    renderer_entity: Entity,
    label: Option<&'static str>,
    png_bytes: &[u8],
) {
    // Gradients texture
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let size = Extent3d {
        width: info.width,
        height: info.height,
        depth_or_array_layers: 1,
    };

    cmd.assemble_wgpu_texture_with_usage::<Gradients>(
        renderer_entity,
        TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        },
    );

    cmd.assemble_wgpu_texture_data_with_usage::<Gradients, _>(
        renderer_entity,
        GradientDataComponent::construct(buf),
        ImageCopyTextureBase {
            texture: (),
            mip_level: 0,
            origin: Default::default(),
            aspect: TextureAspect::All,
        },
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(info.line_size as u32).unwrap()),
            rows_per_image: Some(NonZeroU32::new(size.height).unwrap()),
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<Gradients>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label,
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );
}
