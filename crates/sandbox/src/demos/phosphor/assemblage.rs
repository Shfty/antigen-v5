use std::num::NonZeroU32;

use antigen_core::Construct;
use antigen_wgpu::{
    buffer_size_of,
    wgpu::{
        BufferAddress, Extent3d, ImageCopyTextureBase, ImageDataLayout, TextureAspect,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
        COPY_BUFFER_ALIGNMENT,
    },
    AssembleWgpu,
};
use legion::Entity;

use crate::phosphor::{LineIndex, LineIndexDataComponent, OriginComponent};

use super::{
    MeshIndex, MeshIndexDataComponent, MeshVertex, MeshVertexData, MeshVertexDataComponent,
    Oscilloscope, BLACK, BLUE, GREEN, RED, WHITE,
};

pub fn assemble_oscilloscope(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_head: &mut BufferAddress,
    index_head: &mut BufferAddress,
    origin: (f32, f32, f32),
    color: (f32, f32, f32),
    osc: Oscilloscope,
    intensity: f32,
    delta_intensity: f32,
) {
    let entity = cmd.push(());
    cmd.add_component(entity, OriginComponent::construct(origin));
    cmd.add_component(entity, osc);

    let vertices = vec![
        MeshVertexData {
            position: [0.0, 0.0, 0.0],
            surface_color: [color.0, color.1, color.2],
            line_color: [color.0, color.1, color.2],
            intensity,
            delta_intensity,
            ..Default::default()
        },
        MeshVertexData {
            position: [0.0, 0.0, 0.0],
            surface_color: [color.0, color.1, color.2],
            line_color: [color.0, color.1, color.2],
            intensity,
            delta_intensity,
            ..Default::default()
        },
    ];
    cmd.assemble_wgpu_buffer_data_with_usage::<MeshVertex, _>(
        entity,
        MeshVertexDataComponent::construct(vertices),
        buffer_size_of::<MeshVertexData>() * *vertex_head as BufferAddress,
        Some(buffer_target),
    );

    let indices = vec![(*vertex_head as u32), (*vertex_head + 1) as u32];
    println!("Ocilloscope indices: {:#?}", indices);

    cmd.assemble_wgpu_buffer_data_with_usage::<LineIndex, _>(
        entity,
        LineIndexDataComponent::construct(indices),
        buffer_size_of::<u32>() * *index_head as BufferAddress,
        Some(buffer_target),
    );

    *vertex_head += 2;
    *index_head += 2;
}

pub fn assemble_box_bot(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_head: &mut BufferAddress,
    mesh_index_head: &mut BufferAddress,
    line_index_head: &mut BufferAddress,
    (x, y, z): (f32, f32, f32),
) {
    // Cube lines
    assemble_line_strip(
        cmd,
        buffer_target,
        vertex_head,
        line_index_head,
        vec![
            MeshVertexData::new((-25.0, -25.0, -25.0), RED, RED, 2.0, -30.0),
            MeshVertexData::new((25.0, -25.0, -25.0), GREEN, GREEN, 2.0, -30.0),
            MeshVertexData::new((25.0, -25.0, 25.0), BLUE, GREEN, 2.0, -30.0),
            MeshVertexData::new((-25.0, -25.0, 25.0), WHITE, WHITE, 2.0, -30.0),
            MeshVertexData::new((-25.0, -25.0, -25.0), RED, RED, 2.0, -30.0),
        ]
        .into_iter()
        .map(|mut v| {
            v.position[0] += x;
            v.position[1] += y;
            v.position[2] += z;
            v
        })
        .collect(),
    );

    assemble_line_strip(
        cmd,
        buffer_target,
        vertex_head,
        line_index_head,
        vec![
            MeshVertexData::new((-25.0, 25.0, -25.0), RED, RED, 2.0, -30.0),
            MeshVertexData::new((25.0, 25.0, -25.0), GREEN, RED, 2.0, -30.0),
            MeshVertexData::new((25.0, 25.0, 25.0), BLUE, RED, 2.0, -30.0),
            MeshVertexData::new((-25.0, 25.0, 25.0), WHITE, RED, 2.0, -30.0),
            MeshVertexData::new((-25.0, 25.0, -25.0), BLACK, RED, 2.0, -30.0),
        ]
        .into_iter()
        .map(|mut v| {
            v.position[0] += x;
            v.position[1] += y;
            v.position[2] += z;
            v
        })
        .collect(),
    );

    assemble_line_list(
        cmd,
        buffer_target,
        vertex_head,
        line_index_head,
        vec![
            MeshVertexData::new((-25.0, -25.0, -25.0), RED, RED, 2.0, -30.0),
            MeshVertexData::new((-25.0, 25.0, -25.0), RED, RED, 2.0, -30.0),
            MeshVertexData::new((25.0, -25.0, -25.0), GREEN, GREEN, 2.0, -30.0),
            MeshVertexData::new((25.0, 25.0, -25.0), GREEN, GREEN, 2.0, -30.0),
            MeshVertexData::new((25.0, -25.0, 25.0), BLUE, BLUE, 2.0, -30.0),
            MeshVertexData::new((25.0, 25.0, 25.0), BLUE, BLUE, 2.0, -30.0),
            MeshVertexData::new((-25.0, -25.0, 25.0), WHITE, WHITE, 2.0, -30.0),
            MeshVertexData::new((-25.0, 25.0, 25.0), WHITE, WHITE, 2.0, -30.0),
        ]
        .into_iter()
        .map(|mut v| {
            v.position[0] += x;
            v.position[1] += y;
            v.position[2] += z;
            v
        })
        .collect(),
    );

    // Body cube
    assemble_mesh(
        cmd,
        buffer_target,
        vertex_head,
        mesh_index_head,
        vec![
            MeshVertexData::new((1.0, 1.0, 1.0), BLACK, BLACK, 0.0, -16.0),
            MeshVertexData::new((-1.0, 1.0, 1.0), BLACK, BLACK, 0.0, -16.0),
            MeshVertexData::new((-1.0, 1.0, -1.0), BLACK, BLACK, 0.0, -16.0),
            MeshVertexData::new((1.0, 1.0, -1.0), BLACK, BLACK, 0.0, -16.0),
            MeshVertexData::new((1.0, -1.0, 1.0), BLACK, BLACK, 0.0, -16.0),
            MeshVertexData::new((-1.0, -1.0, 1.0), BLACK, BLACK, 0.0, -16.0),
            MeshVertexData::new((-1.0, -1.0, -1.0), BLACK, BLACK, 0.0, -16.0),
            MeshVertexData::new((1.0, -1.0, -1.0), BLACK, BLACK, 0.0, -16.0),
        ]
        .into_iter()
        .map(|mut vd| {
            vd.position[0] *= 25.0;
            vd.position[1] *= 25.0;
            vd.position[2] *= 25.0;
            vd.position[0] += x;
            vd.position[1] += y;
            vd.position[2] += z;
            vd
        })
        .collect(),
        vec![
            // Top
            0, 1, 2, 0, 2, 3, // Bottom
            4, 7, 5, 7, 6, 5, // Front
            3, 2, 6, 3, 6, 7, // Back
            0, 5, 1, 0, 4, 5, // Right
            0, 3, 7, 0, 7, 4, // Left
            1, 5, 6, 1, 6, 2,
        ]
        .into_iter()
        .map(|id| id + (*vertex_head) as u16)
        .collect(),
    );

    // Visor cube
    assemble_mesh(
        cmd,
        buffer_target,
        vertex_head,
        mesh_index_head,
        vec![
            MeshVertexData::new((1.0, 1.0, 1.0), RED, RED, 2.0, -14.0),
            MeshVertexData::new((-1.0, 1.0, 1.0), RED, RED, 2.0, -14.0),
            MeshVertexData::new((-1.0, 1.0, -1.0), RED, RED, 2.0, -14.0),
            MeshVertexData::new((1.0, 1.0, -1.0), RED, RED, 2.0, -14.0),
            MeshVertexData::new((1.0, -1.0, 1.0), RED, RED, 2.0, -14.0),
            MeshVertexData::new((-1.0, -1.0, 1.0), RED, RED, 2.0, -14.0),
            MeshVertexData::new((-1.0, -1.0, -1.0), RED, RED, 2.0, -14.0),
            MeshVertexData::new((1.0, -1.0, -1.0), RED, RED, 2.0, -14.0),
        ]
        .into_iter()
        .map(|mut vd| {
            vd.position[0] *= 10.0;
            vd.position[1] *= 2.5;
            vd.position[2] *= 2.5;
            vd.position[2] -= 25.0;
            vd.position[0] += x;
            vd.position[1] += y;
            vd.position[2] += z;
            vd
        })
        .collect(),
        vec![
            // Top
            0, 1, 2, 0, 2, 3, // Bottom
            4, 7, 5, 7, 6, 5, // Front
            3, 2, 6, 3, 6, 7, // Back
            0, 5, 1, 0, 4, 5, // Right
            0, 3, 7, 0, 7, 4, // Left
            1, 5, 6, 1, 6, 2,
        ]
        .into_iter()
        .map(|id| id + (*vertex_head as u16))
        .collect(),
    );
}

pub fn assemble_lines(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_head: &mut BufferAddress,
    index_head: &mut BufferAddress,
    vertices: Vec<MeshVertexData>,
    indices: Vec<u32>,
) {
    let entity = cmd.push(());
    let vertex_count = vertices.len();
    let index_count = indices.len();
    cmd.assemble_wgpu_buffer_data_with_usage::<MeshVertex, _>(
        entity,
        MeshVertexDataComponent::construct(vertices),
        buffer_size_of::<MeshVertexData>() * *vertex_head as BufferAddress,
        Some(buffer_target),
    );
    cmd.assemble_wgpu_buffer_data_with_usage::<LineIndex, _>(
        entity,
        LineIndexDataComponent::construct(indices),
        buffer_size_of::<u32>() * *index_head as BufferAddress,
        Some(buffer_target),
    );

    *vertex_head += vertex_count as BufferAddress;
    *index_head += index_count as BufferAddress;
}

pub fn assemble_line_list(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_head: &mut BufferAddress,
    index_head: &mut BufferAddress,
    vertices: Vec<MeshVertexData>,
) {
    let mut vs = *vertex_head as u32;
    let indices = vertices
        .chunks(2)
        .flat_map(|_| {
            let ret = [vs, vs + 1];
            vs += 2;
            ret
        })
        .collect::<Vec<_>>();

    println!("Line list indices: {:#?}", indices);

    assemble_lines(
        cmd,
        buffer_target,
        vertex_head,
        index_head,
        vertices,
        indices,
    )
}

pub fn assemble_line_indices(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    line_index_head: &mut BufferAddress,
    indices: Vec<u32>,
) {
    let entity = cmd.push(());
    let index_count = indices.len();
    cmd.assemble_wgpu_buffer_data_with_usage::<LineIndex, _>(
        entity,
        LineIndexDataComponent::construct(indices),
        buffer_size_of::<u32>() * *line_index_head as BufferAddress,
        Some(buffer_target),
    );

    *line_index_head += index_count as BufferAddress;
}

pub fn assemble_line_strip(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_head: &mut BufferAddress,
    index_head: &mut BufferAddress,
    vertices: Vec<MeshVertexData>,
) {
    let mut indices =
        (*vertex_head..(*vertex_head + vertices.len() as BufferAddress)).collect::<Vec<_>>();

    let first = indices.remove(0) as u32;
    let last = indices.pop().unwrap() as u32;
    let inter = indices.into_iter().flat_map(|i| [i as u32, i as u32]);
    let indices = std::iter::once(first)
        .chain(inter)
        .chain(std::iter::once(last))
        .collect();

    println!("Line strip indices: {:#?}", indices);

    assemble_lines(
        cmd,
        buffer_target,
        vertex_head,
        index_head,
        vertices,
        indices,
    )
}

pub fn pad_align_triangle_list(indices: &mut Vec<u16>) {
    while (buffer_size_of::<u16>() * indices.len() as BufferAddress) % COPY_BUFFER_ALIGNMENT > 0 {
        indices.extend(std::iter::repeat(0).take(3));
    }
}

pub fn assemble_mesh(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_buffer_index: &mut BufferAddress,
    index_buffer_index: &mut BufferAddress,
    vertices: Vec<MeshVertexData>,
    mut indices: Vec<u16>,
) {
    let entity = cmd.push(());
    let vertex_offset = buffer_size_of::<MeshVertexData>() * *vertex_buffer_index;
    let index_offset = buffer_size_of::<u16>() * *index_buffer_index;

    pad_align_triangle_list(&mut indices);

    let vertex_count = vertices.len();
    let index_count = indices.len();

    println!("Index count: {}", index_count);
    println!("Index offset: {}", index_offset);

    cmd.assemble_wgpu_buffer_data_with_usage::<MeshVertex, _>(
        entity,
        MeshVertexDataComponent::construct(vertices),
        vertex_offset,
        Some(buffer_target),
    );

    cmd.assemble_wgpu_buffer_data_with_usage::<MeshIndex, _>(
        entity,
        MeshIndexDataComponent::construct(indices),
        index_offset,
        Some(buffer_target),
    );

    *vertex_buffer_index += vertex_count as BufferAddress;
    *index_buffer_index += index_count as BufferAddress;
}

pub fn assemble_triangle_list(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    vertex_buffer_index: &mut BufferAddress,
    index_buffer_index: &mut BufferAddress,
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
    vertex_buffer_index: &mut BufferAddress,
    index_buffer_index: &mut BufferAddress,
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

pub fn assemble_png_texture_with_usage<C, U, I>(
    cmd: &mut legion::systems::CommandBuffer,
    renderer_entity: Entity,
    label: Option<&'static str>,
    png_bytes: &[u8],
) where
    C: Construct<Vec<u8>, I> + Send + Sync + 'static,
    U: Send + Sync + 'static,
{
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

    cmd.assemble_wgpu_texture_with_usage::<U>(
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

    cmd.assemble_wgpu_texture_data_with_usage::<U, _>(
        renderer_entity,
        C::construct(buf),
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

    cmd.assemble_wgpu_texture_view_with_usage::<U>(
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
