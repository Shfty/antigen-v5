mod components;
mod systems;

pub use components::*;
pub use systems::*;

use crate::{
    BufferComponent, CommandBuffersComponent, DirtyFlag, IndirectComponent, MeshIndices, MeshUvs,
    MeshVertices, RenderPipelineComponent, SurfaceComponent, TextureComponent,
    TextureViewComponent,
};

pub enum Opaque {}
pub enum Wire {}

pub enum Vertex {}
pub enum Index {}
pub enum Uniform {}

fn create_vertices() -> Vec<[f32; 3]> {
    vec![
        // top (0, 0, 1)
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        // bottom (0, 0, -1)
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        // right (1, 0, 0)
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        // left (-1, 0, 0)
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0],
        // front (0, 1, 0)
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // back (0, -1, 0)
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
    ]
}

fn create_uvs() -> Vec<[f32; 2]> {
    vec![
        // top (0, 0, 1)
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        // bottom (0, 0, -1)
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        // right (1, 0, 0)
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        // left (-1, 0, 0)
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        // front (0, 1, 0)
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        // back (0, -1, 0)
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
}

fn create_indices() -> Vec<u16> {
    vec![
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ]
}

fn create_texels(size: usize) -> Vec<u8> {
    (0..size * size)
        .map(|id| {
            // get high five for recognizing this ;)
            let cx = 3.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let cy = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let (mut x, mut y, mut count) = (cx, cy, 0);
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            count
        })
        .collect()
}

fn generate_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
    let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
    let mx_view = cgmath::Matrix4::look_at_rh(
        cgmath::Point3::new(1.5f32, -5.0, 3.0),
        cgmath::Point3::new(0f32, 0.0, 0.0),
        cgmath::Vector3::unit_z(),
    );
    let mx_correction = OPENGL_TO_WGPU_MATRIX;
    mx_correction * mx_projection * mx_view
}

#[legion::system]
#[read_component(wgpu::Device)]
pub fn assemble(
    cmd: &mut legion::systems::CommandBuffer,
    #[state] (entity, window_entity): &(legion::Entity, legion::Entity),
) {
    cmd.add_component(*entity, Cube);
    cmd.add_component(*entity, RenderPipelineComponent::<Opaque>::pending());
    cmd.add_component(*entity, RenderPipelineComponent::<Wire>::pending());
    cmd.add_component(*entity, CommandBuffersComponent::new());
    cmd.add_component(
        *entity,
        IndirectComponent::<SurfaceComponent>::foreign(*window_entity),
    );
    cmd.add_component(
        *entity,
        IndirectComponent::<TextureViewComponent>::foreign(*window_entity),
    );

    let cube_vertices = create_vertices();
    let cube_uvs = create_uvs();
    let cube_indices = create_indices();

    let vertex_count = cube_vertices.len();
    let index_count = cube_indices.len();

    // Cube Mesh
    cmd.add_component(*entity, MeshVertices::new(cube_vertices));
    cmd.add_component(*entity, DirtyFlag::<MeshVertices<[f32; 3]>>::new_dirty());

    cmd.add_component(*entity, MeshUvs::new(cube_uvs));
    cmd.add_component(*entity, DirtyFlag::<MeshUvs<[f32; 2]>>::new_dirty());

    cmd.add_component(*entity, MeshIndices::new(cube_indices));
    cmd.add_component(*entity, DirtyFlag::<MeshIndices<u16>>::new_dirty());

    // Mandelbrot texture
    let texture_size = 256u32;
    let texels = create_texels(texture_size as usize);
    let texture_extent = wgpu::Extent3d {
        width: texture_size,
        height: texture_size,
        depth_or_array_layers: 1,
    };

    cmd.add_component(
        *entity,
        TextureComponent::pending(
            wgpu::TextureDescriptor {
                label: None,
                size: texture_extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            },
            texels,
        ),
    );

    cmd.add_component(
        *entity,
        TextureViewComponent::pending(wgpu::TextureViewDescriptor::default()),
    );

    // Camera matrix
    let mx_total = generate_matrix(100 as f32 / 100 as f32);
    let mx_ref: [f32; 16] = *mx_total.as_ref();
    cmd.add_component(*entity, mx_ref);

    // Buffers
    cmd.add_component(
        *entity,
        BufferComponent::<Vertex>::pending(wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            usage: wgpu::BufferUsages::VERTEX,
            size: vertex_count as u64,
            mapped_at_creation: false,
        }),
    );

    cmd.add_component(
        *entity,
        BufferComponent::<Index>::pending(wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            usage: wgpu::BufferUsages::INDEX,
            size: index_count as u64,
            mapped_at_creation: false,
        }),
    );

    cmd.add_component(
        *entity,
        BufferComponent::<Index>::pending(wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            usage: wgpu::BufferUsages::UNIFORM,
            size: std::mem::size_of::<[f32; 4 * 4]>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        }),
    );
}
