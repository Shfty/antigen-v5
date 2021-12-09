mod components;
mod systems;

use antigen_winit::AssembleWinit;
pub use components::*;
use legion::systems::CommandBuffer;
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddIndirectComponent, Construct, ImmutableSchedule, RwLock,
    Serial, Single, Usage,
};
use antigen_wgpu::{
    wgpu::{
        util::BufferInitDescriptor, BufferAddress, BufferDescriptor, BufferUsages, Device,
        Extent3d, ImageCopyTextureBase, ImageDataLayout, ShaderModuleDescriptor, ShaderSource,
        TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
    AssembleWgpu, MeshUvs, MeshVertices, RenderAttachmentTextureView,
    SurfaceConfigurationComponent,
};

use std::{borrow::Cow, num::NonZeroU32};

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

const INDICES: [u16; 36] = [
    0, 3, 2, 2, 1, 0, // top
    4, 7, 6, 6, 5, 4, // bottom
    8, 11, 10, 10, 9, 8, // right
    12, 15, 14, 14, 13, 12, // left
    16, 19, 18, 18, 17, 16, // front
    20, 23, 22, 22, 21, 20, // back
];

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

fn generate_matrix(aspect_ratio: f32) -> nalgebra::Matrix4<f32> {
    let projection = nalgebra_glm::perspective_lh_zo(aspect_ratio, 45.0, 1.0, 10.0);

    let view = nalgebra_glm::look_at_lh(
        &nalgebra::vector![1.5f32, 3.0, 5.0],
        &nalgebra::vector![0f32, 0.0, 0.0],
        &nalgebra::Vector3::y_axis(),
    );

    projection * view
}

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut CommandBuffer) {
    // Window
    let window_entity = cmd.push(());

    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Cube");

    // Renderer
    let renderer_entity = cmd.push(());

    cmd.add_component(renderer_entity, Cube);

    // Renderer resources
    cmd.assemble_wgpu_bind_group(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<OpaquePass>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<WirePass>(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);

    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Shader
    cmd.assemble_wgpu_shader(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        },
    );

    // Vertex data
    let cube_vertices = create_vertices();
    let vertex_count = cube_vertices.len();
    let vertex_size = std::mem::size_of::<[f32; 3]>();
    cmd.assemble_wgpu_buffer_data_with_usage::<Vertex, _>(
        renderer_entity,
        MeshVerticesComponent::construct(cube_vertices),
        0,
    );

    let cube_uvs = create_uvs();
    let uv_size = std::mem::size_of::<[f32; 2]>();
    let uvs_offset = (vertex_size * vertex_count) as BufferAddress;
    cmd.assemble_wgpu_buffer_data_with_usage::<Vertex, _>(
        renderer_entity,
        MeshUvsComponent::construct(cube_uvs),
        uvs_offset,
    );

    // Texture data
    let texture_size = 256u32;
    let texels = create_texels(texture_size as usize);
    let texture_extent = Extent3d {
        width: texture_size,
        height: texture_size,
        depth_or_array_layers: 1,
    };

    cmd.assemble_wgpu_texture_data_with_usage::<Mandelbrot, _>(
        renderer_entity,
        TexelsComponent::construct(texels),
        ImageCopyTextureBase {
            texture: (),
            mip_level: 0,
            origin: Default::default(),
            aspect: TextureAspect::All,
        },
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(texture_extent.width).unwrap()),
            rows_per_image: Some(NonZeroU32::new(texture_extent.height).unwrap()),
        },
    );

    // View / projection matrix uniform
    let matrix = generate_matrix(1.0);
    let mut buf: [f32; 16] = [0.0; 16];
    buf.copy_from_slice(matrix.as_slice());

    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        renderer_entity,
        ViewProjectionMatrix::construct(buf),
        0,
    );

    // Buffers
    cmd.assemble_wgpu_buffer_with_usage::<Vertex>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Vertex Buffer"),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            size: ((vertex_count * vertex_size) + (vertex_count * uv_size)) as BufferAddress,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_init_with_usage::<Index>(
        renderer_entity,
        BufferInitDescriptor {
            label: Some("Index Buffer"),
            usage: BufferUsages::INDEX,
            contents: bytemuck::cast_slice(&INDICES),
        },
    );

    cmd.assemble_wgpu_buffer_with_usage::<Uniform>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Uniform Buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            size: std::mem::size_of::<[f32; 4 * 4]>() as BufferAddress,
            mapped_at_creation: false,
        },
    );

    // Texture
    cmd.assemble_wgpu_texture_with_usage::<Mandelbrot>(
        renderer_entity,
        TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Uint,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        },
    );

    // Texture view
    cmd.assemble_wgpu_texture_view_with_usage::<Mandelbrot>(
        renderer_entity,
        renderer_entity,
        Default::default(),
    );
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_shader_modules_system(),
            antigen_wgpu::create_buffers_system::<Vertex>(),
            antigen_wgpu::create_buffers_init_system::<Index>(),
            antigen_wgpu::create_buffers_system::<Uniform>(),
            antigen_wgpu::create_textures_system::<Mandelbrot>(),
            antigen_wgpu::create_texture_views_system::<
                crate::demos::wgpu_examples::cube::Mandelbrot,
            >(),
        ],
        parallel![
            antigen_wgpu::buffer_write_system::<
                Vertex,
                Usage<MeshVertices, RwLock<Vec<[f32; 3]>>>,
                Vec<[f32; 3]>,
            >(),
            antigen_wgpu::buffer_write_system::<
                Vertex,
                Usage<MeshUvs, RwLock<Vec<[f32; 2]>>>,
                Vec<[f32; 2]>,
            >(),
            antigen_wgpu::buffer_write_system::<Uniform, ViewProjectionMatrix, [f32; 16]>(),
            antigen_wgpu::texture_write_system::<Mandelbrot, TexelsComponent, Vec<u8>>(),
        ],
        cube_prepare_system(),
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![cube_render_system()]
}
