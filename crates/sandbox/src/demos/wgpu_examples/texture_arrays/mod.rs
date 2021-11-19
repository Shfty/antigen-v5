mod components;
mod systems;

use std::num::NonZeroU32;

use antigen_winit::AssembleWinit;
pub use components::*;
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddComponentWithUsage, AddIndirectComponent, ImmutableSchedule,
    LazyComponent, RwLock, Serial, Single,
};

use antigen_wgpu::{
    wgpu::{
        include_spirv_raw, BufferAddress, BufferDescriptor, BufferUsages, Device, Extent3d,
        ImageCopyTextureBase, ImageDataLayout, IndexFormat, TextureAspect,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
    AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent,
};

use bytemuck::{Pod, Zeroable};

const INDEX_FORMAT: IndexFormat = IndexFormat::Uint16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    _pos: [f32; 2],
    _tex_coord: [f32; 2],
    _index: u32,
}

fn vertex(pos: [i8; 2], tc: [i8; 2], index: i8) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
        _index: index as u32,
    }
}

fn create_vertices() -> Vec<Vertex> {
    vec![
        // left rectangle
        vertex([-1, -1], [0, 1], 0),
        vertex([-1, 1], [0, 0], 0),
        vertex([0, 1], [1, 0], 0),
        vertex([0, -1], [1, 1], 0),
        // right rectangle
        vertex([0, -1], [0, 1], 1),
        vertex([0, 1], [0, 0], 1),
        vertex([1, 1], [1, 0], 1),
        vertex([1, -1], [1, 1], 1),
    ]
}

fn create_indices() -> Vec<u16> {
    vec![
        // Left rectangle
        0, 1, 2, // 1st
        2, 0, 3, // 2nd
        // Right rectangle
        4, 5, 6, // 1st
        6, 4, 7, // 2nd
    ]
}

#[derive(Copy, Clone)]
enum Color {
    Red,
    Green,
}

fn create_texture_data(color: Color) -> [u8; 4] {
    match color {
        Color::Red => [255, 0, 0, 255],
        Color::Green => [0, 255, 0, 255],
    }
}

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Texture Arrays");

    // Renderer
    cmd.add_component(renderer_entity, TextureArrays);
    cmd.assemble_wgpu_render_pipeline(renderer_entity);
    cmd.assemble_wgpu_bind_group(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Vertex shader
    cmd.assemble_wgpu_shader_spirv_with_usage::<Vertex>(
        renderer_entity,
        include_spirv_raw!("shader.vert.spv"),
    );

    // Fragment shader is initialized in the prepare system
    cmd.add_component(
        renderer_entity,
        FragmentShaderComponent::new(RwLock::new(LazyComponent::Pending)),
    );

    // Uniform workaround flag
    cmd.add_component_with_usage::<UniformWorkaround>(renderer_entity, RwLock::new(false));

    // Buffer data
    let vertex_data = create_vertices();
    let vertex_count = vertex_data.len();
    cmd.assemble_wgpu_buffer_data_with_usage::<Vertex, _>(
        renderer_entity,
        RwLock::new(vertex_data),
        0,
    );

    let index_data = create_indices();
    let index_count = index_data.len();
    cmd.assemble_wgpu_buffer_data_with_usage::<Index, _>(
        renderer_entity,
        RwLock::new(index_data),
        0,
    );

    // Buffers
    cmd.assemble_wgpu_buffer_with_usage::<Vertex>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (std::mem::size_of::<Vertex>() * vertex_count) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_with_usage::<Index>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Index Buffer"),
            size: (std::mem::size_of::<Index>() * index_count) as BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Texture data
    let red_texture_data = create_texture_data(Color::Red);
    cmd.assemble_wgpu_texture_data_with_usage::<Red, _>(
        renderer_entity,
        RwLock::new(red_texture_data),
        ImageCopyTextureBase {
            texture: (),
            mip_level: 0,
            origin: Default::default(),
            aspect: TextureAspect::All,
        },
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(4).unwrap()),
            rows_per_image: None,
        },
    );

    let green_texture_data = create_texture_data(Color::Green);
    cmd.assemble_wgpu_texture_data_with_usage::<Green, _>(
        renderer_entity,
        RwLock::new(green_texture_data),
        ImageCopyTextureBase {
            texture: (),
            mip_level: 0,
            origin: Default::default(),
            aspect: TextureAspect::All,
        },
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(4).unwrap()),
            rows_per_image: None,
        },
    );

    // Textures
    let texture_descriptor = TextureDescriptor {
        size: Extent3d::default(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        label: None,
    };

    cmd.assemble_wgpu_texture_with_usage::<Red>(renderer_entity, texture_descriptor.clone());
    cmd.assemble_wgpu_texture_with_usage::<Green>(renderer_entity, texture_descriptor.clone());

    // Texture views
    cmd.assemble_wgpu_texture_view_with_usage::<Red>(renderer_entity, Default::default());
    cmd.assemble_wgpu_texture_view_with_usage::<Green>(renderer_entity, Default::default());

    // Texture sampler
    cmd.assemble_wgpu_sampler(renderer_entity, Default::default());
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_shader_modules_usage_spirv_system::<Vertex>(),
            antigen_wgpu::create_buffers_system::<Vertex>(),
            antigen_wgpu::create_buffers_system::<Index>(),
            antigen_wgpu::create_textures_system::<Red>(),
            antigen_wgpu::create_textures_system::<Green>(),
            antigen_wgpu::create_texture_views_system::<Red>(),
            antigen_wgpu::create_texture_views_system::<Green>(),
            antigen_wgpu::create_samplers_system(),
        ],
        parallel![
            antigen_wgpu::buffer_write_system::<Vertex, RwLock<Vec<Vertex>>, Vec<Vertex>>(),
            antigen_wgpu::buffer_write_system::<Index, RwLock<Vec<Index>>, Vec<Index>>(),
            antigen_wgpu::texture_write_system::<Red, RwLock<[u8; 4]>, [u8; 4]>(),
            antigen_wgpu::texture_write_system::<Green, RwLock<[u8; 4]>, [u8; 4]>(),
        ],
        texture_arrays_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![texture_arrays_render_system()]
}
