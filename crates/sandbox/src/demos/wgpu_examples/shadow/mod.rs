mod components;
mod systems;

use std::{num::NonZeroU32, ops::Range};

use antigen_winit::{AssembleWinit, RedrawUnconditionally};
pub use components::*;
use legion::Entity;
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddIndirectComponent, AsUsage, ImmutableSchedule, LazyComponent,
    RwLock, Serial, Single,
};

use antigen_wgpu::{AssembleWgpu, BufferComponent, RenderAttachmentTextureView, SurfaceConfigurationComponent, TextureViewComponent, wgpu::{
        AddressMode, BufferAddress, BufferDescriptor, BufferUsages, Color, CompareFunction, Device,
        Extent3d, FilterMode, IndexFormat, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource,
        TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        TextureViewDescriptor, TextureViewDimension,
    }};

use bytemuck::{Pod, Zeroable};

type Index = u16;

const MAX_LIGHTS: usize = 10;
const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;
const SHADOW_FORMAT: TextureFormat = TextureFormat::Depth32Float;
const SHADOW_SIZE: Extent3d = Extent3d {
    width: 512,
    height: 512,
    depth_or_array_layers: MAX_LIGHTS as u32,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightRaw {
    proj: [[f32; 4]; 4],
    pos: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalUniforms {
    proj: [[f32; 4]; 4],
    num_lights: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectUniforms {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

fn create_cube() -> (Vec<Vertex>, Vec<Index>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0, 1]),
        vertex([1, -1, 1], [0, 0, 1]),
        vertex([1, 1, 1], [0, 0, 1]),
        vertex([-1, 1, 1], [0, 0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [0, 0, -1]),
        vertex([1, 1, -1], [0, 0, -1]),
        vertex([1, -1, -1], [0, 0, -1]),
        vertex([-1, -1, -1], [0, 0, -1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [1, 0, 0]),
        vertex([1, 1, -1], [1, 0, 0]),
        vertex([1, 1, 1], [1, 0, 0]),
        vertex([1, -1, 1], [1, 0, 0]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [-1, 0, 0]),
        vertex([-1, 1, 1], [-1, 0, 0]),
        vertex([-1, 1, -1], [-1, 0, 0]),
        vertex([-1, -1, -1], [-1, 0, 0]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [0, 1, 0]),
        vertex([-1, 1, -1], [0, 1, 0]),
        vertex([-1, 1, 1], [0, 1, 0]),
        vertex([1, 1, 1], [0, 1, 0]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, -1, 0]),
        vertex([-1, -1, 1], [0, -1, 0]),
        vertex([-1, -1, -1], [0, -1, 0]),
        vertex([1, -1, -1], [0, -1, 0]),
    ];

    let index_data: &[Index] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

fn create_plane(size: i8) -> (Vec<Vertex>, Vec<Index>) {
    let vertex_data = [
        vertex([size, -size, 0], [0, 0, 1]),
        vertex([size, size, 0], [0, 0, 1]),
        vertex([-size, -size, 0], [0, 0, 1]),
        vertex([-size, size, 0], [0, 0, 1]),
    ];

    let index_data: &[Index] = &[0, 1, 2, 2, 1, 3];

    (vertex_data.to_vec(), index_data.to_vec())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [i8; 4],
    _normal: [i8; 4],
}

fn vertex(pos: [i8; 3], nor: [i8; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1],
        _normal: [nor[0], nor[1], nor[2], 0],
    }
}

pub fn assemble_object(
    cmd: &mut legion::systems::CommandBuffer,
    entity: Entity,
    mesh: Mesh,
    mx_world: nalgebra::Matrix4<f32>,
    rotation_speed: f32,
    color: Color,
    uniform_offset: u32,
) {
    cmd.add_component(entity, mesh);
    cmd.add_component(entity, ObjectTag::as_usage(RwLock::new(mx_world)));
    cmd.add_component(entity, RotationSpeed::as_usage(rotation_speed));
    cmd.add_component(entity, color);
    cmd.add_component(entity, UniformOffset::as_usage(uniform_offset));
}

pub fn assemble_light(
    cmd: &mut legion::systems::CommandBuffer,
    entity: Entity,
    position: nalgebra::Vector3<f32>,
    color: Color,
    fov: f32,
    depth: Range<f32>,
    shadow_texture_entity: Entity,
    base_array_layer: u32,
) {
    cmd.add_component(entity, position);
    cmd.add_component(entity, color);
    cmd.add_component(entity, FieldOfView::as_usage(fov));
    cmd.add_component(entity, depth);
    cmd.assemble_wgpu_texture_view_with_usage::<ShadowPass>(
        entity,
        shadow_texture_entity,
        TextureViewDescriptor {
            label: Some("shadow"),
            format: None,
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer,
            array_layer_count: NonZeroU32::new(1),
        },
    );
}

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Redraw the window unconditionally
    cmd.add_component(window_entity, RedrawUnconditionally);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Shadow");

    // Renderer
    cmd.add_component(renderer_entity, Shadow);

    cmd.assemble_wgpu_render_pipeline_with_usage::<ForwardPass>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<ForwardPass>(renderer_entity);

    cmd.assemble_wgpu_render_pipeline_with_usage::<ShadowPass>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<ShadowPass>(renderer_entity);

    cmd.assemble_wgpu_bind_group_with_usage::<ObjectTag>(renderer_entity);

    cmd.assemble_wgpu_command_buffers(renderer_entity);
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);

    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Uniform buffers
    let uniform_size = std::mem::size_of::<GlobalUniforms>() as BufferAddress;
    cmd.assemble_wgpu_buffer_with_usage::<ForwardPass>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: uniform_size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_with_usage::<ShadowPass>(
        renderer_entity,
        BufferDescriptor {
            label: None,
            size: uniform_size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.add_component(
        renderer_entity,
        ObjectTag::as_usage(BufferComponent::new(LazyComponent::Pending)),
    );

    cmd.add_component(
        renderer_entity,
        LightTag::as_usage(BufferComponent::new(LazyComponent::Pending)),
    );

    // Shader
    cmd.assemble_wgpu_shader(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        },
    );

    // Plane mesh data
    let plane_mesh_entity = cmd.push(());

    let (plane_vertices, plane_indices) = create_plane(7);
    let plane_vertices_len = plane_vertices.len();
    let plane_indices_len = plane_indices.len();
    cmd.add_component(plane_mesh_entity, PlaneMesh);
    cmd.assemble_wgpu_buffer_data_with_usage::<VertexTag, _>(
        plane_mesh_entity,
        RwLock::new(plane_vertices),
        0,
    );
    cmd.assemble_wgpu_buffer_data_with_usage::<IndexTag, _>(
        plane_mesh_entity,
        RwLock::new(plane_indices),
        0,
    );

    cmd.add_component(plane_mesh_entity, IndexFormat::Uint16);
    cmd.add_component(
        plane_mesh_entity,
        IndexCount::as_usage(plane_indices_len as BufferAddress),
    );

    // Plane mesh buffers
    cmd.assemble_wgpu_buffer_with_usage::<VertexTag>(
        plane_mesh_entity,
        BufferDescriptor {
            label: Some("Plane Vertex Buffer"),
            size: (std::mem::size_of::<Vertex>() * plane_vertices_len) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_with_usage::<IndexTag>(
        plane_mesh_entity,
        BufferDescriptor {
            label: Some("Plane Index Buffer"),
            size: (std::mem::size_of::<Index>() * plane_indices_len) as BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Cube mesh data
    let cube_mesh_entity = cmd.push(());

    cmd.add_component(cube_mesh_entity, CubeMesh);

    let (cube_vertices, cube_indices) = create_cube();
    let cube_vertices_len = cube_vertices.len();
    let cube_indices_len = cube_indices.len();

    cmd.assemble_wgpu_buffer_data_with_usage::<VertexTag, _>(
        cube_mesh_entity,
        RwLock::new(cube_vertices),
        0,
    );
    cmd.assemble_wgpu_buffer_data_with_usage::<IndexTag, _>(
        cube_mesh_entity,
        RwLock::new(cube_indices),
        0,
    );
    cmd.add_component(cube_mesh_entity, IndexFormat::Uint16);
    cmd.add_component(
        cube_mesh_entity,
        IndexCount::as_usage(cube_indices_len as BufferAddress),
    );

    // Cube mesh buffers
    cmd.assemble_wgpu_buffer_with_usage::<VertexTag>(
        cube_mesh_entity,
        BufferDescriptor {
            label: Some("Cubes Vertex Buffer"),
            size: (std::mem::size_of::<Vertex>() * cube_vertices_len) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_with_usage::<IndexTag>(
        cube_mesh_entity,
        BufferDescriptor {
            label: Some("Cubes Index Buffer"),
            size: (std::mem::size_of::<Index>() * cube_indices_len) as BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Forward depth view
    cmd.add_component(
        renderer_entity,
        ForwardPass::as_usage(TextureViewComponent::new(LazyComponent::Pending)),
    );

    // Shadow texture
    cmd.assemble_wgpu_texture_with_usage::<ShadowPass>(
        renderer_entity,
        TextureDescriptor {
            size: SHADOW_SIZE,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            label: None,
        },
    );

    // Shadow texture view
    cmd.assemble_wgpu_texture_view_with_usage::<ShadowPass>(
        renderer_entity,
        renderer_entity,
        Default::default(),
    );

    // Shadow sampler
    cmd.assemble_wgpu_sampler_with_usage::<ShadowPass>(
        renderer_entity,
        SamplerDescriptor {
            label: Some("shadow"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            compare: Some(CompareFunction::LessEqual),
            ..Default::default()
        },
    );

    // Light storage buffer
    cmd.add_component(
        renderer_entity,
        LightTag::as_usage(BufferComponent::new(LazyComponent::Pending)),
    );

    // Object uniform buffer
    cmd.add_component(
        renderer_entity,
        ObjectTag::as_usage(BufferComponent::new(LazyComponent::Pending)),
    );

    // Assemble objects
    // Plane
    let plane_entity = cmd.push(());
    assemble_object(
        cmd,
        plane_entity,
        Mesh::Plane,
        nalgebra::Matrix4::identity(),
        0.0,
        Color::WHITE,
        0,
    );

    // Cubes
    struct CubeDesc {
        offset: nalgebra::Vector3<f32>,
        angle: f32,
        scale: f32,
        rotation: f32,
    }
    let cube_descs = [
        CubeDesc {
            offset: nalgebra::vector![-2.0, -2.0, 2.0],
            angle: 10.0,
            scale: 0.7,
            rotation: 0.1,
        },
        CubeDesc {
            offset: nalgebra::vector![2.0, -2.0, 2.0],
            angle: 50.0,
            scale: 1.3,
            rotation: 0.2,
        },
        CubeDesc {
            offset: nalgebra::vector![-2.0, 2.0, 2.0],
            angle: 140.0,
            scale: 1.1,
            rotation: 0.3,
        },
        CubeDesc {
            offset: nalgebra::vector![2.0, 2.0, 2.0],
            angle: 210.0,
            scale: 0.9,
            rotation: 0.4,
        },
    ];

    for (i, cube) in cube_descs.iter().enumerate() {
        let mx_world = nalgebra::Matrix4::new_rotation(cube.offset.normalize() * cube.angle)
            * nalgebra::Matrix4::new_translation(&cube.offset);
        let mx_world = mx_world.scale(cube.scale);

        let cube_entity = cmd.push(());
        assemble_object(
            cmd,
            cube_entity,
            Mesh::Cube,
            mx_world,
            cube.rotation,
            Color::GREEN,
            (i as u32 + 1) * 256,
        );
    }

    // Lights
    let light_entity = cmd.push(());
    assemble_light(
        cmd,
        light_entity,
        nalgebra::vector![7.0, -5.0, 10.0],
        Color {
            r: 0.5,
            g: 1.0,
            b: 0.5,
            a: 1.0,
        },
        60.0,
        1.0..20.0,
        renderer_entity,
        0,
    );

    let light_entity = cmd.push(());
    assemble_light(
        cmd,
        light_entity,
        nalgebra::vector![-5.0, 7.0, 10.0],
        Color {
            r: 1.0,
            g: 0.5,
            b: 0.5,
            a: 1.0,
        },
        45.0,
        1.0..20.0,
        renderer_entity,
        1,
    );

    // 'Lights are dirty' flag
    cmd.add_component(renderer_entity, LightsAreDirty::as_usage(RwLock::new(true)));
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_shader_modules_system(),
            antigen_wgpu::create_buffers_system::<VertexTag>(),
            antigen_wgpu::create_buffers_system::<IndexTag>(),
            antigen_wgpu::buffer_write_system::<VertexTag, RwLock<Vec<Vertex>>, Vec<Vertex>>(),
            antigen_wgpu::buffer_write_system::<IndexTag, RwLock<Vec<Index>>, Vec<Index>>(),
            antigen_wgpu::create_textures_system::<ShadowPass>(),
            antigen_wgpu::create_texture_views_system::<ShadowPass>(),
            antigen_wgpu::create_samplers_with_usage_system::<ShadowPass>(),
        ],
        shadow_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![shadow_render_system()]
}
