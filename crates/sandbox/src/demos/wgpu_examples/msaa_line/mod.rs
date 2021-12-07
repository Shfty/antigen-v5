mod components;
mod systems;

use antigen_winit::{AssembleWinit, WindowComponent};
pub use components::*;
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddIndirectComponent,  ImmutableSchedule, RwLock, Serial,
    Single, Usage,
};

use antigen_wgpu::{
    wgpu::{
        BufferAddress, BufferDescriptor, BufferUsages, Device, Extent3d, ShaderModuleDescriptor,
        ShaderSource, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
    AssembleWgpu, MeshVertices, MsaaFramebuffer, MsaaFramebufferTextureDescriptor,
    MsaaFramebufferTextureView, RenderAttachmentTextureView, SurfaceConfigurationComponent,
    TextureViewDescriptorComponent,
};

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 2],
    _color: [f32; 4],
}

const LINE_COUNT: u32 = 50;
const VERTEX_COUNT: u32 = LINE_COUNT * 2;

const SAMPLE_COUNT: u32 = 4;

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "MSAA Line");

    // Renderer
    cmd.add_component(renderer_entity, MsaaLine);
    cmd.assemble_wgpu_pipeline_layout(renderer_entity);
    cmd.assemble_wgpu_render_bundle(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);

    cmd.add_indirect_component::<SurfaceConfigurationComponent>(
        renderer_entity,
        window_entity,
    );
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Window reference for input handling
    cmd.add_indirect_component::<WindowComponent>(renderer_entity, window_entity);

    // Shader
    cmd.assemble_wgpu_shader(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        },
    );

    // Vertex data
    let mut vertex_data = vec![];

    for i in 0..LINE_COUNT {
        let percent = i as f32 / LINE_COUNT as f32;
        let (sin, cos) = (percent * 2.0 * std::f32::consts::PI).sin_cos();
        vertex_data.push(Vertex {
            _pos: [0.0, 0.0],
            _color: [1.0, -sin, cos, 1.0],
        });
        vertex_data.push(Vertex {
            _pos: [1.0 * cos, 1.0 * sin],
            _color: [sin, -cos, 1.0, 1.0],
        });
    }

    cmd.assemble_wgpu_buffer_data_with_usage::<VertexBuffer, _>(
        renderer_entity,
        Usage::<MeshVertices, _>::new(RwLock::new(vertex_data)),
        0,
    );

    // Vertex buffer
    cmd.assemble_wgpu_buffer_with_usage::<VertexBuffer>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: std::mem::size_of::<Vertex>() as BufferAddress * LINE_COUNT as BufferAddress * 2,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // MSAA framebuffer
    cmd.assemble_wgpu_texture_with_usage::<MsaaFramebuffer>(
        renderer_entity,
        TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 800,
                height: 600,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: SAMPLE_COUNT,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT,
        },
    );

    // Texture view
    cmd.assemble_wgpu_texture_view_with_usage::<MsaaFramebuffer>(
        renderer_entity,
        renderer_entity,
        Default::default(),
    );

    cmd.add_indirect_component_self::<MsaaFramebufferTextureDescriptor>(renderer_entity);
    cmd.add_indirect_component_self::<Usage<MsaaFramebuffer, TextureViewDescriptorComponent>>(
        renderer_entity,
    );
    cmd.add_indirect_component_self::<MsaaFramebufferTextureView>(renderer_entity);
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_shader_modules_system(),
            antigen_wgpu::create_buffers_system::<VertexBuffer>(),
            serial![
                antigen_wgpu::create_textures_system::<MsaaFramebuffer>(),
                antigen_wgpu::create_texture_views_system::<MsaaFramebuffer>(),
            ]
        ],
        antigen_wgpu::buffer_write_system::<
            VertexBuffer,
            Usage::<MeshVertices, RwLock<Vec<Vertex>>>,
            Vec<Vertex>,
        >(),
        msaa_line_prepare_system()
    ]
}

pub fn keyboard_event_schedule() -> ImmutableSchedule<Single> {
    single![msaa_line_key_event_system()]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![msaa_line_render_system()]
}
