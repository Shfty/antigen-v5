mod components;
mod systems;

pub use components::*;
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddIndirectComponent, ImmutableSchedule, Serial, Single, Usage,
};

use antigen_wgpu::{
    wgpu::{
        BufferAddress, BufferDescriptor, BufferUsages, Device, Extent3d, ShaderModuleDescriptor,
        ShaderSource, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        TextureViewDescriptor,
    },
    BufferComponent, CommandBuffersComponent, MeshVertices, MsaaFramebuffer,
    MsaaFramebufferTexture, MsaaFramebufferTextureView, PipelineLayoutComponent,
    RenderAttachmentTextureView, RenderBundleComponent, ShaderModuleComponent, SurfaceComponent,
    TextureComponent, TextureViewComponent,
};

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 2],
    _color: [f32; 4],
}

pub enum VertexBuffer {}

pub type VertexBufferComponent<'a> = Usage<VertexBuffer, BufferComponent<'a>>;

const LINE_COUNT: u32 = 50;
const VERTEX_COUNT: u32 = LINE_COUNT * 2;

const SAMPLE_COUNT: u32 = 4;

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    antigen_winit::assemble_window(cmd, &(window_entity,));
    antigen_wgpu::assemble_window_surface(cmd, &(window_entity,));

    // Add title to window
    antigen_winit::assemble_window_title(cmd, &(window_entity,), &"MSAA Line");

    // Renderer
    cmd.add_component(renderer_entity, MsaaLine);
    cmd.add_component(renderer_entity, PipelineLayoutComponent::pending());
    cmd.add_component(renderer_entity, RenderBundleComponent::pending());
    cmd.add_component(renderer_entity, CommandBuffersComponent::new());
    cmd.add_indirect_component::<SurfaceComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Shader
    cmd.add_component(
        renderer_entity,
        ShaderModuleComponent::pending(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        }),
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

    antigen_wgpu::assemble_buffer_data::<VertexBuffer, _>(
        cmd,
        renderer_entity,
        MeshVertices::new(vertex_data),
        0,
    );

    // Vertex buffer
    cmd.add_component(
        renderer_entity,
        VertexBufferComponent::new(BufferComponent::pending(BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: std::mem::size_of::<Vertex>() as BufferAddress * LINE_COUNT as BufferAddress * 2,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })),
    );

    // MSAA framebuffer
    cmd.add_component(
        renderer_entity,
        MsaaFramebufferTexture::new(TextureComponent::pending(TextureDescriptor {
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
        })),
    );

    // Texture view
    cmd.add_component(
        renderer_entity,
        MsaaFramebufferTextureView::new(TextureViewComponent::pending(
            TextureViewDescriptor::default(),
        )),
    );

    cmd.add_indirect_component_self::<MsaaFramebufferTextureView<'static>>(renderer_entity);
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_shader_modules_system(),
            antigen_wgpu::create_buffers_system::<VertexBuffer>(),
            antigen_wgpu::create_textures_system::<MsaaFramebuffer>(),
            antigen_wgpu::create_texture_views_system::<MsaaFramebuffer>(),
        ],
        parallel![antigen_wgpu::buffer_write_system::<
            VertexBuffer,
            MeshVertices::<Vertex>,
            Vec<Vertex>,
        >(),],
        msaa_line_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![msaa_line_render_system()]
}
