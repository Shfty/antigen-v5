mod components;
mod systems;

pub use components::*;
pub use systems::*;

use antigen_core::{serial, single, AddIndirectComponent, ImmutableSchedule, Serial, Single};

use antigen_wgpu::{
    wgpu::{Device, ShaderModuleDescriptor, ShaderSource},
    CommandBuffersComponent, RenderAttachmentTextureView, RenderPipelineComponent,
    SurfaceConfigurationComponent,
};

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    antigen_winit::assemble_window(cmd, &(window_entity,));
    antigen_wgpu::assemble_window_surface(cmd, &(window_entity,));

    // Add title to window
    antigen_winit::assemble_window_title(cmd, &(window_entity,), &"Hello Triangle");

    // Renderer
    cmd.add_component(renderer_entity, HelloTriangle);
    cmd.add_component(renderer_entity, RenderPipelineComponent::pending());
    cmd.add_component(renderer_entity, CommandBuffersComponent::new());
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    antigen_wgpu::assemble_shader(
        cmd,
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        },
    );
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        antigen_wgpu::create_shader_modules_system(),
        hello_triangle_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![hello_triangle_render_system()]
}
