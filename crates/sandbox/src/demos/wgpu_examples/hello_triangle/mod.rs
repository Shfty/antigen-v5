mod components;
mod systems;

use antigen_winit::AssembleWinit;
pub use components::*;
pub use systems::*;

use antigen_core::{AddIndirectComponent,  ImmutableSchedule, Serial, Single, serial, single};

use antigen_wgpu::{AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent, wgpu::{Device, ShaderModuleDescriptor, ShaderSource}};

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Hello Triangle");

    // Renderer
    cmd.add_component(renderer_entity, HelloTriangle);
    cmd.assemble_wgpu_render_pipeline(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    cmd.assemble_wgpu_shader(
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
