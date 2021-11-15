mod components;
mod systems;

pub use components::*;
pub use systems::*;

use antigen_core::{serial, single, AddIndirectComponent, ImmutableSchedule, Serial, Single};

use antigen_wgpu::{
    wgpu::{Device, ShaderModuleDescriptor, ShaderSource},
    RenderAttachmentTextureView, SurfaceConfigurationComponent,
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
    antigen_winit::assemble_window_title(cmd, &(window_entity,), &"Conservative Raster");

    // Renderer
    cmd.add_component(renderer_entity, ConservativeRaster);

    antigen_wgpu::assemble_render_pipeline_usage::<TriangleConservative>(cmd, renderer_entity);
    antigen_wgpu::assemble_render_pipeline_usage::<TriangleRegular>(cmd, renderer_entity);
    antigen_wgpu::assemble_render_pipeline_usage::<Upscale>(cmd, renderer_entity);
    antigen_wgpu::assemble_render_pipeline_usage::<Lines>(cmd, renderer_entity);
    antigen_wgpu::assemble_render_pipeline_usage::<Upscale>(cmd, renderer_entity);

    /*
    cmd.add_component(
        renderer_entity,
        Usage::<Upscale, _>::new(BindGroupLayoutComponent::pending()),
    );
    */

    antigen_wgpu::assemble_command_buffers(cmd, renderer_entity);
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    antigen_wgpu::assemble_shader_usage::<TriangleAndLines>(
        cmd,
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "triangle_and_lines.wgsl"
            ))),
        },
    );

    antigen_wgpu::assemble_shader_usage::<Upscale>(
        cmd,
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("upscale.wgsl"))),
        },
    );
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        antigen_wgpu::create_shader_modules_usage_system::<TriangleAndLines>(),
        antigen_wgpu::create_shader_modules_usage_system::<Upscale>(),
        conservative_raster_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![conservative_raster_render_system()]
}
