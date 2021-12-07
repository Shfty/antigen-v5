mod components;
mod systems;

use antigen_winit::AssembleWinit;
pub use components::*;
pub use systems::*;

use antigen_core::{AddIndirectComponent,  ImmutableSchedule, Serial, Single, Usage, serial, single};

use antigen_wgpu::{
    wgpu::{
        Device, Extent3d, FilterMode, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
    AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent,
    TextureViewDescriptorComponent,
};

const RENDER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Conservative Raster");

    // Renderer
    cmd.add_component(renderer_entity, ConservativeRaster);

    cmd.assemble_wgpu_render_pipeline_with_usage::<TriangleConservative>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<TriangleRegular>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<Upscale>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<Lines>(renderer_entity);

    cmd.assemble_wgpu_bind_group_layout_with_usage::<Upscale>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<Upscale>(renderer_entity);

    cmd.assemble_wgpu_command_buffers(renderer_entity);
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    cmd.assemble_wgpu_shader_with_usage::<TriangleAndLines>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "triangle_and_lines.wgsl"
            ))),
        },
    );

    cmd.assemble_wgpu_shader_with_usage::<Upscale>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("upscale.wgsl"))),
        },
    );

    cmd.assemble_wgpu_texture_with_usage::<LowResTarget>(
        renderer_entity,
        TextureDescriptor {
            label: Some("Low Resolution Target"),
            size: Extent3d {
                width: 800 / 16,
                height: 600 / 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: RENDER_TARGET_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<LowResTarget>(renderer_entity, renderer_entity, Default::default());

    cmd.assemble_wgpu_sampler_with_usage::<LowResTarget>(
        renderer_entity,
        SamplerDescriptor {
            label: Some("Nearest Neighbor Sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..Default::default()
        },
    );

    // Indirect components for resize
    cmd.add_indirect_component_self::<LowResTextureDescriptorComponent>(renderer_entity);
    cmd.add_indirect_component_self::<Usage<LowResTarget, TextureViewDescriptorComponent>>(renderer_entity);
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        antigen_wgpu::create_shader_modules_usage_system::<TriangleAndLines>(),
        antigen_wgpu::create_shader_modules_usage_system::<Upscale>(),
        antigen_wgpu::create_textures_system::<LowResTarget>(),
        antigen_wgpu::create_texture_views_system::<LowResTarget>(),
        antigen_wgpu::create_samplers_with_usage_system::<LowResTarget>(),
        conservative_raster_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![conservative_raster_render_system()]
}
