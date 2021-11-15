mod components;
mod systems;

pub use components::*;
pub use systems::*;

use antigen_core::{
    serial, single, AddIndirectComponent, ChangedFlag, ImmutableSchedule, Serial, Single, Usage,
};

use antigen_wgpu::{
    wgpu::{
        Device, Extent3d, FilterMode, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
    RenderAttachmentTextureView, SurfaceConfigurationComponent, TextureDescriptorComponent,
    TextureViewDescriptorComponent,
};

const RENDER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

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

    antigen_wgpu::assemble_bind_group_layout_usage::<Upscale>(cmd, renderer_entity);
    antigen_wgpu::assemble_bind_group_usage::<Upscale>(cmd, renderer_entity);

    antigen_wgpu::assemble_command_buffers(cmd, renderer_entity);
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<ChangedFlag<SurfaceConfigurationComponent>>(
        renderer_entity,
        window_entity,
    );
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

    antigen_wgpu::assemble_texture::<LowResTarget>(
        cmd,
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

    antigen_wgpu::assemble_texture_view::<LowResTarget>(cmd, renderer_entity, Default::default());

    antigen_wgpu::assemble_sampler::<LowResTarget>(
        cmd,
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
    cmd.add_indirect_component_self::<Usage<LowResTarget, ChangedFlag<TextureDescriptorComponent>>>(renderer_entity);
    cmd.add_indirect_component_self::<Usage<LowResTarget, ChangedFlag<TextureViewDescriptorComponent>>>(renderer_entity);
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        antigen_wgpu::create_shader_modules_usage_system::<TriangleAndLines>(),
        antigen_wgpu::create_shader_modules_usage_system::<Upscale>(),
        antigen_wgpu::create_textures_system::<LowResTarget>(),
        antigen_wgpu::create_texture_views_system::<LowResTarget>(),
        antigen_wgpu::create_samplers_system::<LowResTarget>(),
        conservative_raster_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![conservative_raster_render_system()]
}
