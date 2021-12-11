mod components;
mod systems;

use std::{f32::consts::PI, num::NonZeroU32, time::Instant};

use antigen_winit::{
    winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoopWindowTarget},
    },
    AssembleWinit, EventLoopHandler, RedrawUnconditionally,
};
pub use components::*;
use legion::Entity;
pub use systems::*;

use antigen_core::{parallel, serial, single, AddIndirectComponent, Construct, ImmutableWorld};

use antigen_wgpu::{
    wgpu::{
        AddressMode, BufferAddress, BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode,
        ImageCopyTextureBase, ImageDataLayout, Maintain, Origin3d, SamplerDescriptor,
        ShaderModuleDescriptor, ShaderSource, TextureAspect, TextureDescriptor, TextureDimension,
        TextureFormat, TextureUsages, TextureViewDescriptor,
    },
    AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent,
};

const MAX_GRADIENT_SIZE: usize = 8;
const MAX_GRADIENT_COUNT: usize = 4;
const MAX_LINES: usize = 6;

pub fn projection_matrix(aspect: f32, zoom: f32) -> [[f32; 4]; 4] {
    nalgebra::Matrix4::<f32>::new_orthographic(-aspect * zoom, aspect * zoom, -zoom, zoom, 0.0, 1.0)
        .into()
}

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let time_entity = cmd.push(());
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble time entity
    cmd.add_component(time_entity, StartTimeComponent::construct(Instant::now()));
    cmd.add_component(time_entity, TotalTimeComponent::construct(0.0));
    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        time_entity,
        TotalTimeComponent::construct(0.0),
        0,
        Some(renderer_entity),
    );
    cmd.add_component(time_entity, TimestampComponent::construct(Instant::now()));
    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        time_entity,
        DeltaTimeComponent::construct(0.0),
        std::mem::size_of::<f32>() as BufferAddress,
        Some(renderer_entity),
    );

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Phosphor");

    // Redraw window unconditionally
    cmd.add_component(window_entity, RedrawUnconditionally);

    // Renderer
    cmd.add_component(renderer_entity, Phosphor);
    cmd.assemble_wgpu_render_pipeline_with_usage::<HdrDecay>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<HdrRaster>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<Uniform>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<HdrFrontBuffer>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<HdrBackBuffer>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<Blit>(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);
    cmd.add_component(renderer_entity, BufferFlipFlopComponent::construct(false));
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // HDR front buffer
    cmd.assemble_wgpu_texture_with_usage::<HdrFrontBuffer>(
        renderer_entity,
        TextureDescriptor {
            label: Some("HDR Front Buffer"),
            size: Extent3d {
                width: 640,
                height: 480,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<HdrFrontBuffer>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label: Some("HDR Front Buffer View"),
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    // HDR back buffer
    cmd.assemble_wgpu_texture_with_usage::<HdrBackBuffer>(
        renderer_entity,
        TextureDescriptor {
            label: Some("HDR Back Buffer"),
            size: Extent3d {
                width: 640,
                height: 480,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<HdrBackBuffer>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label: Some("HDR Back Buffer View"),
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    // Gradients texture
    cmd.assemble_wgpu_texture_with_usage::<Gradients>(
        renderer_entity,
        TextureDescriptor {
            label: Some("Gradients Texture"),
            size: Extent3d {
                width: MAX_GRADIENT_SIZE as u32,
                height: MAX_GRADIENT_COUNT as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<Gradients>(
        renderer_entity,
        renderer_entity,
        TextureViewDescriptor {
            label: Some("Gradients Texture View"),
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    fn red() -> [f32; 4] {
        [1.0, 0.0, 0.0, 1.0]
    }

    fn green() -> [f32; 4] {
        [0.0, 1.0, 0.0, 1.0]
    }

    fn blue() -> [f32; 4] {
        [0.0, 0.0, 1.0, 1.0]
    }

    fn white() -> [f32; 4] {
        [1.0, 1.0, 1.0, 1.0]
    }

    fn black() -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }

    let mut gradient_data: GradientData = [Gradient::default(); MAX_GRADIENT_COUNT];
    gradient_data[0] = Gradient::new(vec![black(), red(), white()]);
    gradient_data[1] = Gradient::new(vec![black(), green(), white()]);
    gradient_data[2] = Gradient::new(vec![black(), blue(), white()]);

    cmd.assemble_wgpu_texture_data_with_usage::<Gradients, _>(
        renderer_entity,
        GradientDataComponent::construct(gradient_data),
        ImageCopyTextureBase {
            texture: (),
            mip_level: 0,
            origin: Origin3d::default(),
            aspect: TextureAspect::All,
        },
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(
                NonZeroU32::new(std::mem::size_of::<f32>() as u32 * 4 * MAX_GRADIENT_SIZE as u32)
                    .unwrap(),
            ),
            rows_per_image: None,
        },
    );

    // HDR sampler
    cmd.assemble_wgpu_sampler_with_usage::<Linear>(
        renderer_entity,
        SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        },
    );

    // Shaders
    cmd.assemble_wgpu_shader_with_usage::<HdrDecay>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("hdr_decay.wgsl"))),
        },
    );

    cmd.assemble_wgpu_shader_with_usage::<HdrRaster>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("hdr_raster.wgsl"))),
        },
    );

    cmd.assemble_wgpu_shader_with_usage::<Blit>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("blit.wgsl"))),
        },
    );

    // Uniform buffer
    cmd.assemble_wgpu_buffer_with_usage::<Uniform>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<UniformData>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        renderer_entity,
        ProjectionMatrixComponent::construct(projection_matrix(640.0 / 480.0, 10.0)),
        std::mem::size_of::<[f32; 4]>() as BufferAddress,
        Some(renderer_entity),
    );

    // Vertex buffer
    let intensity = 3.0;
    let delta_intensity = -3.0;
    let gradient = 0.0;
    let vertices = vec![
        VertexData {
            position: [-1.0, -1.0, 0.0, 1.0],
            intensity,
            delta_intensity,
            gradient,
            ..Default::default()
        },
        VertexData {
            position: [-1.0, 1.0, 0.0, 1.0],
            intensity,
            delta_intensity,
            gradient,
            ..Default::default()
        },
        VertexData {
            position: [0.0, -1.0, 0.0, 1.0],
            intensity,
            delta_intensity,
            gradient,
            ..Default::default()
        },
        VertexData {
            position: [0.0, 1.0, 0.0, 1.0],
            intensity,
            delta_intensity,
            gradient,
            ..Default::default()
        },
        VertexData {
            position: [0.0, -1.0, 0.0, 1.0],
            intensity,
            delta_intensity,
            gradient,
            ..Default::default()
        },
        VertexData {
            position: [0.0, 1.0, 0.0, 1.0],
            intensity,
            delta_intensity,
            gradient,
            ..Default::default()
        },
        VertexData {
            position: [1.0, -1.0, 0.0, 1.0],
            intensity,
            delta_intensity,
            gradient,
            ..Default::default()
        },
        VertexData {
            position: [1.0, 1.0, 0.0, 1.0],
            intensity,
            delta_intensity,
            gradient,
            ..Default::default()
        },
    ];

    cmd.assemble_wgpu_buffer_with_usage::<Vertex>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (std::mem::size_of::<VertexData>() * vertices.len()) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_data_with_usage::<Vertex, _>(
        renderer_entity,
        VertexDataComponent::construct(vertices),
        0,
        None,
    );

    // Instance buffer
    cmd.assemble_wgpu_buffer_with_usage::<Instance>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<InstanceData>() * MAX_LINES) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    assemble_oscilloscope(
        cmd,
        renderer_entity,
        0,
        (-80.0, 40.0),
        Oscilloscope::new(3.33, 30.0, |f| (f.sin(), f.cos())),
    );

    assemble_oscilloscope(
        cmd,
        renderer_entity,
        1,
        (0.0, 40.0),
        Oscilloscope::new(3.33, 30.0, |f| (f.sin(), (f * 1.2).sin())),
    );

    assemble_oscilloscope(
        cmd,
        renderer_entity,
        2,
        (80.0, 40.0),
        Oscilloscope::new(3.33, 30.0, |f| (f.cos(), (f * 1.2).cos())),
    );
}

fn assemble_oscilloscope(
    cmd: &mut legion::systems::CommandBuffer,
    buffer_target: Entity,
    buffer_index: usize,
    origin: (f32, f32),
    osc: Oscilloscope,
) {
    let entity = cmd.push(());
    cmd.add_component(entity, OriginComponent::construct(origin));
    cmd.add_component(entity, osc);
    cmd.assemble_wgpu_buffer_data_with_usage::<Instance, _>(
        entity,
        InstanceDataComponent::construct(InstanceData {
            position: [0.0, 0.0, 0.0, 1.0],
            prev_position: [0.0, 0.0, 0.0, 1.0],
        }),
        (std::mem::size_of::<InstanceData>() * buffer_index) as BufferAddress,
        Some(buffer_target),
    );
}

pub fn winit_event_handler<T>(mut f: impl EventLoopHandler<T>) -> impl EventLoopHandler<T> {
    let mut prepare_schedule = serial![
        antigen_wgpu::create_shader_modules_with_usage_system::<HdrDecay>(),
        antigen_wgpu::create_shader_modules_with_usage_system::<HdrRaster>(),
        antigen_wgpu::create_shader_modules_with_usage_system::<Blit>(),
        antigen_wgpu::create_buffers_system::<Uniform>(),
        antigen_wgpu::create_buffers_system::<Vertex>(),
        antigen_wgpu::create_buffers_system::<Instance>(),
        antigen_wgpu::create_textures_system::<HdrFrontBuffer>(),
        antigen_wgpu::create_textures_system::<HdrBackBuffer>(),
        antigen_wgpu::create_textures_system::<Gradients>(),
        antigen_wgpu::create_texture_views_system::<HdrFrontBuffer>(),
        antigen_wgpu::create_texture_views_system::<HdrBackBuffer>(),
        antigen_wgpu::create_texture_views_system::<Gradients>(),
        antigen_wgpu::create_samplers_with_usage_system::<Linear>(),
        antigen_wgpu::buffer_write_system::<Uniform, TotalTimeComponent, f32>(),
        antigen_wgpu::buffer_write_system::<Uniform, DeltaTimeComponent, f32>(),
        antigen_wgpu::buffer_write_system::<Uniform, ProjectionMatrixComponent, [[f32; 4]; 4]>(),
        antigen_wgpu::buffer_write_system::<Vertex, VertexDataComponent, Vec<VertexData>>(),
        antigen_wgpu::buffer_write_system::<Instance, InstanceDataComponent, InstanceData>(),
        antigen_wgpu::texture_write_system::<Gradients, GradientDataComponent, GradientData>(),
        phosphor_prepare_system()
    ];

    let mut render_schedule = serial![
        parallel![
            phosphor_update_total_time_system(),
            phosphor_update_delta_time_system(),
        ],
        phosphor_update_instances_system(),
        phosphor_render_system(),
        phosphor_update_timestamp_system(),
        antigen_wgpu::device_poll_system(Maintain::Wait),
    ];

    let mut surface_resize_schedule = single![phosphor_resize_system()];

    move |world: &ImmutableWorld,
          event: Event<'static, T>,
          event_loop_window_target: &EventLoopWindowTarget<T>,
          control_flow: &mut ControlFlow| {
        match &event {
            Event::MainEventsCleared => {
                surface_resize_schedule.execute(world);
                prepare_schedule.execute(world);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(_) => {
                    surface_resize_schedule.execute(world);
                }
                _ => (),
            },
            Event::RedrawEventsCleared => {
                render_schedule.execute(world);
            }
            _ => (),
        }

        f(world, event, event_loop_window_target, control_flow);
    }
}
