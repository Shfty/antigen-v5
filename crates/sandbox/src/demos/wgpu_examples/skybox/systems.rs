use crate::wgpu_examples::skybox::{DEPTH_FORMAT, IMAGE_SIZE};

use super::{
    Camera, DepthTextureView, EntityPipelineComponent, SkyPipelineComponent, Skybox,
    SkyboxTextureComponent, SkyboxTextureViewComponent, UniformBufferComponent, Vertex,
    VertexBufferComponent, VertexCountComponent,
};
use antigen_core::{
    Changed, ChangedTrait, GetIndirect, IndirectComponent, LazyComponent,
    ReadWriteLock, RwLock,
};

use antigen_wgpu::{
    wgpu::{
        util::DeviceExt, vertex_attr_array, BindGroupDescriptor, BindGroupEntry,
        BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType,
        BufferAddress, BufferBindingType, Color, CommandEncoderDescriptor, CompareFunction,
        DepthBiasState, DepthStencilState, Device, Extent3d, Features, FragmentState, FrontFace,
        LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PrimitiveState, Queue,
        RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
        RenderPipelineDescriptor, ShaderStages, StencilState, SurfaceConfiguration,
        TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
        TextureView, TextureViewDescriptor, TextureViewDimension, VertexBufferLayout, VertexState,
        VertexStepMode,
    },
    BindGroupComponent, CommandBuffersComponent, RenderAttachmentTextureView, SamplerComponent,
    ShaderModuleComponent, SurfaceConfigurationComponent,
};

use antigen_winit::{winit::event::WindowEvent, WindowComponent, WindowEventComponent};
use legion::{world::SubWorld, IntoQuery};

fn create_depth_texture(config: &SurfaceConfiguration, device: &Device) -> TextureView {
    let depth_texture = device.create_texture(&TextureDescriptor {
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        label: None,
    });

    depth_texture.create_view(&TextureViewDescriptor::default())
}

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(Queue)]
#[read_component(SurfaceConfigurationComponent)]
pub fn skybox_prepare(
    world: &legion::world::SubWorld,
    _: &Skybox,
    shader_module: &ShaderModuleComponent,
    entity_pipeline_component: &EntityPipelineComponent,
    sky_pipeline_component: &SkyPipelineComponent,
    bind_group_component: &BindGroupComponent,
    sampler: &SamplerComponent,
    uniform_buffer: &UniformBufferComponent,
    depth_texture_view_component: &DepthTextureView,
    texture_component: &SkyboxTextureComponent,
    texture_view_component: &SkyboxTextureViewComponent,
    surface_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    let device = <&Device>::query().iter(world).next().unwrap();

    let queue = <&Queue>::query().iter(world).next().unwrap();

    let surface_component = world.get_indirect(surface_component).unwrap();
    let config = surface_component.read();

    if config.width == 0 || config.height == 0 {
        return;
    }

    if depth_texture_view_component.read().is_pending() {
        let depth_texture_view = create_depth_texture(&config, device);
        depth_texture_view_component
            .write()
            .set_ready(depth_texture_view);
    }

    if !entity_pipeline_component.read().is_pending() {
        return;
    }

    let shader_module = shader_module.read();
    let shader_module = if let LazyComponent::Ready(shader_module) = &*shader_module {
        shader_module
    } else {
        return;
    };

    let sampler = sampler.read();
    let sampler = if let LazyComponent::Ready(sampler) = &*sampler {
        sampler
    } else {
        return;
    };

    let uniform_buffer = uniform_buffer.read();
    let uniform_buffer = if let LazyComponent::Ready(uniform_buffer) = &*uniform_buffer {
        uniform_buffer
    } else {
        return;
    };

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: TextureViewDimension::Cube,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create the render pipelines
    let sky_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Sky"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader_module,
            entry_point: "vs_sky",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader_module,
            entry_point: "fs_sky",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState {
            front_face: FrontFace::Cw,
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: CompareFunction::LessEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
    });
    sky_pipeline_component.write().set_ready(sky_pipeline);

    let entity_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Entity"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader_module,
            entry_point: "vs_entity",
            buffers: &[VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as BufferAddress,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x3],
            }],
        },
        fragment: Some(FragmentState {
            module: &shader_module,
            entry_point: "fs_entity",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState {
            front_face: FrontFace::Cw,
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: CompareFunction::LessEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
    });
    entity_pipeline_component.write().set_ready(entity_pipeline);

    let device_features = device.features();

    let skybox_format = if device_features.contains(Features::TEXTURE_COMPRESSION_ASTC_LDR) {
        println!("Using ASTC_LDR");
        TextureFormat::Astc4x4RgbaUnormSrgb
    } else if device_features.contains(Features::TEXTURE_COMPRESSION_ETC2) {
        println!("Using ETC2");
        TextureFormat::Etc2RgbUnormSrgb
    } else if device_features.contains(Features::TEXTURE_COMPRESSION_BC) {
        println!("Using BC");
        TextureFormat::Bc1RgbaUnormSrgb
    } else {
        println!("Using plain");
        TextureFormat::Bgra8UnormSrgb
    };

    let size = Extent3d {
        width: IMAGE_SIZE,
        height: IMAGE_SIZE,
        depth_or_array_layers: 6,
    };

    let layer_size = Extent3d {
        depth_or_array_layers: 1,
        ..size
    };
    let max_mips = layer_size.max_mips();

    println!(
        "Copying {:?} skybox images of size {}, {}, 6 with {} mips to gpu",
        skybox_format, IMAGE_SIZE, IMAGE_SIZE, max_mips,
    );

    let bytes = match skybox_format {
        TextureFormat::Astc4x4RgbaUnormSrgb => &include_bytes!("images/astc.dds")[..],
        TextureFormat::Etc2RgbUnormSrgb => &include_bytes!("images/etc2.dds")[..],
        TextureFormat::Bc1RgbaUnormSrgb => &include_bytes!("images/bc1.dds")[..],
        TextureFormat::Bgra8UnormSrgb => &include_bytes!("images/bgra.dds")[..],
        _ => unreachable!(),
    };

    let image = ddsfile::Dds::read(&mut std::io::Cursor::new(&bytes)).unwrap();

    let texture = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            size,
            mip_level_count: max_mips as u32,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: skybox_format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            label: None,
        },
        &image.data,
    );

    let texture_view = texture.create_view(&TextureViewDescriptor {
        label: None,
        dimension: Some(TextureViewDimension::Cube),
        ..TextureViewDescriptor::default()
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler),
            },
        ],
        label: None,
    });

    texture_component.write().set_ready(texture);
    texture_view_component.write().set_ready(texture_view);
    bind_group_component.write().set_ready(bind_group);
}

#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn skybox_resize(
    world: &SubWorld,
    _: &Skybox,
    surface_config: &IndirectComponent<SurfaceConfigurationComponent>,
    camera_data: &Changed<RwLock<[f32; 52]>>,
    depth_texture_view_component: &DepthTextureView,
) {
    let surface_config = world.get_indirect(surface_config).unwrap();

    if surface_config.get_changed() {
        let surface_config = surface_config.read();

        let device = <&Device>::query().iter(world).next().unwrap();

        if surface_config.width == 0 || surface_config.height == 0 {
            return;
        }

        let depth_texture_view = create_depth_texture(&surface_config, device);

        depth_texture_view_component
            .write()
            .set_ready(depth_texture_view);

        let camera = Camera {
            angle_xz: 0.2,
            angle_y: 0.2,
            dist: 30.0,
        };
        *camera_data.write() =
            camera.to_uniform_data(surface_config.width as f32 / surface_config.height as f32);
        camera_data.set_changed(true);
    }
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
#[read_component(VertexBufferComponent)]
#[read_component(VertexCountComponent)]
pub fn skybox_render(
    world: &legion::world::SubWorld,
    _: &Skybox,
    entity_pipeline: &EntityPipelineComponent,
    sky_pipeline: &SkyPipelineComponent,
    bind_group: &BindGroupComponent,
    depth_view: &DepthTextureView,
    command_buffers: &CommandBuffersComponent,
    render_attachment_view: &IndirectComponent<RenderAttachmentTextureView>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let entity_pipeline = entity_pipeline.read();
    let entity_pipeline = if let LazyComponent::Ready(entity_pipeline) = &*entity_pipeline {
        entity_pipeline
    } else {
        return;
    };

    let sky_pipeline = sky_pipeline.read();
    let sky_pipeline = if let LazyComponent::Ready(sky_pipeline) = &*sky_pipeline {
        sky_pipeline
    } else {
        return;
    };

    let bind_group = bind_group.read();
    let bind_group = if let LazyComponent::Ready(bind_group) = &*bind_group {
        bind_group
    } else {
        return;
    };

    let depth_view = depth_view.read();
    let depth_view = if let LazyComponent::Ready(depth_view) = &*depth_view {
        depth_view
    } else {
        return;
    };

    let render_attachment_view = world.get_indirect(render_attachment_view).unwrap();
    let render_attachment_view = render_attachment_view.read();
    let render_attachment_view =
        if let LazyComponent::Ready(render_attachment_view) = &*render_attachment_view {
            render_attachment_view
        } else {
            return;
        };

    let vertex_buffers = <(&VertexBufferComponent, &VertexCountComponent)>::query()
        .iter(world)
        .map(|(vertex_buffer, vertex_count)| (vertex_buffer.read(), **vertex_count))
        .collect::<Vec<_>>();

    let vertex_buffers = vertex_buffers
        .iter()
        .map(|(vertex_buffer, vertex_count)| (&*vertex_buffer, *vertex_count))
        .collect::<Vec<_>>();

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[RenderPassColorAttachment {
            view: &render_attachment_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                }),
                store: true,
            },
        }],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(1.0),
                store: false,
            }),
            stencil_ops: None,
        }),
    });

    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.set_pipeline(&entity_pipeline);

    for (vertex_buffer, vertex_count) in vertex_buffers {
        let vertex_buffer = if let LazyComponent::Ready(vertex_buffer) = &**vertex_buffer {
            vertex_buffer
        } else {
            return;
        };

        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw(0..vertex_count as u32, 0..1);
    }

    rpass.set_pipeline(&sky_pipeline);
    rpass.draw(0..3, 0..1);
    drop(rpass);

    command_buffers.write().push(encoder.finish());
}

#[legion::system(par_for_each)]
#[read_component(WindowComponent)]
#[read_component(SurfaceConfigurationComponent)]
#[read_component(WindowEventComponent)]
pub fn skybox_cursor_moved(
    world: &SubWorld,
    _: &Skybox,
    camera_data: &Changed<RwLock<[f32; 52]>>,
    window: &IndirectComponent<WindowComponent>,
    surface_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    let window = world
        .get_indirect(window)
        .expect("No indirect WindowComponent");
    let window = window.read();
    let window = if let LazyComponent::Ready(window) = &*window {
        window
    } else {
        return;
    };

    let surface_component = world
        .get_indirect(surface_component)
        .expect("No indirect SurfaceConfigurationComponent");
    let config = surface_component.read();

    let window_event = <&WindowEventComponent>::query()
        .iter(world)
        .next()
        .expect("No WindowEventComponent");

    let window_event = window_event.read();
    let (window_id, position) = if let (
        Some(window_id),
        Some(WindowEvent::CursorMoved { position, .. }),
    ) = &*window_event
    {
        (window_id, position)
    } else {
        return;
    };

    if window.id() != *window_id {
        return;
    }

    let norm_x = position.x as f32 / config.width as f32;
    let norm_y = position.y as f32 / config.height as f32;

    let camera = Camera {
        angle_xz: norm_x * 5.0,
        angle_y: norm_y,
        dist: 30.0,
    };

    *camera_data.write() = camera.to_uniform_data(config.width as f32 / config.height as f32);
    camera_data.set_changed(true);

    window.request_redraw();
}
