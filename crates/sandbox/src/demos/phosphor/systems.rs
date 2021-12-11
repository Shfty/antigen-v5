use std::time::Instant;

use super::components::*;
use antigen_core::{
    lazy_read_ready_else_return, Changed, ChangedTrait, GetIndirect, IndirectComponent,
    LazyComponent, ReadWriteLock, Usage,
};

use antigen_wgpu::{
    wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
        BindingResource, BindingType, BufferAddress, BufferBindingType, BufferSize, Color,
        CommandEncoderDescriptor, Device, Extent3d, FragmentState, LoadOp, MultisampleState,
        Operations, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
        TextureFormat, TextureSampleType, TextureViewDimension, VertexAttribute,
        VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
    },
    CommandBuffersComponent, RenderAttachmentTextureView, SurfaceConfigurationComponent,
    TextureDescriptorComponent, TextureViewDescriptorComponent,
};

use legion::{world::SubWorld, IntoQuery};

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn phosphor_prepare(
    world: &legion::world::SubWorld,
    _: &Phosphor,
    uniform_bind_group_component: &UniformBindGroupComponent,
    hdr_decay_shader: &HdrDecayShaderComponent,
    hdr_raster_shader: &HdrRasterShaderComponent,
    hdr_blit_pipeline_component: &HdrBlitPipelineComponent,
    hdr_raster_pipeline_component: &HdrRasterPipelineComponent,
    hdr_front_bind_group_component: &FrontBindGroupComponent,
    hdr_front_buffer_view: &HdrFrontBufferViewComponent,
    hdr_back_bind_group_component: &BackBindGroupComponent,
    hdr_back_buffer_view: &HdrBackBufferViewComponent,
    gradients_view: &GradientTextureViewComponent,
    linear_sampler: &LinearSamplerComponent,
    blit_shader: &BlitShaderComponent,
    blit_pipeline_component: &BlitPipelineComponent,
    uniform_buffer: &UniformBufferComponent,
    surface_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    // Fetch resources
    let device = <&Device>::query().iter(world).next().unwrap();

    lazy_read_ready_else_return!(hdr_decay_shader);
    lazy_read_ready_else_return!(hdr_raster_shader);
    lazy_read_ready_else_return!(hdr_front_buffer_view);
    lazy_read_ready_else_return!(hdr_back_buffer_view);

    lazy_read_ready_else_return!(gradients_view);

    lazy_read_ready_else_return!(linear_sampler);

    lazy_read_ready_else_return!(blit_shader);

    lazy_read_ready_else_return!(uniform_buffer);

    let surface_component = world.get_indirect(surface_component).unwrap();
    let config = surface_component.read();

    let uniform_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Uniform Bind Group Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(80),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
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

    // Uniform bind group
    if uniform_bind_group_component.read().is_pending() {
        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gradients_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&linear_sampler),
                },
            ],
            label: None,
        });

        uniform_bind_group_component
            .write()
            .set_ready(uniform_bind_group);
    }

    // HDR bind group
    let hdr_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("HDR Bind Group Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    if hdr_front_bind_group_component.read().is_pending() {
        let hdr_front_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &hdr_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&hdr_back_buffer_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gradients_view),
                },
            ],
            label: None,
        });
        hdr_front_bind_group_component
            .write()
            .set_ready(hdr_front_bind_group);
    }

    if hdr_back_bind_group_component.read().is_pending() {
        let hdr_back_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &hdr_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&hdr_front_buffer_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gradients_view),
                },
            ],
            label: None,
        });
        hdr_back_bind_group_component
            .write()
            .set_ready(hdr_back_bind_group);
    }

    // Don't update if the pipeline has already been initialized
    if !hdr_blit_pipeline_component.read().is_pending() {
        return;
    }

    // HDR pipeline
    let hdr_pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&uniform_bind_group_layout, &hdr_bind_group_layout],
        push_constant_ranges: &[],
    });

    let hdr_blit_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&hdr_pipeline_layout),
        vertex: VertexState {
            module: &hdr_decay_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &hdr_decay_shader,
            entry_point: "fs_main",
            targets: &[TextureFormat::Rgba32Float.into()],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    hdr_blit_pipeline_component
        .write()
        .set_ready(hdr_blit_pipeline);

    let hdr_raster_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&hdr_pipeline_layout),
        vertex: VertexState {
            module: &hdr_raster_shader,
            entry_point: "vs_main",
            buffers: &[
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<VertexData>() as BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 4]>() as BufferAddress,
                            shader_location: 1,
                        },
                    ],
                },
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<InstanceData>() as BufferAddress,
                    step_mode: VertexStepMode::Instance,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 2,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: std::mem::size_of::<[f32; 4]>() as BufferAddress,
                            shader_location: 3,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 8]>() as BufferAddress,
                            shader_location: 4,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 9]>() as BufferAddress,
                            shader_location: 5,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 10]>() as BufferAddress,
                            shader_location: 6,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 11]>() as BufferAddress,
                            shader_location: 7,
                        },
                    ],
                },
            ],
        },
        fragment: Some(FragmentState {
            module: &hdr_raster_shader,
            entry_point: "fs_main",
            targets: &[TextureFormat::Rgba32Float.into()],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    hdr_raster_pipeline_component
        .write()
        .set_ready(hdr_raster_pipeline);

    // Blit pipeline
    let blit_pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&uniform_bind_group_layout, &hdr_bind_group_layout],
        push_constant_ranges: &[],
    });

    let blit_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&blit_pipeline_layout),
        vertex: VertexState {
            module: &blit_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &blit_shader,
            entry_point: "fs_main",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    blit_pipeline_component.write().set_ready(blit_pipeline);
}

// Game tick update
#[legion::system(par_for_each)]
pub fn phosphor_update_total_time(
    start_time: &StartTimeComponent,
    total_time: &Changed<TotalTimeComponent>,
) {
    *total_time.write() = Instant::now().duration_since(**start_time).as_secs_f32();
    println!("Total time: {:#?}", total_time.read());
    total_time.set_changed(true);
}

#[legion::system(par_for_each)]
pub fn phosphor_update_delta_time(
    timestamp: &TimestampComponent,
    delta_time: &Changed<DeltaTimeComponent>,
) {
    let timestamp = *timestamp.read();
    *delta_time.write() = Instant::now().duration_since(timestamp).as_secs_f32();
    println!("Delta time: {:#?}", delta_time.read());
    delta_time.set_changed(true);
}

#[legion::system(par_for_each)]
pub fn phosphor_update_timestamp(timestamp: &TimestampComponent) {
    *timestamp.write() = Instant::now();
}

#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(Changed<TotalTimeComponent>)]
pub fn phosphor_update_instances(
    world: &legion::world::SubWorld,
    origin: &OriginComponent,
    oscilloscope: &Oscilloscope,
    instance_data: &Changed<InstanceDataComponent>,
) {
    let time = <&Changed<TotalTimeComponent>>::query()
        .iter(world)
        .next()
        .unwrap();
    let time = *time.read();

    {
        let (x, y) = *origin.read();
        let (fx, fy) = oscilloscope.eval(time);

        let mut instance = instance_data.write();
        instance.prev_position = instance.position;

        instance.position[0] = x + fx;
        instance.position[1] = y + fy;
    }

    instance_data.set_changed(true);
}

#[legion::system(par_for_each)]
#[read_component(SurfaceConfigurationComponent)]
pub fn phosphor_resize(
    world: &SubWorld,
    _: &Phosphor,
    surface_config: &IndirectComponent<SurfaceConfigurationComponent>,
    hdr_front_bind_group: &FrontBindGroupComponent,
    hdr_back_bind_group: &BackBindGroupComponent,
    hdr_front_buffer_desc: &Usage<HdrFrontBuffer, TextureDescriptorComponent<'static>>,
    hdr_front_buffer_view_desc: &Usage<HdrFrontBuffer, TextureViewDescriptorComponent<'static>>,
    hdr_back_buffer_desc: &Usage<HdrBackBuffer, TextureDescriptorComponent<'static>>,
    hdr_back_buffer_view_desc: &Usage<HdrBackBuffer, TextureViewDescriptorComponent<'static>>,
    projection_matrix: &Changed<ProjectionMatrixComponent>,
) {
    let surface_config = world.get_indirect(surface_config).unwrap();
    if !surface_config.get_changed() {
        return;
    }

    let surface_config = surface_config.read();

    let extent = Extent3d {
        width: surface_config.width,
        height: surface_config.height,
        depth_or_array_layers: 1,
    };

    hdr_front_buffer_desc.write().size = extent;
    hdr_back_buffer_desc.write().size = extent;

    hdr_front_buffer_desc.set_changed(true);
    hdr_back_buffer_desc.set_changed(true);
    hdr_front_buffer_view_desc.set_changed(true);
    hdr_back_buffer_view_desc.set_changed(true);
    hdr_front_bind_group.write().set_pending();
    hdr_back_bind_group.write().set_pending();

    *projection_matrix.write() = super::projection_matrix(
        surface_config.width as f32 / surface_config.height as f32,
        100.0,
    );
    projection_matrix.set_changed(true);
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
#[read_component(Changed<InstanceDataComponent>)]
pub fn phosphor_render(
    world: &legion::world::SubWorld,
    _: &Phosphor,
    uniform_bind_group: &UniformBindGroupComponent,
    hdr_blit_pipeline: &HdrBlitPipelineComponent,
    hdr_raster_pipeline: &HdrRasterPipelineComponent,
    hdr_front_bind_group: &FrontBindGroupComponent,
    hdr_back_bind_group: &BackBindGroupComponent,
    hdr_front_view: &HdrFrontBufferViewComponent,
    hdr_back_view: &HdrBackBufferViewComponent,
    blit_pipeline: &BlitPipelineComponent,
    vertex_buffer: &VertexBufferComponent,
    instance_buffer: &InstanceBufferComponent,
    buffer_flip_flop: &BufferFlipFlopComponent,
    command_buffers: &CommandBuffersComponent,
    render_attachment_view: &IndirectComponent<RenderAttachmentTextureView>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    lazy_read_ready_else_return!(uniform_bind_group);

    lazy_read_ready_else_return!(hdr_blit_pipeline);
    lazy_read_ready_else_return!(hdr_raster_pipeline);
    lazy_read_ready_else_return!(hdr_front_bind_group);
    lazy_read_ready_else_return!(hdr_back_bind_group);
    lazy_read_ready_else_return!(hdr_front_view);
    lazy_read_ready_else_return!(hdr_back_view);

    lazy_read_ready_else_return!(vertex_buffer);
    lazy_read_ready_else_return!(instance_buffer);

    let buffer_flip_state = *buffer_flip_flop.read();

    lazy_read_ready_else_return!(blit_pipeline);

    let render_attachment_view = world.get_indirect(render_attachment_view).unwrap();
    lazy_read_ready_else_return!(render_attachment_view);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[RenderPassColorAttachment {
            view: if buffer_flip_state {
                hdr_front_view
            } else {
                hdr_back_view
            },
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });
    rpass.set_pipeline(hdr_blit_pipeline);
    rpass.set_bind_group(0, uniform_bind_group, &[]);
    rpass.set_bind_group(
        1,
        if buffer_flip_state {
            hdr_front_bind_group
        } else {
            hdr_back_bind_group
        },
        &[],
    );
    rpass.draw(0..4, 0..1);
    drop(rpass);

    let instance_count = <&Changed<InstanceDataComponent>>::query().iter(world).count();

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[RenderPassColorAttachment {
            view: if buffer_flip_state {
                hdr_front_view
            } else {
                hdr_back_view
            },
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });
    rpass.set_pipeline(hdr_raster_pipeline);
    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
    rpass.set_vertex_buffer(1, instance_buffer.slice(..));
    rpass.set_bind_group(0, uniform_bind_group, &[]);
    rpass.set_bind_group(
        1,
        if buffer_flip_state {
            hdr_front_bind_group
        } else {
            hdr_back_bind_group
        },
        &[],
    );
    rpass.draw(0..14, 0..instance_count as u32);
    drop(rpass);

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[RenderPassColorAttachment {
            view: render_attachment_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });
    rpass.set_pipeline(blit_pipeline);
    rpass.set_bind_group(0, uniform_bind_group, &[]);
    rpass.set_bind_group(
        1,
        if buffer_flip_state {
            hdr_back_bind_group
        } else {
            hdr_front_bind_group
        },
        &[],
    );
    rpass.draw(0..4, 0..1);
    drop(rpass);

    command_buffers.write().push(encoder.finish());

    *buffer_flip_flop.write() = !buffer_flip_state;
}
