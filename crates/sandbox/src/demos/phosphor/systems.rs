use std::time::Instant;

use crate::phosphor::HDR_TEXTURE_FORMAT;

use super::*;
use antigen_core::{
    lazy_read_ready_else_return, Changed, ChangedTrait, GetIndirect, IndirectComponent,
    LazyComponent, ReadWriteLock, Usage,
};

use antigen_wgpu::{
    buffer_size_of,
    wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
        BindingResource, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState,
        BufferBindingType, BufferSize, Color, ColorTargetState, ColorWrites,
        CommandEncoderDescriptor, CompareFunction, ComputePassDescriptor,
        ComputePipelineDescriptor, DepthBiasState, DepthStencilState, Device, Extent3d, Face,
        FragmentState, FrontFace, IndexFormat, LoadOp, MultisampleState, Operations,
        PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment,
        RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        SamplerBindingType, ShaderStages, StencilState, TextureFormat, TextureSampleType,
        TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
        VertexStepMode,
    },
    CommandBuffersComponent, RenderAttachmentTextureView, SurfaceConfigurationComponent,
    TextureDescriptorComponent, TextureViewDescriptorComponent,
};

use antigen_winit::{winit::event::WindowEvent, WindowComponent, WindowEventComponent};
use legion::{world::SubWorld, IntoQuery};

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn phosphor_prepare(
    world: &legion::world::SubWorld,
    _: &PhosphorRenderer,
    // Render pipelines
    compute_pipeline_component: &ComputeLineInstancesPipeline,
    beam_line_pipeline_component: &BeamLinePipelineComponent,
    beam_mesh_pipeline_component: &BeamMeshPipelineComponent,
    phosphor_decay_pipeline_component: &PhosphorDecayPipelineComponent,
    tonemap_pipeline_component: &TonemapPipelineComponent,
    // Bind groups
    compute_bind_group_component: &ComputeBindGroupComponent,
    uniform_bind_group_component: &UniformBindGroupComponent,
    front_bind_group_component: &FrontBindGroupComponent,
    back_bind_group_component: &BackBindGroupComponent,
    // Shaders
    compute_line_instances_shader: &ComputeLineInstancesShader,
    beam_line_shader: &BeamLineShaderComponent,
    beam_mesh_shader: &BeamMeshShaderComponent,
    phosphor_decay_shader: &PhosphorDecayShaderComponent,
    tonemap_shader: &TonemapShaderComponent,
    // Texture views
    beam_buffer_view: &BeamBufferViewComponent,
    phosphor_front_buffer_view: &PhosphorFrontBufferViewComponent,
    phosphor_back_buffer_view: &PhosphorBackBufferViewComponent,
    // Samplers
    linear_sampler: &LinearSamplerComponent,
    // Buffers
    uniform_buffer: &UniformBufferComponent,
    mesh_vertex_buffer: &MeshVertexBufferComponent,
    line_index_buffer: &LineIndexBufferComponent,
    line_instance_buffer: &LineInstanceBufferComponent,
    // Misc
    surface_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    // Fetch resources
    let device = <&Device>::query().iter(world).next().unwrap();

    lazy_read_ready_else_return!(compute_line_instances_shader);
    lazy_read_ready_else_return!(phosphor_decay_shader);
    lazy_read_ready_else_return!(beam_line_shader);
    lazy_read_ready_else_return!(beam_mesh_shader);
    lazy_read_ready_else_return!(tonemap_shader);

    lazy_read_ready_else_return!(beam_buffer_view);
    lazy_read_ready_else_return!(phosphor_front_buffer_view);
    lazy_read_ready_else_return!(phosphor_back_buffer_view);

    lazy_read_ready_else_return!(linear_sampler);

    lazy_read_ready_else_return!(uniform_buffer);
    lazy_read_ready_else_return!(mesh_vertex_buffer);
    lazy_read_ready_else_return!(line_index_buffer);
    lazy_read_ready_else_return!(line_instance_buffer);

    let surface_component = world.get_indirect(surface_component).unwrap();
    let config = surface_component.read();

    // Compute bind group
    let compute_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(48),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(4),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(96),
                },
                count: None,
            },
        ],
    });

    if compute_bind_group_component.read().is_pending() {
        let compute_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: mesh_vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: line_index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: line_instance_buffer.as_entire_binding(),
                },
            ],
            label: None,
        });

        compute_bind_group_component
            .write()
            .set_ready(compute_bind_group);
    }

    // Compute pipeline
    let compute_pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: compute_line_instances_shader,
        entry_point: "main",
    });

    compute_pipeline_component
        .write()
        .set_ready(compute_pipeline);

    // Uniform bind group
    let uniform_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Uniform Bind Group Layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(144),
            },
            count: None,
        }],
    });

    if uniform_bind_group_component.read().is_pending() {
        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: None,
        });

        uniform_bind_group_component
            .write()
            .set_ready(uniform_bind_group);
    }

    // Beam pipeline layout
    let beam_pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&uniform_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Beam mesh pipeline
    let beam_mesh_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&beam_pipeline_layout),
        vertex: VertexState {
            module: &beam_mesh_shader,
            entry_point: "vs_main",
            buffers: &[VertexBufferLayout {
                array_stride: buffer_size_of::<MeshVertexData>(),
                step_mode: VertexStepMode::Vertex,
                attributes: &[
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: buffer_size_of::<[f32; 3]>(),
                        shader_location: 1,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: buffer_size_of::<[f32; 6]>(),
                        shader_location: 2,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32,
                        offset: buffer_size_of::<[f32; 9]>(),
                        shader_location: 3,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32,
                        offset: buffer_size_of::<[f32; 10]>(),
                        shader_location: 4,
                    },
                ],
            }],
        },
        fragment: Some(FragmentState {
            module: &beam_mesh_shader,
            entry_point: "fs_main",
            targets: &[HDR_TEXTURE_FORMAT.into()],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: 4,
            ..Default::default()
        },
        multiview: None,
    });

    beam_mesh_pipeline_component
        .write()
        .set_ready(beam_mesh_pipeline);

    // Beam line pipeline
    let beam_line_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&beam_pipeline_layout),
        vertex: VertexState {
            module: &beam_line_shader,
            entry_point: "vs_main",
            buffers: &[
                VertexBufferLayout {
                    array_stride: buffer_size_of::<LineVertexData>(),
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 3]>(),
                            shader_location: 1,
                        },
                    ],
                },
                VertexBufferLayout {
                    array_stride: buffer_size_of::<LineInstanceData>(),
                    step_mode: VertexStepMode::Instance,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 2,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: buffer_size_of::<[f32; 3]>(),
                            shader_location: 3,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: buffer_size_of::<[f32; 6]>(),
                            shader_location: 4,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 9]>(),
                            shader_location: 5,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 10]>(),
                            shader_location: 6,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: buffer_size_of::<[f32; 12]>(),
                            shader_location: 7,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: buffer_size_of::<[f32; 15]>(),
                            shader_location: 8,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: buffer_size_of::<[f32; 18]>(),
                            shader_location: 9,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 21]>(),
                            shader_location: 10,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 22]>(),
                            shader_location: 11,
                        },
                    ],
                },
            ],
        },
        fragment: Some(FragmentState {
            module: &beam_line_shader,
            entry_point: "fs_main",
            targets: &[ColorTargetState {
                format: HDR_TEXTURE_FORMAT,
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent::REPLACE,
                }),
                write_mask: ColorWrites::ALL,
            }],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: CompareFunction::Less,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: 4,
            ..Default::default()
        },
        multiview: None,
    });

    beam_line_pipeline_component
        .write()
        .set_ready(beam_line_pipeline);

    // Phosphor bind group
    let phosphor_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Phosphor Bind Group Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    if front_bind_group_component.read().is_pending() {
        let front_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &phosphor_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&phosphor_back_buffer_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&beam_buffer_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&linear_sampler),
                },
            ],
            label: None,
        });
        front_bind_group_component
            .write()
            .set_ready(front_bind_group);
    }

    if back_bind_group_component.read().is_pending() {
        let back_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &phosphor_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&phosphor_front_buffer_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&beam_buffer_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&linear_sampler),
                },
            ],
            label: None,
        });
        back_bind_group_component.write().set_ready(back_bind_group);
    }

    // Don't update if the pipeline has already been initialized
    if !phosphor_decay_pipeline_component.read().is_pending() {
        return;
    }

    // Phosphor pipeline layout
    let phosphor_decay_pipeline_layout =
        device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&uniform_bind_group_layout, &phosphor_bind_group_layout],
            push_constant_ranges: &[],
        });

    // Phosphor decay pipeline
    let phosphor_decay_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&phosphor_decay_pipeline_layout),
        vertex: VertexState {
            module: &phosphor_decay_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &phosphor_decay_shader,
            entry_point: "fs_main",
            targets: &[HDR_TEXTURE_FORMAT.into()],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
    });

    phosphor_decay_pipeline_component
        .write()
        .set_ready(phosphor_decay_pipeline);

    // Tonemap pipeline
    let tonemap_pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&phosphor_bind_group_layout],
        push_constant_ranges: &[],
    });

    let tonemap_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&tonemap_pipeline_layout),
        vertex: VertexState {
            module: &tonemap_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &tonemap_shader,
            entry_point: "fs_main",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
    });

    tonemap_pipeline_component
        .write()
        .set_ready(tonemap_pipeline);
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
pub fn phosphor_update_timers(timer_component: &TimerComponent) {
    let now = Instant::now();
    let timer = *timer_component.read();
    if now.duration_since(timer.timestamp) > timer.duration {
        timer_component.write().timestamp = now;
        timer_component.set_changed(true);
    }
}

#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(Changed<TotalTimeComponent>)]
#[read_component(Changed<DeltaTimeComponent>)]
pub fn phosphor_update_oscilloscopes(
    world: &legion::world::SubWorld,
    origin: &OriginComponent,
    oscilloscope: &Oscilloscope,
    vertex_data: &Changed<MeshVertexDataComponent>,
) {
    let total_time = <&Changed<TotalTimeComponent>>::query()
        .iter(world)
        .next()
        .unwrap();
    let total_time = *total_time.read();

    let delta_time = <&Changed<DeltaTimeComponent>>::query()
        .iter(world)
        .next()
        .unwrap();
    let delta_time = *delta_time.read();

    {
        let (x, y, z) = *origin.read();
        let (fx, fy, fz) = oscilloscope.eval(total_time);

        let mut vertices = vertex_data.write();

        vertices[0] = vertices[1];
        vertices[0].intensity += vertices[0].delta_intensity * delta_time;

        vertices[1].position[0] = x + fx;
        vertices[1].position[1] = y + fy;
        vertices[1].position[2] = z + fz;
    }

    vertex_data.set_changed(true);
}

#[legion::system(par_for_each)]
#[read_component(SurfaceConfigurationComponent)]
pub fn phosphor_resize(
    world: &SubWorld,
    _: &PhosphorRenderer,
    surface_config: &IndirectComponent<SurfaceConfigurationComponent>,
    // Bind groups
    front_bind_group: &FrontBindGroupComponent,
    back_bind_group: &BackBindGroupComponent,
    // Buffer descriptors
    beam_buffer_desc: &Usage<BeamBuffer, TextureDescriptorComponent<'static>>,
    beam_depth_buffer_desc: &Usage<BeamDepthBuffer, TextureDescriptorComponent<'static>>,
    beam_multisample_desc: &Usage<BeamMultisample, TextureDescriptorComponent<'static>>,
    phosphor_front_buffer_desc: &Usage<PhosphorFrontBuffer, TextureDescriptorComponent<'static>>,
    phosphor_back_buffer_desc: &Usage<PhosphorBackBuffer, TextureDescriptorComponent<'static>>,
    // Buffer view descriptors
    beam_buffer_view_desc: &Usage<BeamBuffer, TextureViewDescriptorComponent<'static>>,
    beam_depth_buffer_view_desc: &Usage<BeamDepthBuffer, TextureViewDescriptorComponent<'static>>,
    beam_multisample_view_desc: &Usage<BeamMultisample, TextureViewDescriptorComponent<'static>>,
    phosphor_front_buffer_view_desc: &Usage<
        PhosphorFrontBuffer,
        TextureViewDescriptorComponent<'static>,
    >,
    phosphor_back_buffer_view_desc: &Usage<
        PhosphorBackBuffer,
        TextureViewDescriptorComponent<'static>,
    >,
    // Matrices
    perspective_matrix: &Changed<PerspectiveMatrixComponent>,
    orthographic_matrix: &Changed<OrthographicMatrixComponent>,
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

    beam_buffer_desc.write().size = extent;
    beam_depth_buffer_desc.write().size = extent;
    beam_multisample_desc.write().size = extent;
    phosphor_front_buffer_desc.write().size = extent;
    phosphor_back_buffer_desc.write().size = extent;

    beam_buffer_desc.set_changed(true);
    beam_depth_buffer_desc.set_changed(true);
    beam_multisample_desc.set_changed(true);
    phosphor_front_buffer_desc.set_changed(true);
    phosphor_back_buffer_desc.set_changed(true);

    beam_buffer_view_desc.set_changed(true);
    beam_depth_buffer_view_desc.set_changed(true);
    beam_multisample_view_desc.set_changed(true);
    phosphor_front_buffer_view_desc.set_changed(true);
    phosphor_back_buffer_view_desc.set_changed(true);

    front_bind_group.write().set_pending();
    back_bind_group.write().set_pending();

    let aspect = surface_config.width as f32 / surface_config.height as f32;

    *perspective_matrix.write() = super::perspective_matrix(aspect, (0.0, 0.0), 1.0, 500.0);
    perspective_matrix.set_changed(true);

    *orthographic_matrix.write() = super::orthographic_matrix(aspect, 200.0, 1.0, 500.0);
    orthographic_matrix.set_changed(true);
}

#[legion::system(par_for_each)]
#[read_component(WindowComponent)]
#[read_component(SurfaceConfigurationComponent)]
#[read_component(WindowEventComponent)]
pub fn phosphor_cursor_moved(
    world: &SubWorld,
    _: &PhosphorRenderer,
    projection_matrix: &Changed<PerspectiveMatrixComponent>,
    window: &IndirectComponent<WindowComponent>,
    surface_config: &IndirectComponent<SurfaceConfigurationComponent>,
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

    let surface_config = world
        .get_indirect(surface_config)
        .expect("No indirect SurfaceConfigurationComponent");
    let surface_config = surface_config.read();

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

    let norm_x = ((position.x as f32 / surface_config.width as f32) * 2.0) - 1.0;
    let norm_y = ((position.y as f32 / surface_config.height as f32) * 2.0) - 1.0;

    *projection_matrix.write() = super::perspective_matrix(
        surface_config.width as f32 / surface_config.height as f32,
        (-norm_x, norm_y),
        1.0,
        500.0,
    );
    projection_matrix.set_changed(true);
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
pub fn phosphor_render(
    world: &legion::world::SubWorld,
    _: &PhosphorRenderer,
    // Pipelines
    compute_pipeline: &ComputeLineInstancesPipeline,
    beam_line_pipeline: &BeamLinePipelineComponent,
    beam_mesh_pipeline: &BeamMeshPipelineComponent,
    phosphor_decay_pipeline: &PhosphorDecayPipelineComponent,
    tonemap_pipeline: &TonemapPipelineComponent,
    // Bind groups
    compute_bind_group: &ComputeBindGroupComponent,
    uniform_bind_group: &UniformBindGroupComponent,
    front_bind_group: &FrontBindGroupComponent,
    back_bind_group: &BackBindGroupComponent,
    // Texture views
    beam_buffer_view: &BeamBufferViewComponent,
    beam_depth_view: &BeamDepthBufferViewComponent,
    beam_multisample_view: &BeamMultisampleViewComponent,
    phosphor_front_view: &PhosphorFrontBufferViewComponent,
    phosphor_back_view: &PhosphorBackBufferViewComponent,
    // Buffers
    line_vertex_buffer: &LineVertexBufferComponent,
    line_instance_buffer: &LineInstanceBufferComponent,
    mesh_vertex_buffer: &MeshVertexBufferComponent,
    mesh_index_buffer: &MeshIndexBufferComponent,
    // Misc
    buffer_flip_flop: &BufferFlipFlopComponent,
    command_buffers: &CommandBuffersComponent,
    render_attachment_view: &IndirectComponent<RenderAttachmentTextureView>,
    mesh_index_count: &MeshIndexCountComponent,
    line_index_count: &LineIndexCountComponent,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    lazy_read_ready_else_return!(compute_pipeline);
    lazy_read_ready_else_return!(phosphor_decay_pipeline);
    lazy_read_ready_else_return!(beam_line_pipeline);
    lazy_read_ready_else_return!(beam_mesh_pipeline);
    lazy_read_ready_else_return!(tonemap_pipeline);

    lazy_read_ready_else_return!(uniform_bind_group);
    lazy_read_ready_else_return!(compute_bind_group);
    lazy_read_ready_else_return!(front_bind_group);
    lazy_read_ready_else_return!(back_bind_group);

    lazy_read_ready_else_return!(beam_buffer_view);
    lazy_read_ready_else_return!(beam_depth_view);
    lazy_read_ready_else_return!(beam_multisample_view);
    lazy_read_ready_else_return!(phosphor_front_view);
    lazy_read_ready_else_return!(phosphor_back_view);

    lazy_read_ready_else_return!(line_vertex_buffer);
    lazy_read_ready_else_return!(line_instance_buffer);
    lazy_read_ready_else_return!(mesh_vertex_buffer);
    lazy_read_ready_else_return!(mesh_index_buffer);

    let buffer_flip_state = *buffer_flip_flop.read();
    let mesh_index_count = *mesh_index_count.read();
    let line_index_count = *line_index_count.read();
    let line_count = line_index_count / 2;

    let render_attachment_view = world.get_indirect(render_attachment_view).unwrap();
    lazy_read_ready_else_return!(render_attachment_view);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    // Compute line instances
    let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Compute Pass"),
    });
    cpass.set_pipeline(compute_pipeline);
    cpass.set_bind_group(0, compute_bind_group, &[]);
    cpass.dispatch(line_count as u32, 1, 1);
    drop(cpass);

    // Draw beam meshes
    println!(
        "Drawing {} mesh indices ({} triangles)",
        mesh_index_count,
        mesh_index_count / 3
    );
    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[RenderPassColorAttachment {
            view: beam_multisample_view,
            resolve_target: Some(beam_buffer_view),
            ops: Operations {
                load: LoadOp::Clear(CLEAR_COLOR),
                store: true,
            },
        }],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: beam_depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        }),
    });
    rpass.set_pipeline(beam_mesh_pipeline);
    rpass.set_vertex_buffer(0, mesh_vertex_buffer.slice(..));
    rpass.set_index_buffer(mesh_index_buffer.slice(..), IndexFormat::Uint16);
    rpass.set_bind_group(0, uniform_bind_group, &[]);
    rpass.draw_indexed(0..mesh_index_count as u32, 0, 0..1);
    drop(rpass);

    // Draw beam lines
    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[RenderPassColorAttachment {
            view: beam_multisample_view,
            resolve_target: Some(beam_buffer_view),
            ops: Operations {
                load: LoadOp::Load,
                store: true,
            },
        }],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: beam_depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: false,
            }),
            stencil_ops: None,
        }),
    });
    rpass.set_pipeline(beam_line_pipeline);
    rpass.set_vertex_buffer(0, line_vertex_buffer.slice(..));
    rpass.set_vertex_buffer(1, line_instance_buffer.slice(..));
    rpass.set_bind_group(0, uniform_bind_group, &[]);

    rpass.draw(0..14, 0..line_count as u32);
    drop(rpass);

    // Combine beam buffer with phosphor back buffer in phosphor front buffer
    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[RenderPassColorAttachment {
            view: if buffer_flip_state {
                phosphor_front_view
            } else {
                phosphor_back_view
            },
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });
    rpass.set_pipeline(phosphor_decay_pipeline);
    rpass.set_bind_group(0, uniform_bind_group, &[]);
    rpass.set_bind_group(
        1,
        if buffer_flip_state {
            front_bind_group
        } else {
            back_bind_group
        },
        &[],
    );
    rpass.draw(0..4, 0..1);
    drop(rpass);

    // Tonemap phosphor buffer to surface
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
    rpass.set_pipeline(tonemap_pipeline);
    rpass.set_bind_group(
        0,
        if buffer_flip_state {
            back_bind_group
        } else {
            front_bind_group
        },
        &[],
    );
    rpass.draw(0..4, 0..1);
    drop(rpass);

    // Finish encoding
    command_buffers.write().push(encoder.finish());

    // Flip buffer flag
    *buffer_flip_flop.write() = !buffer_flip_state;
}
