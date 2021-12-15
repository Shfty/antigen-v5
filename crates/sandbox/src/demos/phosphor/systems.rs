use std::time::Instant;

use crate::phosphor::{HDR_TEXTURE_FORMAT, MAX_LINES, MAX_MESH_INDICES};

use super::components::*;
use antigen_core::{
    lazy_read_ready_else_return, Changed, ChangedTrait, GetIndirect, IndirectComponent,
    LazyComponent, ReadWriteLock, Usage,
};

use antigen_wgpu::{CommandBuffersComponent, RenderAttachmentTextureView, SurfaceConfigurationComponent, TextureDescriptorComponent, TextureViewDescriptorComponent, buffer_size_of, wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState, BufferBindingType, BufferSize, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, CompareFunction, DepthBiasState, DepthStencilState, Device, Extent3d, Face, FragmentState, FrontFace, IndexFormat, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, StencilState, TextureFormat, TextureSampleType, TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode}};

use antigen_winit::{winit::event::WindowEvent, WindowComponent, WindowEventComponent};
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
    hdr_line_shader: &HdrLineShaderComponent,
    hdr_mesh_shader: &HdrMeshShaderComponent,
    hdr_decay_pipeline_component: &HdrDecayPipelineComponent,
    hdr_line_pipeline_component: &HdrLinePipelineComponent,
    hdr_mesh_pipeline_component: &HdrMeshPipelineComponent,
    hdr_front_bind_group_component: &FrontBindGroupComponent,
    hdr_front_buffer_view: &HdrFrontBufferViewComponent,
    hdr_back_bind_group_component: &BackBindGroupComponent,
    hdr_back_buffer_view: &HdrBackBufferViewComponent,
    gradients_view: &GradientTextureViewComponent,
    linear_sampler: &LinearSamplerComponent,
    tonemap_shader: &TonemapShaderComponent,
    tonemap_pipeline_component: &TonemapPipelineComponent,
    uniform_buffer: &UniformBufferComponent,
    surface_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    // Fetch resources
    let device = <&Device>::query().iter(world).next().unwrap();

    lazy_read_ready_else_return!(hdr_decay_shader);
    lazy_read_ready_else_return!(hdr_line_shader);
    lazy_read_ready_else_return!(hdr_mesh_shader);
    lazy_read_ready_else_return!(hdr_front_buffer_view);
    lazy_read_ready_else_return!(hdr_back_buffer_view);

    lazy_read_ready_else_return!(gradients_view);

    lazy_read_ready_else_return!(linear_sampler);

    lazy_read_ready_else_return!(tonemap_shader);

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
                    min_binding_size: BufferSize::new(144),
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
    if !hdr_decay_pipeline_component.read().is_pending() {
        return;
    }

    // HDR pipeline layout
    let hdr_pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&uniform_bind_group_layout, &hdr_bind_group_layout],
        push_constant_ranges: &[],
    });

    // HDR decay pipeline
    let hdr_decay_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
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
            targets: &[HDR_TEXTURE_FORMAT.into()],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    hdr_decay_pipeline_component
        .write()
        .set_ready(hdr_decay_pipeline);

    // HDR mesh pipeline
    let hdr_mesh_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&hdr_pipeline_layout),
        vertex: VertexState {
            module: &hdr_mesh_shader,
            entry_point: "vs_main",
            buffers: &[VertexBufferLayout {
                array_stride: buffer_size_of::<MeshVertexData>(),
                step_mode: VertexStepMode::Vertex,
                attributes: &[
                    VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32,
                        offset: buffer_size_of::<[f32; 4]>(),
                        shader_location: 1,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32,
                        offset: buffer_size_of::<[f32; 5]>(),
                        shader_location: 2,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32,
                        offset: buffer_size_of::<[f32; 6]>(),
                        shader_location: 3,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32,
                        offset: buffer_size_of::<[f32; 7]>(),
                        shader_location: 4,
                    },
                ],
            }],
        },
        fragment: Some(FragmentState {
            module: &hdr_line_shader,
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
        multisample: MultisampleState::default(),
    });

    hdr_mesh_pipeline_component
        .write()
        .set_ready(hdr_mesh_pipeline);

    // HDR line pipeline
    let hdr_line_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&hdr_pipeline_layout),
        vertex: VertexState {
            module: &hdr_line_shader,
            entry_point: "vs_main",
            buffers: &[
                VertexBufferLayout {
                    array_stride: buffer_size_of::<LineVertexData>(),
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 4]>(),
                            shader_location: 1,
                        },
                    ],
                },
                VertexBufferLayout {
                    array_stride: buffer_size_of::<LineInstanceData>(),
                    step_mode: VertexStepMode::Instance,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 2,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 4]>(),
                            shader_location: 3,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 5]>(),
                            shader_location: 4,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 6]>(),
                            shader_location: 5,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 7]>(),
                            shader_location: 6,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: buffer_size_of::<[f32; 8]>(),
                            shader_location: 7,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 12]>(),
                            shader_location: 8,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 13]>(),
                            shader_location: 9,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 14]>(),
                            shader_location: 10,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: buffer_size_of::<[f32; 15]>(),
                            shader_location: 11,
                        },
                    ],
                },
            ],
        },
        fragment: Some(FragmentState {
            module: &hdr_line_shader,
            entry_point: "fs_main",
            targets: &[ColorTargetState {
                format: HDR_TEXTURE_FORMAT,
                blend: Some(BlendState {
                    color: BlendComponent::REPLACE,
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
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
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
    });

    hdr_line_pipeline_component
        .write()
        .set_ready(hdr_line_pipeline);

    // Blit pipeline
    let tonemap_pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&uniform_bind_group_layout, &hdr_bind_group_layout],
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
    line_data: &Changed<LineInstanceDataComponent>,
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

        let mut instance = line_data.write();

        instance[0].v0 = instance[0].v1;
        instance[0].v0.intensity += instance[0].v0.delta_intensity * delta_time;

        instance[0].v1.position[0] = x + fx;
        instance[0].v1.position[1] = y + fy;
        instance[0].v1.position[2] = z + fz;
    }

    line_data.set_changed(true);
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
    hdr_depth_buffer_desc: &Usage<HdrDepthBuffer, TextureDescriptorComponent<'static>>,
    hdr_depth_buffer_view_desc: &Usage<HdrDepthBuffer, TextureViewDescriptorComponent<'static>>,
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

    hdr_front_buffer_desc.write().size = extent;
    hdr_back_buffer_desc.write().size = extent;
    hdr_depth_buffer_desc.write().size = extent;

    hdr_front_buffer_desc.set_changed(true);
    hdr_back_buffer_desc.set_changed(true);
    hdr_depth_buffer_desc.set_changed(true);

    hdr_front_buffer_view_desc.set_changed(true);
    hdr_back_buffer_view_desc.set_changed(true);
    hdr_depth_buffer_view_desc.set_changed(true);

    hdr_front_bind_group.write().set_pending();
    hdr_back_bind_group.write().set_pending();

    let aspect = surface_config.width as f32 / surface_config.height as f32;

    *perspective_matrix.write() = super::perspective_matrix(aspect, (0.0, 0.0));
    perspective_matrix.set_changed(true);

    *orthographic_matrix.write() = super::orthographic_matrix(aspect, 200.0);
    orthographic_matrix.set_changed(true);
}

#[legion::system(par_for_each)]
#[read_component(WindowComponent)]
#[read_component(SurfaceConfigurationComponent)]
#[read_component(WindowEventComponent)]
pub fn phosphor_cursor_moved(
    world: &SubWorld,
    _: &Phosphor,
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
    );
    projection_matrix.set_changed(true);
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
#[read_component(Changed<LineInstanceDataComponent>)]
pub fn phosphor_render(
    world: &legion::world::SubWorld,
    _: &Phosphor,
    uniform_bind_group: &UniformBindGroupComponent,
    hdr_decay_pipeline: &HdrDecayPipelineComponent,
    hdr_line_pipeline: &HdrLinePipelineComponent,
    hdr_mesh_pipeline: &HdrMeshPipelineComponent,
    hdr_front_bind_group: &FrontBindGroupComponent,
    hdr_back_bind_group: &BackBindGroupComponent,
    hdr_front_view: &HdrFrontBufferViewComponent,
    hdr_back_view: &HdrBackBufferViewComponent,
    hdr_depth_view: &HdrDepthBufferViewComponent,
    tonemap_pipeline: &TonemapPipelineComponent,
    line_vertex_buffer: &LineVertexBufferComponent,
    line_instance_buffer: &LineInstanceBufferComponent,
    mesh_vertex_buffer: &MeshVertexBufferComponent,
    mesh_index_buffer: &MeshIndexBufferComponent,
    buffer_flip_flop: &BufferFlipFlopComponent,
    timer: &TimerComponent,
    command_buffers: &CommandBuffersComponent,
    render_attachment_view: &IndirectComponent<RenderAttachmentTextureView>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    lazy_read_ready_else_return!(uniform_bind_group);

    lazy_read_ready_else_return!(hdr_decay_pipeline);
    lazy_read_ready_else_return!(hdr_front_bind_group);
    lazy_read_ready_else_return!(hdr_back_bind_group);
    lazy_read_ready_else_return!(hdr_front_view);
    lazy_read_ready_else_return!(hdr_back_view);
    lazy_read_ready_else_return!(hdr_depth_view);

    lazy_read_ready_else_return!(hdr_line_pipeline);
    lazy_read_ready_else_return!(line_vertex_buffer);
    lazy_read_ready_else_return!(line_instance_buffer);

    lazy_read_ready_else_return!(hdr_mesh_pipeline);
    lazy_read_ready_else_return!(mesh_vertex_buffer);
    lazy_read_ready_else_return!(mesh_index_buffer);

    let buffer_flip_state = *buffer_flip_flop.read();

    lazy_read_ready_else_return!(tonemap_pipeline);

    let render_attachment_view = world.get_indirect(render_attachment_view).unwrap();
    lazy_read_ready_else_return!(render_attachment_view);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    let draw_changed = timer.get_changed();
    if draw_changed {
        timer.set_changed(false);
    }

    // Copy texels from backbuffer and apply phosphor decay
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
    rpass.set_pipeline(hdr_decay_pipeline);
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

    // Draw meshes
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
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: hdr_depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        }),
    });
    rpass.set_pipeline(hdr_mesh_pipeline);
    rpass.set_vertex_buffer(0, mesh_vertex_buffer.slice(..));
    rpass.set_index_buffer(mesh_index_buffer.slice(..), IndexFormat::Uint16);
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
    rpass.draw_indexed(0..MAX_MESH_INDICES as u32, 0, 0..1);
    drop(rpass);

    // Draw lines
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
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: hdr_depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: false,
            }),
            stencil_ops: None,
        }),
    });
    rpass.set_pipeline(hdr_line_pipeline);
    rpass.set_vertex_buffer(0, line_vertex_buffer.slice(..));
    rpass.set_vertex_buffer(1, line_instance_buffer.slice(..));
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

    let draw_count = if draw_changed {
        MAX_LINES
    } else {
        MAX_LINES / 2
    } as u32;
    rpass.draw(0..14, 0..draw_count);
    drop(rpass);

    // Tonemap HDR buffer to surface
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

    // Finish encoding
    command_buffers.write().push(encoder.finish());

    // Flip buffer flag
    *buffer_flip_flop.write() = !buffer_flip_state;
}
