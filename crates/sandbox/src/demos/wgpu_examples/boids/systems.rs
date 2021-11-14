use std::sync::atomic::{AtomicUsize, Ordering};

use super::{
    BackBufferBindGroupComponent, BackBufferComponent, Boids, ComputeShaderModuleComponent,
    DrawShaderModuleComponent, FrontBufferBindGroupComponent, FrontBufferComponent,
    UniformBufferComponent, VertexBufferComponent, NUM_PARTICLES,
};
use antigen_core::{GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock};

use antigen_wgpu::{
    wgpu::{
        vertex_attr_array, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingType, BufferAddress, BufferBindingType, BufferSize, Color,
        CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, Device,
        FragmentState, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor,
        PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        ShaderStages, VertexBufferLayout, VertexState, VertexStepMode,
    },
    CommandBuffersComponent, ComputePipelineComponent, RenderAttachmentTextureView,
    RenderPipelineComponent, SurfaceConfigurationComponent,
};

use legion::IntoQuery;

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn boids_prepare(
    world: &legion::world::SubWorld,
    _: &Boids,
    compute_shader: &ComputeShaderModuleComponent,
    draw_shader: &DrawShaderModuleComponent,
    render_pipeline_component: &RenderPipelineComponent,
    compute_pipeline_component: &ComputePipelineComponent,
    sim_param_buffer: &UniformBufferComponent,
    front_buffer: &FrontBufferComponent,
    back_buffer: &BackBufferComponent,
    front_buffer_bind_group: &FrontBufferBindGroupComponent,
    back_buffer_bind_group: &BackBufferBindGroupComponent,
    surface_configuration_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    if !render_pipeline_component.read().is_pending() {
        return;
    }

    let compute_shader = compute_shader.read();
    let compute_shader = if let LazyComponent::Ready(compute_shader) = &*compute_shader {
        compute_shader
    } else {
        return;
    };

    let draw_shader = draw_shader.read();
    let draw_shader = if let LazyComponent::Ready(draw_shader) = &*draw_shader {
        draw_shader
    } else {
        return;
    };

    let sim_param_buffer = sim_param_buffer.read();
    let sim_param_buffer = if let LazyComponent::Ready(sim_param_buffer) = &*sim_param_buffer {
        sim_param_buffer
    } else {
        return;
    };

    let front_buffer = front_buffer.read();
    let front_buffer = if let LazyComponent::Ready(front_buffer) = &*front_buffer {
        front_buffer
    } else {
        return;
    };

    let back_buffer = back_buffer.read();
    let back_buffer = if let LazyComponent::Ready(back_buffer) = &*back_buffer {
        back_buffer
    } else {
        return;
    };

    let surface_configuration_component =
        world.get_indirect(surface_configuration_component).unwrap();
    let config = surface_configuration_component.read();

    let device = <&Device>::query().iter(world).next().unwrap();

    let compute_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(
                        7 * std::mem::size_of::<f32>() as BufferAddress,
                    ),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new((NUM_PARTICLES * 16) as _),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new((NUM_PARTICLES * 16) as _),
                },
                count: None,
            },
        ],
        label: None,
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute"),
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });

    // create render pipeline with empty bind group layout

    let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("render"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&render_pipeline_layout),
        vertex: VertexState {
            module: &draw_shader,
            entry_point: "main",
            buffers: &[
                VertexBufferLayout {
                    array_stride: 4 * 4,
                    step_mode: VertexStepMode::Instance,
                    attributes: &vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                },
                VertexBufferLayout {
                    array_stride: 2 * 4,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![2 => Float32x2],
                },
            ],
        },
        fragment: Some(FragmentState {
            module: &draw_shader,
            entry_point: "main",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    render_pipeline_component.write().set_ready(render_pipeline);

    // create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Compute pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "main",
    });

    compute_pipeline_component
        .write()
        .set_ready(compute_pipeline);

    // buffer for the three 2d triangle vertices of each instance

    let front_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &compute_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: sim_param_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: front_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: back_buffer.as_entire_binding(), // bind to opposite buffer
            },
        ],
        label: None,
    });
    front_buffer_bind_group.write().set_ready(front_bind_group);

    let back_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &compute_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: sim_param_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: back_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: front_buffer.as_entire_binding(), // bind to opposite buffer
            },
        ],
        label: None,
    });
    back_buffer_bind_group.write().set_ready(back_bind_group);
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
pub fn boids_render(
    world: &legion::world::SubWorld,
    _: &Boids,
    render_pipeline: &RenderPipelineComponent,
    compute_pipeline: &ComputePipelineComponent,
    vertex_buffer: &VertexBufferComponent,
    front_buffer: &FrontBufferComponent,
    back_buffer: &BackBufferComponent,
    front_buffer_bind_group: &FrontBufferBindGroupComponent,
    back_buffer_bind_group: &BackBufferBindGroupComponent,
    command_buffers: &CommandBuffersComponent,
    texture_view: &IndirectComponent<RenderAttachmentTextureView>,
    #[state] frame_num_atomic: &AtomicUsize,
    #[state] work_group_count: &u32,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let render_pipeline = render_pipeline.read();
    let render_pipeline = if let LazyComponent::Ready(render_pipeline) = &*render_pipeline {
        render_pipeline
    } else {
        return;
    };

    let compute_pipeline = compute_pipeline.read();
    let compute_pipeline = if let LazyComponent::Ready(compute_pipeline) = &*compute_pipeline {
        compute_pipeline
    } else {
        return;
    };

    let vertex_buffer = vertex_buffer.read();
    let vertex_buffer = if let LazyComponent::Ready(vertex_buffer) = &*vertex_buffer {
        vertex_buffer
    } else {
        return;
    };

    let front_buffer = front_buffer.read();
    let front_buffer = if let LazyComponent::Ready(front_buffer) = &*front_buffer {
        front_buffer
    } else {
        return;
    };

    let back_buffer = back_buffer.read();
    let back_buffer = if let LazyComponent::Ready(back_buffer) = &*back_buffer {
        back_buffer
    } else {
        return;
    };

    let front_buffer_bind_group = front_buffer_bind_group.read();
    let front_buffer_bind_group =
        if let LazyComponent::Ready(front_buffer_bind_group) = &*front_buffer_bind_group {
            front_buffer_bind_group
        } else {
            return;
        };

    let back_buffer_bind_group = back_buffer_bind_group.read();
    let back_buffer_bind_group =
        if let LazyComponent::Ready(back_buffer_bind_group) = &*back_buffer_bind_group {
            back_buffer_bind_group
        } else {
            return;
        };

    let texture_view = world.get_indirect(texture_view).unwrap();

    let frame_num = frame_num_atomic.load(Ordering::Relaxed);

    let texture_view = texture_view.read();
    let texture_view = if let LazyComponent::Ready(texture_view) = &*texture_view {
        texture_view
    } else {
        return;
    };

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    encoder.push_debug_group("compute boid movement");
    {
        // compute pass
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(
            0,
            if frame_num % 2 == 0 {
                front_buffer_bind_group
            } else {
                back_buffer_bind_group
            },
            &[],
        );
        cpass.dispatch(*work_group_count, 1, 1);
    }
    encoder.pop_debug_group();

    encoder.push_debug_group("render boids");
    {
        // render pass
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&render_pipeline);
        // render dst particles
        rpass.set_vertex_buffer(
            0,
            if (frame_num + 1) % 2 == 0 {
                front_buffer.slice(..)
            } else {
                back_buffer.slice(..)
            },
        );
        // the three instance-local vertices
        rpass.set_vertex_buffer(1, vertex_buffer.slice(..));
        rpass.draw(0..3, 0..NUM_PARTICLES as u32);
    }
    encoder.pop_debug_group();

    // update frame count
    frame_num_atomic.fetch_add(1, Ordering::Relaxed);

    // done
    command_buffers.write().push(encoder.finish());
}
