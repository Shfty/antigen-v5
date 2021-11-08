use super::{
    Bunnies, Bunnymark, Global, Globals, Local, Locals, Logo, PlayfieldExtent, BUNNY_SIZE, GRAVITY,
};
use antigen_core::{
    ChangedFlag, GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock, SizeComponent,
};

use antigen_wgpu::{
    wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
        BindingResource, BindingType, BlendState, BufferAddress, BufferBinding, BufferBindingType,
        BufferSize, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, Device,
        DynamicOffset, FragmentState, LoadOp, MultisampleState, Operations,
        PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment,
        RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, SurfaceConfiguration,
        TextureSampleType, TextureViewDimension, VertexState,
    },
    BindGroupComponent, BufferComponent, CommandBuffersComponent, RenderAttachment,
    RenderPipelineComponent, SamplerComponent, ShaderModuleComponent, SurfaceComponent,
    TextureViewComponent,
};

use legion::IntoQuery;

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceComponent)]
pub fn bunnymark_prepare(
    world: &legion::world::SubWorld,
    _: &Bunnymark,
    shader_module: &ShaderModuleComponent<()>,
    render_pipeline_component: &RenderPipelineComponent<()>,
    global_buffer: &BufferComponent<Global>,
    local_buffer: &BufferComponent<Local>,
    texture_view: &TextureViewComponent<'static, Logo>,
    sampler: &SamplerComponent<'static, Logo>,
    global_bind_group: &BindGroupComponent<Global>,
    local_bind_group: &BindGroupComponent<Local>,
    surface_component: &IndirectComponent<SurfaceComponent>,
) {
    if !render_pipeline_component.read().is_pending() {
        return;
    }

    let device = <&Device>::query().iter(world).next().unwrap();
    let surface_component = world.get_indirect(surface_component).unwrap();
    let config = ReadWriteLock::<SurfaceConfiguration>::read(surface_component);

    let shader_module = shader_module.read();
    let shader_module = if let LazyComponent::Ready(shader_module) = &*shader_module {
        shader_module
    } else {
        return;
    };

    let global_buffer = global_buffer.read();
    let global_buffer = if let LazyComponent::Ready(global_buffer) = &*global_buffer {
        global_buffer
    } else {
        return;
    };

    let local_buffer = local_buffer.read();
    let local_buffer = if let LazyComponent::Ready(local_buffer) = &*local_buffer {
        local_buffer
    } else {
        return;
    };

    let texture_view = texture_view.read();
    let texture_view = if let LazyComponent::Ready(texture_view) = &*texture_view {
        texture_view
    } else {
        return;
    };

    let sampler = sampler.read();
    let sampler = if let LazyComponent::Ready(sampler) = &*sampler {
        sampler
    } else {
        return;
    };

    let global_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<Globals>() as _),
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
                ty: BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
                count: None,
            },
        ],
        label: None,
    });

    let local_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: BufferSize::new(std::mem::size_of::<Locals>() as _),
            },
            count: None,
        }],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&global_bind_group_layout, &local_bind_group_layout],
        push_constant_ranges: &[],
    });

    render_pipeline_component
        .write()
        .set_ready(device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::default(),
                }],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
        }));

    global_bind_group
        .write()
        .set_ready(device.create_bind_group(&BindGroupDescriptor {
            layout: &global_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
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
        }));

    local_bind_group
        .write()
        .set_ready(device.create_bind_group(&BindGroupDescriptor {
            layout: &local_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &local_buffer,
                    offset: 0,
                    size: BufferSize::new(std::mem::size_of::<Locals>() as _),
                }),
            }],
            label: None,
        }));
}

#[legion::system(par_for_each)]
pub fn bunnymark_tick(
    bunnies: &Bunnies,
    dirty_flag: &ChangedFlag<Bunnies>,
    extent: &SizeComponent<(u32, u32), PlayfieldExtent>,
) {
    let delta = 0.01;
    for bunny in bunnies.write().iter_mut() {
        bunny.position[0] += bunny.velocity[0] * delta;
        bunny.position[1] += bunny.velocity[1] * delta;
        bunny.velocity[1] += GRAVITY * delta;
        if (bunny.velocity[0] > 0.0
            && bunny.position[0] + 0.5 * BUNNY_SIZE > extent.read().0 as f32)
            || (bunny.velocity[0] < 0.0 && bunny.position[0] - 0.5 * BUNNY_SIZE < 0.0)
        {
            bunny.velocity[0] *= -1.0;
        }
        if bunny.velocity[1] < 0.0 && bunny.position[1] < 0.5 * BUNNY_SIZE {
            bunny.velocity[1] *= -1.0;
        }
    }
    dirty_flag.set(true);
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(TextureViewComponent<'static, RenderAttachment>)]
pub fn bunnymark_render(
    world: &legion::world::SubWorld,
    _: &Bunnymark,
    bunnies: &Bunnies,
    render_pipeline: &RenderPipelineComponent<()>,
    command_buffers: &CommandBuffersComponent,
    global_bind_group: &BindGroupComponent<Global>,
    local_bind_group: &BindGroupComponent<Local>,
    texture_view: &IndirectComponent<TextureViewComponent<'static, RenderAttachment>>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let global_bind_group = global_bind_group.read();
    let global_bind_group = if let LazyComponent::Ready(global_bind_group) = &*global_bind_group {
        global_bind_group
    } else {
        return;
    };

    let local_bind_group = local_bind_group.read();
    let local_bind_group = if let LazyComponent::Ready(local_bind_group) = &*local_bind_group {
        local_bind_group
    } else {
        return;
    };

    if let LazyComponent::Ready(render_pipeline) = &*render_pipeline.read() {
        let texture_view = world.get_indirect(texture_view).unwrap();

        if let LazyComponent::Ready(texture_view) = &*texture_view.read() {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let clear_color = Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                };
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[RenderPassColorAttachment {
                        view: &texture_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(clear_color),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });
                rpass.set_pipeline(&render_pipeline);
                rpass.set_bind_group(0, &global_bind_group, &[]);

                let uniform_alignment =
                    device.limits().min_uniform_buffer_offset_alignment as BufferAddress;
                for i in 0..bunnies.read().len() {
                    let offset = (i as DynamicOffset) * (uniform_alignment as DynamicOffset);
                    rpass.set_bind_group(1, &local_bind_group, &[offset]);
                    rpass.draw(0..4, 0..1);
                }
            }

            command_buffers.write().push(encoder.finish());
        }
    }
}
