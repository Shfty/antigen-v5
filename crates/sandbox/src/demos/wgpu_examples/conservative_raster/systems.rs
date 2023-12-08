use crate::wgpu_examples::conservative_raster::RENDER_TARGET_FORMAT;

use super::{
    ConservativeRaster, LinesPipelineComponent, LowResSamplerComponent, LowResTarget,
    LowResTextureDescriptorComponent, LowResTextureViewComponent, TriangleAndLinesShaderComponent,
    TriangleConservativePipelineComponent, TriangleRegularPipelineComponent,
    UpscaleBindGroupComponent, UpscaleBindGroupLayoutComponent, UpscalePipelineComponent,
    UpscaleShaderComponent,
};
use antigen_core::{
    ChangedTrait, GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock, Usage,
};

use antigen_wgpu::{
    wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
        BindingResource, BindingType, Color, CommandEncoderDescriptor, Device, Extent3d, Features,
        FragmentState, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PolygonMode,
        PrimitiveState, PrimitiveTopology, RenderPassColorAttachment, RenderPassDescriptor,
        RenderPipelineDescriptor, SamplerBindingType, ShaderStages, TextureSampleType,
        TextureViewDimension, VertexState,
    },
    CommandBuffersComponent, RenderAttachmentTextureView, SurfaceConfigurationComponent,
    TextureViewDescriptorComponent,
};

use legion::{world::SubWorld, IntoQuery};

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn conservative_raster_prepare(
    world: &legion::world::SubWorld,
    _: &ConservativeRaster,
    shader_triangle_and_lines: &TriangleAndLinesShaderComponent,
    shader_upscale: &UpscaleShaderComponent,
    pipeline_conservative_component: &TriangleConservativePipelineComponent,
    pipeline_regular_component: &TriangleRegularPipelineComponent,
    pipeline_upscale_component: &UpscalePipelineComponent,
    pipeline_lines_component: &LinesPipelineComponent,
    bind_group_layout_upscale_component: &UpscaleBindGroupLayoutComponent,
    bind_group_upscale_component: &UpscaleBindGroupComponent,
    low_res_view: &LowResTextureViewComponent,
    low_res_sampler: &LowResSamplerComponent,
    surface_config_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    let device = <&Device>::query().iter(world).next().unwrap();

    if bind_group_layout_upscale_component.read().is_pending() {
        let bind_group_layout_upscale =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("upscale bindgroup"),
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
                        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        bind_group_layout_upscale_component
            .write()
            .set_ready(bind_group_layout_upscale);

        println!("Created upscale bind group");
    }

    let bind_group_layout_upscale = bind_group_layout_upscale_component.read();
    let bind_group_layout_upscale =
        if let LazyComponent::Ready(bind_group_layout_upscale) = &*bind_group_layout_upscale {
            bind_group_layout_upscale
        } else {
            unreachable!();
        };

    let low_res_view = low_res_view.read();
    let low_res_view = if let LazyComponent::Ready(low_res_view) = &*low_res_view {
        low_res_view
    } else {
        return;
    };

    let low_res_sampler = low_res_sampler.read();
    let low_res_sampler = if let LazyComponent::Ready(low_res_sampler) = &*low_res_sampler {
        low_res_sampler
    } else {
        return;
    };

    let shader_upscale = shader_upscale.read();
    let shader_upscale = if let LazyComponent::Ready(shader_upscale) = &*shader_upscale {
        shader_upscale
    } else {
        return;
    };

    // Create low-res target
    if bind_group_upscale_component.read().is_pending() {
        let low_res_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("upscale bind group"),
            layout: &bind_group_layout_upscale,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&low_res_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&low_res_sampler),
                },
            ],
        });

        bind_group_upscale_component
            .write()
            .set_ready(low_res_bind_group);

        println!("Created low-res target");
    }

    if !pipeline_conservative_component.read().is_pending() {
        return;
    }

    let surface_configuration_component = world.get_indirect(surface_config_component).unwrap();
    let config = surface_configuration_component.read();

    let shader_triangle_and_lines = shader_triangle_and_lines.read();
    let shader_triangle_and_lines =
        if let LazyComponent::Ready(shader_triangle_and_lines) = &*shader_triangle_and_lines {
            shader_triangle_and_lines
        } else {
            return;
        };

    let pipeline_layout_empty = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let pipeline_conservative = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Conservative Rasterization"),
        layout: Some(&pipeline_layout_empty),
        vertex: VertexState {
            module: &shader_triangle_and_lines,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader_triangle_and_lines,
            entry_point: "fs_main_red",
            targets: &[RENDER_TARGET_FORMAT.into()],
        }),
        primitive: PrimitiveState {
            conservative: true,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
    });
    pipeline_conservative_component
        .write()
        .set_ready(pipeline_conservative);

    let pipeline_regular = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Regular Rasterization"),
        layout: Some(&pipeline_layout_empty),
        vertex: VertexState {
            module: &shader_triangle_and_lines,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader_triangle_and_lines,
            entry_point: "fs_main_blue",
            targets: &[RENDER_TARGET_FORMAT.into()],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
    });
    pipeline_regular_component
        .write()
        .set_ready(pipeline_regular);

    if device.features().contains(Features::POLYGON_MODE_LINE) {
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Lines"),
            layout: Some(&pipeline_layout_empty),
            vertex: VertexState {
                module: &shader_triangle_and_lines,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader_triangle_and_lines,
                entry_point: "fs_main_white",
                targets: &[config.format.into()],
            }),
            primitive: PrimitiveState {
                polygon_mode: PolygonMode::Line,
                topology: PrimitiveTopology::LineStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        pipeline_lines_component.write().set_ready(pipeline);
    }

    let pipeline_upscale = {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_upscale],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Upscale"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_upscale,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader_upscale,
                entry_point: "fs_main",
                targets: &[config.format.into()],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        })
    };

    pipeline_upscale_component
        .write()
        .set_ready(pipeline_upscale);

    println!("Initialized conservative raster renderer");
}

#[legion::system(par_for_each)]
#[read_component(SurfaceConfigurationComponent)]
#[read_component(LowResTextureDescriptorComponent<'static>)]
#[read_component(Usage<LowResTarget, TextureViewDescriptorComponent<'static>>)]
pub fn conservative_raster_resize(
    world: &SubWorld,
    _: &ConservativeRaster,
    surface_config: &IndirectComponent<SurfaceConfigurationComponent>,
    bind_group_upscale_component: &UpscaleBindGroupComponent,
    low_res_desc: &IndirectComponent<LowResTextureDescriptorComponent<'static>>,
    low_res_view_desc: &IndirectComponent<
        Usage<LowResTarget, TextureViewDescriptorComponent<'static>>,
    >,
) {
    let surface_config = world.get_indirect(surface_config).unwrap();

    let low_res_desc = world.get_indirect(low_res_desc).unwrap();
    let low_res_view_desc = world.get_indirect(low_res_view_desc).unwrap();

    if !surface_config.get_changed() {
        return;
    }

    println!("Surface config changed, recreating low-res target, view and bind group");

    let surface_config = surface_config.read();
    low_res_desc.write().size = Extent3d {
        width: surface_config.width / 16,
        height: surface_config.height / 16,
        depth_or_array_layers: 1,
    };

    low_res_desc.set_changed(true);
    low_res_view_desc.set_changed(true);
    bind_group_upscale_component.write().set_pending();
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
pub fn conservative_raster_render(
    world: &legion::world::SubWorld,
    _: &ConservativeRaster,
    pipeline_triangle_conservative: &TriangleConservativePipelineComponent,
    pipeline_triangle_regular: &TriangleRegularPipelineComponent,
    pipeline_upscale: &UpscalePipelineComponent,
    pipeline_lines: &LinesPipelineComponent,
    bind_group_upscale: &UpscaleBindGroupComponent,
    command_buffers: &CommandBuffersComponent,
    low_res_view: &LowResTextureViewComponent,
    render_attachment_view: &IndirectComponent<RenderAttachmentTextureView>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let pipeline_triangle_conservative = pipeline_triangle_conservative.read();
    let pipeline_triangle_conservative =
        if let LazyComponent::Ready(pipeline_triangle_conservative) =
            &*pipeline_triangle_conservative
        {
            pipeline_triangle_conservative
        } else {
            return;
        };

    let pipeline_triangle_regular = pipeline_triangle_regular.read();
    let pipeline_triangle_regular =
        if let LazyComponent::Ready(pipeline_triangle_regular) = &*pipeline_triangle_regular {
            pipeline_triangle_regular
        } else {
            return;
        };

    let pipeline_upscale = pipeline_upscale.read();
    let pipeline_upscale = if let LazyComponent::Ready(pipeline_upscale) = &*pipeline_upscale {
        pipeline_upscale
    } else {
        return;
    };

    let bind_group_upscale = bind_group_upscale.read();
    let bind_group_upscale = if let LazyComponent::Ready(bind_group_upscale) = &*bind_group_upscale
    {
        bind_group_upscale
    } else {
        return;
    };

    let texture_view = world.get_indirect(render_attachment_view).unwrap();
    let texture_view = texture_view.read();
    let texture_view = if let LazyComponent::Ready(texture_view) = &*texture_view {
        texture_view
    } else {
        return;
    };

    let low_res_view = low_res_view.read();
    let low_res_view = if let LazyComponent::Ready(low_res_view) = &*low_res_view {
        low_res_view
    } else {
        return;
    };

    let pipeline_lines = pipeline_lines.read();

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("primary"),
    });

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("low resolution"),
        color_attachments: &[RenderPassColorAttachment {
            view: &low_res_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });

    rpass.set_pipeline(&pipeline_triangle_conservative);
    rpass.draw(0..3, 0..1);
    rpass.set_pipeline(&pipeline_triangle_regular);
    rpass.draw(0..3, 0..1);
    drop(rpass);

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("full resolution"),
        color_attachments: &[RenderPassColorAttachment {
            view: &texture_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });

    rpass.set_pipeline(&pipeline_upscale);
    rpass.set_bind_group(0, &bind_group_upscale, &[]);
    rpass.draw(0..3, 0..1);

    if let LazyComponent::Ready(pipeline_lines) = &*pipeline_lines {
        rpass.set_pipeline(pipeline_lines);
        rpass.draw(0..4, 0..1);
    };
    drop(rpass);

    command_buffers.write().push(encoder.finish());
}
