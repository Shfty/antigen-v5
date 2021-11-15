use std::borrow::Cow;

use crate::wgpu_examples::conservative_raster::RENDER_TARGET_FORMAT;

use super::{
    ConservativeRaster, LinesPipelineComponent, LowResTextureViewComponent,
    TriangleConservativePipelineComponent, TriangleRegularPipelineComponent,
    UpscaleBindGroupComponent, UpscalePipelineComponent,
};
use antigen_core::{GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock};

use antigen_wgpu::{CommandBuffersComponent, RenderAttachmentTextureView, RenderPipelineComponent, ShaderModuleComponent, SurfaceConfigurationComponent, wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Color, CommandEncoderDescriptor, Device, Extent3d, Features, FilterMode, FragmentState, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, TextureDescriptor, TextureDimension, TextureSampleType, TextureUsages, TextureViewDimension, VertexState}};

use legion::IntoQuery;

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn conservative_raster_prepare(
    world: &legion::world::SubWorld,
    _: &ConservativeRaster,
    shader_module: &ShaderModuleComponent,
    render_pipeline_component: &RenderPipelineComponent,
    surface_config_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    if !render_pipeline_component.read().is_pending() {
        return;
    }
    let device = <&Device>::query().iter(world).next().unwrap();

    let surface_configuration_component =
        world.get_indirect(surface_config_component).unwrap();
    let config = surface_configuration_component.read();

    let pipeline_layout_empty = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let shader_triangle_and_lines = device.create_shader_module(&ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("triangle_and_lines.wgsl"))),
    });

    let pipeline_triangle_conservative = device.create_render_pipeline(&RenderPipelineDescriptor {
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
    });

    let pipeline_triangle_regular = device.create_render_pipeline(&RenderPipelineDescriptor {
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
    });

    let pipeline_lines = if device.features().contains(Features::POLYGON_MODE_LINE) {
        Some(device.create_render_pipeline(&RenderPipelineDescriptor {
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
        }))
    } else {
        None
    };

    let (pipeline_upscale, bind_group_layout_upscale) = {
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
                    ty: BindingType::Sampler {
                        filtering: false,
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
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("upscale.wgsl"))),
        });
        (
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Upscale"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[config.format.into()],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
            }),
            bind_group_layout,
        )
    };

    // Create low-res target

    let texture_view = device
        .create_texture(&TextureDescriptor {
            label: Some("Low Resolution Target"),
            size: Extent3d {
                width: config.width / 16,
                height: config.width / 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: RENDER_TARGET_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        })
        .create_view(&Default::default());

    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("Nearest Neighbor Sampler"),
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("upscale bind group"),
        layout: &bind_group_layout_upscale,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ],
    });
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
    texture_view: &IndirectComponent<RenderAttachmentTextureView>,
    low_res_target: &IndirectComponent<LowResTextureViewComponent>,
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

    let pipeline_lines = pipeline_lines.read();

    let bind_group_upscale = bind_group_upscale.read();
    let bind_group_upscale = if let LazyComponent::Ready(bind_group_upscale) = &*bind_group_upscale
    {
        bind_group_upscale
    } else {
        return;
    };

    let texture_view = world.get_indirect(texture_view).unwrap();
    let texture_view = texture_view.read();
    let texture_view = if let LazyComponent::Ready(texture_view) = &*texture_view {
        texture_view
    } else {
        return;
    };

    let low_res_target = world.get_indirect(low_res_target).unwrap();
    let low_res_target = low_res_target.read();
    let low_res_target = if let LazyComponent::Ready(low_res_target) = &*low_res_target {
        low_res_target
    } else {
        return;
    };

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("primary"),
    });

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("low resolution"),
        color_attachments: &[RenderPassColorAttachment {
            view: &low_res_target,
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
