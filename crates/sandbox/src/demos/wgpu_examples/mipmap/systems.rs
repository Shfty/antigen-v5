use std::{borrow::Cow, num::NonZeroU32};

use crate::wgpu_examples::mipmap::{MIP_LEVEL_COUNT, TEXTURE_FORMAT};

use super::{DrawPipelineComponent, DrawShaderComponent, JuliaSetSamplerComponent, JuliaSetTextureComponent, JuliaSetTextureViewComponent, MIP_PASS_COUNT, Mipmap, UniformBufferComponent, ViewProjectionMatrix};
use antigen_core::{Changed, ChangedTrait, GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock};

use antigen_wgpu::{
    wgpu::{
        AddressMode, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferAddress,
        BufferDescriptor, BufferUsages, Color, CommandEncoder, CommandEncoderDescriptor, Device,
        Face, Features, FilterMode, FragmentState, FrontFace, IndexFormat, LoadOp, Maintain,
        MapMode, MultisampleState, Operations, PipelineStatisticsTypes, PrimitiveState,
        PrimitiveTopology, QuerySet, QuerySetDescriptor, QueryType, Queue,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, Texture, TextureAspect,
        TextureViewDescriptor, VertexState, QUERY_RESOLVE_BUFFER_ALIGNMENT,
    },
    BindGroupComponent, CommandBuffersComponent, RenderAttachmentTextureView,
    SurfaceConfigurationComponent,
};

use legion::{IntoQuery, world::SubWorld};

use bytemuck::{Pod, Zeroable};

struct QuerySets {
    timestamp: QuerySet,
    timestamp_period: f32,
    pipeline_statistics: QuerySet,
    data_buffer: Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TimestampData {
    start: u64,
    end: u64,
}

type TimestampQueries = [TimestampData; MIP_PASS_COUNT as usize];
type PipelineStatisticsQueries = [u64; MIP_PASS_COUNT as usize];

fn pipeline_statistics_offset() -> BufferAddress {
    (std::mem::size_of::<TimestampQueries>() as BufferAddress).max(QUERY_RESOLVE_BUFFER_ALIGNMENT)
}

fn generate_mipmaps(
    encoder: &mut CommandEncoder,
    device: &Device,
    texture: &Texture,
    query_sets: &Option<QuerySets>,
    mip_count: u32,
) {
    let shader = device.create_shader_module(&ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("blit.wgsl"))),
    });

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("blit"),
        layout: None,
        vertex: VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[TEXTURE_FORMAT.into()],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            strip_index_format: Some(IndexFormat::Uint16),
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("mip"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..Default::default()
    });

    let views = (0..mip_count)
        .map(|mip| {
            texture.create_view(&TextureViewDescriptor {
                label: Some("mip"),
                format: None,
                dimension: None,
                aspect: TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: NonZeroU32::new(1),
                base_array_layer: 0,
                array_layer_count: None,
            })
        })
        .collect::<Vec<_>>();

    for target_mip in 1..mip_count as usize {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&views[target_mip - 1]),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        let pipeline_query_index_base = target_mip as u32 - 1;
        let timestamp_query_index_base = (target_mip as u32 - 1) * 2;

        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[RenderPassColorAttachment {
                view: &views[target_mip],
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::WHITE),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        if let Some(ref query_sets) = query_sets {
            rpass.write_timestamp(&query_sets.timestamp, timestamp_query_index_base);
            rpass.begin_pipeline_statistics_query(
                &query_sets.pipeline_statistics,
                pipeline_query_index_base,
            );
        }
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..4, 0..1);
        if let Some(ref query_sets) = query_sets {
            rpass.write_timestamp(&query_sets.timestamp, timestamp_query_index_base + 1);
            rpass.end_pipeline_statistics_query();
        }
    }

    if let Some(ref query_sets) = query_sets {
        let timestamp_query_count = MIP_PASS_COUNT * 2;
        println!("Resolving timestamp queries");
        encoder.resolve_query_set(
            &query_sets.timestamp,
            0..timestamp_query_count,
            &query_sets.data_buffer,
            0,
        );
        println!("Resolving pipeline statistics queries");
        encoder.resolve_query_set(
            &query_sets.pipeline_statistics,
            0..MIP_PASS_COUNT,
            &query_sets.data_buffer,
            pipeline_statistics_offset(),
        );
    }
}

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(Queue)]
#[read_component(Changed<SurfaceConfigurationComponent>)]
#[read_component(RenderAttachmentTextureView)]
pub fn mipmap_prepare(
    world: &legion::world::SubWorld,
    _: &Mipmap,
    draw_pipeline_component: &DrawPipelineComponent,
    bind_group_component: &BindGroupComponent,
    draw_shader: &DrawShaderComponent,
    uniform_buffer: &UniformBufferComponent,
    julia_set_texture: &JuliaSetTextureComponent,
    julia_set_texture_view: &JuliaSetTextureViewComponent,
    julia_set_sampler: &JuliaSetSamplerComponent,
    surface_component: &IndirectComponent<Changed<SurfaceConfigurationComponent>>,
) {
    if !draw_pipeline_component.read().is_pending() {
        return;
    }

    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let queue = if let Some(components) = <&Queue>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let surface_component = world.get_indirect(surface_component).unwrap();
    let config = surface_component.read();

    let julia_set_texture_view = julia_set_texture_view.read();
    let julia_set_texture_view =
        if let LazyComponent::Ready(julia_set_texture_view) = &*julia_set_texture_view {
            julia_set_texture_view
        } else {
            return;
        };

    let draw_shader = draw_shader.read();
    let draw_shader = if let LazyComponent::Ready(draw_shader) = &*draw_shader {
        draw_shader
    } else {
        return;
    };

    let uniform_buffer = uniform_buffer.read();
    let uniform_buffer = if let LazyComponent::Ready(uniform_buffer) = &*uniform_buffer {
        uniform_buffer
    } else {
        return;
    };

    let julia_set_texture = julia_set_texture.read();
    let julia_set_texture = if let LazyComponent::Ready(julia_set_texture) = &*julia_set_texture {
        julia_set_texture
    } else {
        return;
    };

    let julia_set_sampler = julia_set_sampler.read();
    let julia_set_sampler = if let LazyComponent::Ready(julia_set_sampler) = &*julia_set_sampler {
        julia_set_sampler
    } else {
        return;
    };

    // Create the texture
    let mut init_encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    // Create the render pipeline
    let draw_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("draw"),
        layout: None,
        vertex: VertexState {
            module: &draw_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &draw_shader,
            entry_point: "fs_main",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            strip_index_format: Some(IndexFormat::Uint16),
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    // Create bind group
    let bind_group_layout = draw_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&julia_set_texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&julia_set_sampler),
            },
        ],
        label: None,
    });

    draw_pipeline_component.write().set_ready(draw_pipeline);
    bind_group_component.write().set_ready(bind_group);

    // If both kinds of query are supported, use queries
    let query_sets = if device
        .features()
        .contains(Features::TIMESTAMP_QUERY | Features::PIPELINE_STATISTICS_QUERY)
    {
        // For N total mips, it takes N - 1 passes to generate them, and we're measuring those.
        let mip_passes = MIP_LEVEL_COUNT - 1;

        // Create the timestamp query set. We need twice as many queries as we have passes,
        // as we need a query at the beginning and at the end of the operation.
        let timestamp = device.create_query_set(&QuerySetDescriptor {
            label: None,
            count: mip_passes * 2,
            ty: QueryType::Timestamp,
        });
        // Timestamp queries use an device-specific timestamp unit. We need to figure out how many
        // nanoseconds go by for the timestamp to be incremented by one. The period is this value.
        let timestamp_period = queue.get_timestamp_period();

        // We only need one pipeline statistics query per pass.
        let pipeline_statistics = device.create_query_set(&QuerySetDescriptor {
            label: None,
            count: mip_passes,
            ty: QueryType::PipelineStatistics(PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS),
        });

        // This databuffer has to store all of the query results, 2 * passes timestamp queries
        // and 1 * passes statistics queries. Each query returns a u64 value.
        let data_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("query buffer"),
            size: pipeline_statistics_offset()
                + std::mem::size_of::<PipelineStatisticsQueries>() as BufferAddress,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Some(QuerySets {
            timestamp,
            timestamp_period,
            pipeline_statistics,
            data_buffer,
        })
    } else {
        None
    };

    generate_mipmaps(
        &mut init_encoder,
        device,
        &julia_set_texture,
        &query_sets,
        MIP_LEVEL_COUNT,
    );

    queue.submit(Some(init_encoder.finish()));

    if let Some(ref query_sets) = query_sets {
        // We can ignore the future as we're about to wait for the device.
        let _ = query_sets.data_buffer.slice(..).map_async(MapMode::Read);
        // Wait for device to be done rendering mipmaps
        device.poll(Maintain::Wait);
        // This is guaranteed to be ready.
        let timestamp_view = query_sets
            .data_buffer
            .slice(..std::mem::size_of::<TimestampQueries>() as BufferAddress)
            .get_mapped_range();
        let pipeline_stats_view = query_sets
            .data_buffer
            .slice(pipeline_statistics_offset()..)
            .get_mapped_range();
        // Convert the raw data into a useful structure
        let timestamp_data: &TimestampQueries = bytemuck::from_bytes(&*timestamp_view);
        let pipeline_stats_data: &PipelineStatisticsQueries =
            bytemuck::from_bytes(&*pipeline_stats_view);
        // Iterate over the data
        for (idx, (timestamp, pipeline)) in timestamp_data
            .iter()
            .zip(pipeline_stats_data.iter())
            .enumerate()
        {
            // Figure out the timestamp differences and multiply by the period to get nanoseconds
            let nanoseconds =
                (timestamp.end - timestamp.start) as f32 * query_sets.timestamp_period;
            // Nanoseconds is a bit small, so lets use microseconds.
            let microseconds = nanoseconds / 1000.0;
            // Print the data!
            println!(
                "Generating mip level {} took {:.3} μs and called the fragment shader {} times",
                idx + 1,
                microseconds,
                pipeline
            );
        }
    }
}

#[legion::system(par_for_each)]
#[read_component(Changed<SurfaceConfigurationComponent>)]
pub fn mipmap_resize(
    world: &SubWorld,
    _: &Mipmap,
    surface_config: &IndirectComponent<Changed<SurfaceConfigurationComponent>>,
    view_projection: &Changed<ViewProjectionMatrix>,
) {
    let surface_config = world.get_indirect(surface_config).unwrap();

    if surface_config.get_changed() {
        let surface_config = surface_config.read();
        let aspect = surface_config.width as f32 / surface_config.height as f32;
        let matrix = super::generate_matrix(aspect);
        view_projection.write().copy_from_slice(matrix.as_slice());
        view_projection.set_changed(true);
    }
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
pub fn mipmap_render(
    world: &legion::world::SubWorld,
    _: &Mipmap,
    draw_pipeline: &DrawPipelineComponent,
    bind_group: &BindGroupComponent,
    command_buffers: &CommandBuffersComponent,
    texture_view: &IndirectComponent<RenderAttachmentTextureView>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let draw_pipeline = draw_pipeline.read();
    let draw_pipeline = if let LazyComponent::Ready(draw_pipeline) = &*draw_pipeline {
        draw_pipeline
    } else {
        return;
    };

    let bind_group = bind_group.read();
    let bind_group = if let LazyComponent::Ready(bind_group) = &*bind_group {
        bind_group
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

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

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
    rpass.set_pipeline(&draw_pipeline);
    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.draw(0..4, 0..1);
    drop(rpass);

    command_buffers.write().push(encoder.finish());
}
