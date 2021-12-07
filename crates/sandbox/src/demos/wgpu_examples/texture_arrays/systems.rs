use std::num::NonZeroU32;

use super::{
    FragmentShaderComponent, GreenTextureViewComponent, IndexBufferComponent,
    RedTextureViewComponent, TextureArrays, UniformWorkaroundComponent, Vertex,
    VertexBufferComponent, VertexShaderComponent, INDEX_FORMAT,
};
use antigen_core::{GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock};

use antigen_wgpu::{
    wgpu::{
        include_spirv_raw, vertex_attr_array, BindGroupDescriptor, BindGroupEntry,
        BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType,
        BufferAddress, Color, CommandEncoderDescriptor, Device, Features, FragmentState, FrontFace,
        LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PrimitiveState,
        PushConstantRange, RenderPassColorAttachment, RenderPassDescriptor,
        RenderPipelineDescriptor, ShaderStages, TextureSampleType, TextureViewDimension,
        VertexBufferLayout, VertexState, VertexStepMode,
    },
    BindGroupComponent, CommandBuffersComponent, RenderAttachmentTextureView,
    RenderPipelineComponent, SamplerComponent, SurfaceConfigurationComponent,
};

use legion::IntoQuery;

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn texture_arrays_prepare(
    world: &legion::world::SubWorld,
    _: &TextureArrays,
    render_pipeline_component: &RenderPipelineComponent,
    bind_group_component: &BindGroupComponent,
    vertex_shader: &VertexShaderComponent,
    fragment_shader: &FragmentShaderComponent,
    red_texture_view: &RedTextureViewComponent,
    green_texture_view: &GreenTextureViewComponent,
    sampler: &SamplerComponent,
    uniform_workaround: &UniformWorkaroundComponent,
    surface_configuration_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    if !render_pipeline_component.read().is_pending() {
        return;
    }

    let device = <&Device>::query().iter(world).next().unwrap();

    let red_texture_view = red_texture_view.read();
    let red_texture_view = if let LazyComponent::Ready(red_texture_view) = &*red_texture_view {
        red_texture_view
    } else {
        return;
    };

    let green_texture_view = green_texture_view.read();
    let green_texture_view = if let LazyComponent::Ready(green_texture_view) = &*green_texture_view
    {
        green_texture_view
    } else {
        return;
    };

    let sampler = sampler.read();
    let sampler = if let LazyComponent::Ready(sampler) = &*sampler {
        sampler
    } else {
        return;
    };

    let vertex_shader = vertex_shader.read();
    let vertex_shader = if let LazyComponent::Ready(vertex_shader) = &*vertex_shader {
        vertex_shader
    } else {
        return;
    };

    // Fragment shader
    let fs_source = match device.features() {
        f if f.contains(Features::UNSIZED_BINDING_ARRAY) => {
            include_spirv_raw!("unsized-non-uniform.frag.spv")
        }
        f if f
            .contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING) =>
        {
            include_spirv_raw!("non-uniform.frag.spv")
        }
        f if f.contains(Features::TEXTURE_BINDING_ARRAY) => {
            *uniform_workaround.write() = true;
            include_spirv_raw!("uniform.frag.spv")
        }
        _ => unreachable!(),
    };

    fragment_shader
        .write()
        .set_ready(unsafe { device.create_shader_module_spirv(&fs_source) });

    let fragment_shader = fragment_shader.read();
    let fragment_shader = if let LazyComponent::Ready(fragment_shader) = &*fragment_shader {
        fragment_shader
    } else {
        unreachable!()
    };

    let surface_configuration_component =
        world.get_indirect(surface_configuration_component).unwrap();
    let config = surface_configuration_component.read();

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bind group layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(2),
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureViewArray(&[
                    &red_texture_view,
                    &green_texture_view,
                ]),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ],
        layout: &bind_group_layout,
        label: Some("bind group"),
    });
    bind_group_component.write().set_ready(bind_group);

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("main"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: if *uniform_workaround.read() {
            &[PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                range: 0..4,
            }]
        } else {
            &[]
        },
    });

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &vertex_shader,
            entry_point: "main",
            buffers: &[VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as BufferAddress,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Sint32],
            }],
        },
        fragment: Some(FragmentState {
            module: &fragment_shader,
            entry_point: "main",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState {
            front_face: FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });
    render_pipeline_component.write().set_ready(pipeline);
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
pub fn texture_arrays_render(
    world: &legion::world::SubWorld,
    _: &TextureArrays,
    render_pipeline: &RenderPipelineComponent,
    bind_group: &BindGroupComponent,
    vertex_buffer: &VertexBufferComponent,
    index_buffer: &IndexBufferComponent,
    uniform_workaround: &UniformWorkaroundComponent,
    command_buffers: &CommandBuffersComponent,
    texture_view: &IndirectComponent<RenderAttachmentTextureView>,
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

    let bind_group = bind_group.read();
    let bind_group = if let LazyComponent::Ready(bind_group) = &*bind_group {
        bind_group
    } else {
        return;
    };

    let vertex_buffer = vertex_buffer.read();
    let vertex_buffer = if let LazyComponent::Ready(vertex_buffer) = &*vertex_buffer {
        vertex_buffer
    } else {
        return;
    };

    let index_buffer = index_buffer.read();
    let index_buffer = if let LazyComponent::Ready(index_buffer) = &*index_buffer {
        index_buffer
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

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
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

    rpass.set_pipeline(&render_pipeline);
    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
    rpass.set_index_buffer(index_buffer.slice(..), INDEX_FORMAT);
    if *uniform_workaround.read() {
        rpass.set_push_constants(ShaderStages::FRAGMENT, 0, bytemuck::cast_slice(&[0]));
        rpass.draw_indexed(0..6, 0, 0..1);
        rpass.set_push_constants(ShaderStages::FRAGMENT, 0, bytemuck::cast_slice(&[1]));
        rpass.draw_indexed(6..12, 0, 0..1);
    } else {
        rpass.draw_indexed(0..12, 0, 0..1);
    }

    drop(rpass);

    command_buffers.write().push(encoder.finish());
}
