use super::{
    Cube, IndexBufferComponent, MandelbrotTextureViewComponent, OpaquePassRenderPipelineComponent,
    UniformBufferComponent, VertexBufferComponent, ViewProjectionMatrix,
    WirePassRenderPipelineComponent,
};

use antigen_core::{
    Changed, ChangedFlag, ChangedTrait, GetIndirect, IndirectComponent, LazyComponent,
    ReadWriteLock,
};
use antigen_wgpu::{
    wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
        BindingResource, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState,
        BufferAddress, BufferBindingType, BufferSize, Color, ColorTargetState, ColorWrites,
        CommandEncoderDescriptor, Device, Face, Features, FragmentState, FrontFace, IndexFormat,
        LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PolygonMode,
        PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        ShaderStages, TextureSampleType, TextureViewDimension, VertexAttribute, VertexBufferLayout,
        VertexFormat, VertexState, VertexStepMode,
    },
    BindGroupComponent, CommandBuffersComponent, RenderAttachmentTextureView,
    ShaderModuleComponent, SurfaceConfigurationComponent,
};

use legion::{world::SubWorld, IntoQuery};

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(Changed<SurfaceConfigurationComponent>)]
pub fn cube_prepare(
    world: &SubWorld,
    _: &Cube,
    shader_module: &ShaderModuleComponent,
    opaque_pipeline_component: &OpaquePassRenderPipelineComponent,
    wire_pipeline_component: &WirePassRenderPipelineComponent,
    texture_view: &MandelbrotTextureViewComponent,
    uniform_buffer: &UniformBufferComponent,
    bind_group_component: &BindGroupComponent,
    surface_configuration_component: &IndirectComponent<Changed<SurfaceConfigurationComponent>>,
) {
    if !opaque_pipeline_component.read().is_pending() {
        return;
    }

    let shader_module = shader_module.read();
    let shader_module = if let LazyComponent::Ready(shader_module) = &*shader_module {
        shader_module
    } else {
        return;
    };

    let texture_view = texture_view.read();
    let texture_view = if let LazyComponent::Ready(texture_view) = &*texture_view {
        texture_view
    } else {
        return;
    };

    let uniform_buffer = uniform_buffer.read();
    let uniform_buffer = if let LazyComponent::Ready(uniform_buffer) = &*uniform_buffer {
        uniform_buffer
    } else {
        return;
    };

    let surface_configuration_component =
        world.get_indirect(surface_configuration_component).unwrap();
    let config = surface_configuration_component.read();

    let device = <&Device>::query().iter(world).next().unwrap();

    // Create pipeline layout
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    sample_type: TextureSampleType::Uint,
                    view_dimension: TextureViewDimension::D2,
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

    // Create bind group
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(texture_view),
            },
        ],
        label: None,
    });
    bind_group_component.write().set_ready(bind_group);

    let vertex_buffers = [
        VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        },
        VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 0,
                shader_location: 1,
            }],
        },
    ];

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader_module,
            entry_point: "vs_main",
            buffers: &vertex_buffers,
        },
        fragment: Some(FragmentState {
            module: &shader_module,
            entry_point: "fs_main",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState {
            cull_mode: Some(Face::Back),
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    if device.features().contains(Features::POLYGON_MODE_LINE) {
        let pipeline_wire = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &vertex_buffers,
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "fs_wire",
                targets: &[ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent::REPLACE,
                    }),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Line,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
        });
        wire_pipeline_component.write().set_ready(pipeline_wire);
    };

    opaque_pipeline_component.write().set_ready(pipeline);
}

#[legion::system(par_for_each)]
#[read_component(Changed<SurfaceConfigurationComponent>)]
pub fn cube_resize(
    world: &SubWorld,
    _: &Cube,
    surface_config: &IndirectComponent<Changed<SurfaceConfigurationComponent>>,
    view_projection: &ViewProjectionMatrix,
    matrix_dirty: &ChangedFlag<ViewProjectionMatrix>,
) {
    let surface_config = world.get_indirect(surface_config).unwrap();

    if surface_config.get_changed() {
        let surface_config = surface_config.read();
        let aspect = surface_config.width as f32 / surface_config.height as f32;
        let matrix = super::generate_matrix(aspect);
        view_projection.write().copy_from_slice(matrix.as_slice());
        matrix_dirty.set(true);
    }
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
pub fn cube_render(
    world: &SubWorld,
    _: &Cube,
    opaque_pipeline: &OpaquePassRenderPipelineComponent,
    wire_pipeline: &WirePassRenderPipelineComponent,
    bind_group: &BindGroupComponent,
    vertex_buffer: &VertexBufferComponent,
    index_buffer: &IndexBufferComponent,
    command_buffers: &CommandBuffersComponent,
    texture_view: &IndirectComponent<RenderAttachmentTextureView>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let opaque_pipeline = opaque_pipeline.read();
    let opaque_pipeline = if let LazyComponent::Ready(opaque_pipeline) = &*opaque_pipeline {
        opaque_pipeline
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

    {
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[RenderPassColorAttachment {
                view: texture_view,
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
            depth_stencil_attachment: None,
        });

        let vertex_count = 24;
        let index_count = 36;
        let vertex_offset = vertex_count * std::mem::size_of::<[f32; 3]>() as BufferAddress;

        rpass.push_debug_group("Prepare data for draw.");
        rpass.set_pipeline(opaque_pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint16);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..vertex_offset));
        rpass.set_vertex_buffer(1, vertex_buffer.slice(vertex_offset..));
        rpass.pop_debug_group();
        rpass.insert_debug_marker("Draw!");
        rpass.draw_indexed(0..index_count as u32, 0, 0..1);
        if let LazyComponent::Ready(wire_pipeline) = &*wire_pipeline.read() {
            rpass.set_pipeline(wire_pipeline);
            rpass.draw_indexed(0..index_count as u32, 0, 0..1);
        }
    }

    command_buffers.write().push(encoder.finish());
}
