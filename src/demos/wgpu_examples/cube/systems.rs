use std::borrow::Cow;

use super::{Cube, Index, Opaque, Uniform, Vertex, Wire};
use crate::{
    BindGroupComponent, BufferComponent, CommandBuffersComponent, IndirectComponent, LazyComponent,
    ReadWriteLock, RenderPipelineComponent, SurfaceComponent, TextureViewComponent,
};

use legion::{Entity, IntoQuery, world::SubWorld};
use wgpu::SurfaceConfiguration;

#[rustfmt::skip]
#[allow(unused)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(wgpu::Device)]
#[read_component(SurfaceComponent)]
pub fn cube_prepare(
    world: &SubWorld,
    entity: &Entity,
    _: &Cube,
    opaque_pipeline_component: &RenderPipelineComponent<Opaque>,
    wire_pipeline_component: &RenderPipelineComponent<Wire>,
    texture_view: &TextureViewComponent<'static>,
    uniform_buffer: &BufferComponent<Uniform>,
    bind_group_component: &BindGroupComponent,
    surface_component: &IndirectComponent<SurfaceComponent>,
) {
    if !opaque_pipeline_component.read().is_pending() {
        return;
    }

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

    let surface_component = surface_component.get_sub_world(world, entity).unwrap();
    let config = ReadWriteLock::<SurfaceConfiguration>::read(surface_component);

    let device = <&wgpu::Device>::query().iter(world).next().unwrap();

    // Create pipeline layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
        ],
        label: None,
    });
    bind_group_component.write().set_ready(bind_group);

    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let vertex_buffers = [
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 1,
            }],
        },
    ];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &vertex_buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[config.format.into()],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
    });

    if device
        .features()
        .contains(wgpu::Features::POLYGON_MODE_LINE)
    {
        let pipeline_wire = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_wire",
                targets: &[wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            operation: wgpu::BlendOperation::Add,
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Line,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
        });
        wire_pipeline_component.write().set_ready(pipeline_wire);
    };

    opaque_pipeline_component.write().set_ready(pipeline);
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(wgpu::Device)]
#[read_component(TextureViewComponent<'static>)]
pub fn cube_render(
    world: &SubWorld,
    entity: &Entity,
    _: &Cube,
    opaque_pipeline: &RenderPipelineComponent<Opaque>,
    wire_pipeline: &RenderPipelineComponent<Wire>,
    bind_group: &BindGroupComponent,
    vertex_buffer: &BufferComponent<Vertex>,
    index_buffer: &BufferComponent<Index>,
    command_buffers: &CommandBuffersComponent,
    texture_view: &IndirectComponent<TextureViewComponent<'static>>,
) {
    let device = if let Some(components) = <&wgpu::Device>::query().iter(world).next() {
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

    let texture_view = texture_view.get_sub_world(world, entity).unwrap();
    let texture_view = texture_view.read();
    let texture_view = if let LazyComponent::Ready(texture_view) = &*texture_view {
        texture_view
    } else {
        return;
    };

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
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
        let vertex_offset = vertex_count * std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress;

        rpass.push_debug_group("Prepare data for draw.");
        rpass.set_pipeline(opaque_pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
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
