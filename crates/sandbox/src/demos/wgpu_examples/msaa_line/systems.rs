use super::{MsaaLine, Vertex, VertexBufferComponent, SAMPLE_COUNT, VERTEX_COUNT};
use antigen_core::{GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock};

use antigen_wgpu::{
    wgpu::{
        vertex_attr_array, BufferAddress, Color, CommandEncoderDescriptor, Device, FragmentState,
        FrontFace, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PrimitiveState,
        PrimitiveTopology, RenderBundleDescriptor, RenderBundleEncoderDescriptor,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        SurfaceConfiguration, VertexBufferLayout, VertexState, VertexStepMode,
    },
    CommandBuffersComponent, MsaaFramebufferTextureView, PipelineLayoutComponent,
    RenderAttachmentTextureView, RenderBundleComponent, ShaderModuleComponent,
    SurfaceConfigurationComponent,
};

use legion::IntoQuery;

// Initialize the MSAA lines render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn msaa_line_prepare(
    world: &legion::world::SubWorld,
    _: &MsaaLine,
    shader_module: &ShaderModuleComponent,
    pipeline_layout_component: &PipelineLayoutComponent,
    render_bundle_component: &RenderBundleComponent,
    surface_configuration_component: &IndirectComponent<SurfaceConfigurationComponent>,
    vertex_buffer_component: &VertexBufferComponent,
) {
    let device = <&Device>::query().iter(world).next().unwrap();

    // Create pipeline layout if needed
    if pipeline_layout_component.read().is_pending() {
        let pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        pipeline_layout_component.write().set_ready(pipeline_layout);
    }

    let pipeline_layout_component = pipeline_layout_component.read();
    let pipeline_layout = if let LazyComponent::Ready(pipeline_layout) = &*pipeline_layout_component
    {
        pipeline_layout
    } else {
        unreachable!()
    };

    // Create render bundle
    if !render_bundle_component.read().is_pending() {
        return;
    }

    let shader_module = shader_module.read();
    let shader_module = if let LazyComponent::Ready(shader_module) = &*shader_module {
        shader_module
    } else {
        return;
    };

    let vertex_buffer_component = vertex_buffer_component.read();
    let vertex_buffer = if let LazyComponent::Ready(vertex_buffer) = &*vertex_buffer_component {
        vertex_buffer
    } else {
        return;
    };

    let surface_configuration_component =
        world.get_indirect(surface_configuration_component).unwrap();
    let config = ReadWriteLock::<SurfaceConfiguration>::read(surface_configuration_component);

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        vertex: VertexState {
            module: shader_module,
            entry_point: "vs_main",
            buffers: &[VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as BufferAddress,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2, 1 => Float32x4],
            }],
        },
        fragment: Some(FragmentState {
            module: shader_module,
            entry_point: "fs_main",
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::LineList,
            front_face: FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState {
            count: SAMPLE_COUNT,
            ..Default::default()
        },
    });

    let mut encoder = device.create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
        label: None,
        color_formats: &[config.format],
        depth_stencil: None,
        sample_count: SAMPLE_COUNT,
    });

    encoder.set_pipeline(&pipeline);
    encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
    encoder.draw(0..VERTEX_COUNT, 0..1);

    let render_bundle = encoder.finish(&RenderBundleDescriptor {
        label: Some("main"),
    });

    render_bundle_component.write().set_ready(render_bundle);
}

// Render the MSAA lines pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
#[read_component(MsaaFramebufferTextureView)]
pub fn msaa_line_render(
    world: &legion::world::SubWorld,
    _: &MsaaLine,
    render_bundle: &RenderBundleComponent,
    command_buffers: &CommandBuffersComponent,
    render_attachment: &IndirectComponent<RenderAttachmentTextureView>,
    msaa_framebuffer: &IndirectComponent<MsaaFramebufferTextureView>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    let render_bundle = render_bundle.read();
    let render_bundle = if let LazyComponent::Ready(render_bundle) = &*render_bundle {
        render_bundle
    } else {
        return;
    };

    let render_attachment = world.get_indirect(render_attachment).unwrap();
    let render_attachment = render_attachment.read();
    let render_attachment = if let LazyComponent::Ready(render_attachment) = &*render_attachment {
        render_attachment
    } else {
        return;
    };

    let msaa_framebuffer = world.get_indirect(msaa_framebuffer).unwrap();
    let msaa_framebuffer = msaa_framebuffer.read();
    let msaa_framebuffer = if let LazyComponent::Ready(msaa_framebuffer) = &*msaa_framebuffer {
        msaa_framebuffer
    } else {
        return;
    };

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    let ops = Operations {
        load: LoadOp::Clear(Color::BLACK),
        store: true,
    };
    let rpass_color_attachment = if SAMPLE_COUNT == 1 {
        RenderPassColorAttachment {
            view: render_attachment,
            resolve_target: None,
            ops,
        }
    } else {
        RenderPassColorAttachment {
            view: msaa_framebuffer,
            resolve_target: Some(render_attachment),
            ops,
        }
    };

    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[rpass_color_attachment],
        depth_stencil_attachment: None,
    });
    rpass.execute_bundles(std::iter::once(render_bundle));
    drop(rpass);

    command_buffers.write().push(encoder.finish());
}
