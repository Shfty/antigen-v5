use super::{MsaaLine, Vertex, VertexBufferComponent, SAMPLE_COUNT, VERTEX_COUNT};
use antigen_core::{
    ChangedFlag, GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock, Usage,
};

use antigen_wgpu::{
    wgpu::{
        vertex_attr_array, BufferAddress, Color, CommandEncoderDescriptor, Device, Extent3d,
        FragmentState, FrontFace, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor,
        PrimitiveState, PrimitiveTopology, RenderBundleDescriptor, RenderBundleEncoderDescriptor,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        SurfaceConfiguration, VertexBufferLayout, VertexState, VertexStepMode,
    },
    CommandBuffersComponent, MsaaFramebuffer, MsaaFramebufferTextureDescriptor,
    MsaaFramebufferTextureView, PipelineLayoutComponent, RenderAttachmentTextureView,
    RenderBundleComponent, ShaderModuleComponent, SurfaceConfigurationComponent,
    TextureDescriptorComponent, TextureViewDescriptorComponent,
};

use antigen_winit::{
    winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent},
    WindowComponent, WindowEventComponent,
};
use legion::{world::SubWorld, IntoQuery};

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

#[legion::system(par_for_each)]
#[read_component(SurfaceConfigurationComponent)]
#[read_component(ChangedFlag<SurfaceConfigurationComponent>)]
#[read_component(MsaaFramebufferTextureDescriptor<'static>)]
#[read_component(Usage<MsaaFramebuffer, ChangedFlag<TextureDescriptorComponent<'static>>>)]
#[read_component(Usage<MsaaFramebuffer, ChangedFlag<TextureViewDescriptorComponent<'static>>>)]
pub fn msaa_line_resize(
    world: &SubWorld,
    _: &MsaaLine,
    surface_config: &IndirectComponent<SurfaceConfigurationComponent>,
    surface_config_changed: &IndirectComponent<ChangedFlag<SurfaceConfigurationComponent>>,
    msaa_framebuffer_desc: &IndirectComponent<MsaaFramebufferTextureDescriptor<'static>>,
    msaa_framebuffer_desc_changed: &IndirectComponent<
        Usage<MsaaFramebuffer, ChangedFlag<TextureDescriptorComponent<'static>>>,
    >,
    msaa_framebuffer_view_desc_changed: &IndirectComponent<
        Usage<MsaaFramebuffer, ChangedFlag<TextureViewDescriptorComponent<'static>>>,
    >,
) {
    let surface_config_changed = world.get_indirect(surface_config_changed).unwrap();
    let surface_config = world.get_indirect(surface_config).unwrap();
    let msaa_framebuffer_desc = world.get_indirect(msaa_framebuffer_desc).unwrap();
    let msaa_framebuffer_desc_changed = world.get_indirect(msaa_framebuffer_desc_changed).unwrap();
    let msaa_framebuffer_view_desc_changed = world
        .get_indirect(msaa_framebuffer_view_desc_changed)
        .unwrap();

    if !surface_config_changed.get() {
        return;
    }

    println!("Surface config changed, recreating texture and view");

    let surface_config = surface_config.read();
    msaa_framebuffer_desc.write().size = Extent3d {
        width: surface_config.width,
        height: surface_config.height,
        depth_or_array_layers: 1,
    };

    msaa_framebuffer_desc_changed.set(true);
    msaa_framebuffer_view_desc_changed.set(true);
}

#[legion::system(par_for_each)]
#[read_component(WindowComponent)]
#[read_component(WindowEventComponent)]
pub fn msaa_line_key_event(
    world: &SubWorld,
    _: &MsaaLine,
    window: &IndirectComponent<WindowComponent>,
    render_bundle: &RenderBundleComponent,
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

    let window_event = <&WindowEventComponent>::query()
        .iter(world)
        .next()
        .expect("No WindowEventComponent");

    if let (
        Some(window_id),
        Some(WindowEvent::KeyboardInput {
            input:
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(key),
                    ..
                },
            ..
        }),
    ) = &*window_event.read()
    {
        if window.id() != *window_id {
            return;
        }

        match key {
            VirtualKeyCode::Left => {
                println!("Left");
                render_bundle.write().set_pending();
            }
            VirtualKeyCode::Right => {
                println!("Right");
                render_bundle.write().set_pending();
            }
            _ => (),
        }
    }
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
