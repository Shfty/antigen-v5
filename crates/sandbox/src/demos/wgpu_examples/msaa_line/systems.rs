use super::{MsaaLine, Vertex, VertexBufferComponent, VERTEX_COUNT};
use antigen_core::{
    Changed, ChangedTrait, GetIndirect, IndirectComponent, LazyComponent,
    ReadWriteLock, Usage,
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
    TextureViewDescriptorComponent,
};

use antigen_winit::{
    winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent},
    WindowComponent, WindowEventComponent,
};
use legion::{world::SubWorld, IntoQuery};

// Initialize the MSAA lines render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(Changed<SurfaceConfigurationComponent>)]
#[read_component(MsaaFramebufferTextureDescriptor<'static>)]
pub fn msaa_line_prepare(
    world: &legion::world::SubWorld,
    _: &MsaaLine,
    shader_module: &ShaderModuleComponent,
    pipeline_layout_component: &PipelineLayoutComponent,
    render_bundle_component: &RenderBundleComponent,
    vertex_buffer_component: &VertexBufferComponent,
    surface_configuration_component: &IndirectComponent<Changed<SurfaceConfigurationComponent>>,
    msaa_framebuffer_desc: &IndirectComponent<MsaaFramebufferTextureDescriptor<'static>>,
) {
    println!("MSAA Line Prepare");

    let device = <&Device>::query().iter(world).next().unwrap();

    // Create pipeline layout if needed
    if pipeline_layout_component.read().is_pending() {
        let pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        pipeline_layout_component.write().set_ready(pipeline_layout);

        println!("Created pipeline layout");
    }

    let pipeline_layout_component = pipeline_layout_component.read();
    let pipeline_layout = if let LazyComponent::Ready(pipeline_layout) = &*pipeline_layout_component
    {
        pipeline_layout
    } else {
        unreachable!()
    };

    println!("Pipeline layout ready");

    // Create render bundle
    if !render_bundle_component.read().is_pending() {
        return;
    }

    println!("Render bundle is pending");

    let shader_module = shader_module.read();
    let shader_module = if let LazyComponent::Ready(shader_module) = &*shader_module {
        shader_module
    } else {
        return;
    };

    println!("Shader module ready");

    let vertex_buffer_component = vertex_buffer_component.read();
    let vertex_buffer = if let LazyComponent::Ready(vertex_buffer) = &*vertex_buffer_component {
        vertex_buffer
    } else {
        return;
    };

    println!("Vertex buffer ready");

    let surface_configuration_component =
        world.get_indirect(surface_configuration_component).unwrap();
    let config = ReadWriteLock::<SurfaceConfiguration>::read(surface_configuration_component);

    let msaa_framebuffer_desc = world.get_indirect(msaa_framebuffer_desc).unwrap();
    let msaa_framebuffer_desc = msaa_framebuffer_desc.read();

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
            count: msaa_framebuffer_desc.sample_count,
            ..Default::default()
        },
    });

    let mut encoder = device.create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
        label: None,
        color_formats: &[config.format],
        depth_stencil: None,
        sample_count: msaa_framebuffer_desc.sample_count,
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
#[read_component(Changed<SurfaceConfigurationComponent>)]
#[read_component(MsaaFramebufferTextureDescriptor<'static>)]
#[read_component(Usage<MsaaFramebuffer, TextureViewDescriptorComponent<'static>>)]
pub fn msaa_line_resize(
    world: &SubWorld,
    _: &MsaaLine,
    surface_config: &IndirectComponent<Changed<SurfaceConfigurationComponent>>,
    msaa_framebuffer_desc: &IndirectComponent<MsaaFramebufferTextureDescriptor<'static>>,
    msaa_framebuffer_view_desc: &IndirectComponent<
        Usage<MsaaFramebuffer, TextureViewDescriptorComponent<'static>>,
    >,
) {
    let surface_config = world.get_indirect(surface_config).unwrap();
    let msaa_framebuffer_desc = world.get_indirect(msaa_framebuffer_desc).unwrap();
    let msaa_framebuffer_view_desc = world.get_indirect(msaa_framebuffer_view_desc).unwrap();

    if !surface_config.get_changed() {
        return;
    }

    println!("Surface config changed, recreating texture and view");

    let surface_config = surface_config.read();
    msaa_framebuffer_desc.write().size = Extent3d {
        width: surface_config.width,
        height: surface_config.height,
        depth_or_array_layers: 1,
    };

    msaa_framebuffer_desc.set_changed(true);
    msaa_framebuffer_view_desc.set_changed(true);
}

#[legion::system(par_for_each)]
#[read_component(WindowComponent)]
#[read_component(WindowEventComponent)]
#[read_component(MsaaFramebufferTextureDescriptor<'static>)]
pub fn msaa_line_key_event(
    world: &SubWorld,
    _: &MsaaLine,
    window: &IndirectComponent<WindowComponent>,
    render_bundle: &RenderBundleComponent,
    msaa_framebuffer_desc: &IndirectComponent<MsaaFramebufferTextureDescriptor<'static>>,
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

    let window_event = window_event.read();
    let (window_id, key) = if let (
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
    ) = &*window_event
    {
        (window_id, key)
    } else {
        return;
    };

    if window.id() != *window_id {
        return;
    }

    let msaa_framebuffer_desc = world.get_indirect(msaa_framebuffer_desc).unwrap();

    match key {
        VirtualKeyCode::Left => {
            msaa_framebuffer_desc.write().sample_count = 1;
            render_bundle.write().set_pending();
            window.request_redraw();
        }
        VirtualKeyCode::Right => {
            msaa_framebuffer_desc.write().sample_count = 4;
            render_bundle.write().set_pending();
            window.request_redraw();
        }
        _ => (),
    }
}

// Render the MSAA lines pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
#[read_component(MsaaFramebufferTextureDescriptor<'static>)]
#[read_component(MsaaFramebufferTextureView)]
pub fn msaa_line_render(
    world: &legion::world::SubWorld,
    _: &MsaaLine,
    render_bundle: &RenderBundleComponent,
    command_buffers: &CommandBuffersComponent,
    render_attachment: &IndirectComponent<RenderAttachmentTextureView>,
    msaa_framebuffer_desc: &IndirectComponent<MsaaFramebufferTextureDescriptor<'static>>,
    msaa_framebuffer_view: &IndirectComponent<MsaaFramebufferTextureView>,
) {
    println!("MSAA Line Render");

    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    println!("Device ready");

    let render_bundle = render_bundle.read();
    let render_bundle = if let LazyComponent::Ready(render_bundle) = &*render_bundle {
        render_bundle
    } else {
        return;
    };

    println!("Render bundle ready");

    let render_attachment = world.get_indirect(render_attachment).unwrap();
    let render_attachment = render_attachment.read();
    let render_attachment = if let LazyComponent::Ready(render_attachment) = &*render_attachment {
        render_attachment
    } else {
        return;
    };

    println!("Render attachment ready");

    let msaa_framebuffer_desc = world.get_indirect(msaa_framebuffer_desc).unwrap();
    let msaa_framebuffer_desc = msaa_framebuffer_desc.read();

    let msaa_framebuffer_view = world.get_indirect(msaa_framebuffer_view).unwrap();
    let msaa_framebuffer_view = msaa_framebuffer_view.read();
    let msaa_framebuffer_view =
        if let LazyComponent::Ready(msaa_framebuffer_view) = &*msaa_framebuffer_view {
            msaa_framebuffer_view
        } else {
            return;
        };

    println!("MSAA framebuffer view ready");

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    let ops = Operations {
        load: LoadOp::Clear(Color::BLACK),
        store: true,
    };
    let rpass_color_attachment = if msaa_framebuffer_desc.sample_count == 1 {
        RenderPassColorAttachment {
            view: render_attachment,
            resolve_target: None,
            ops,
        }
    } else {
        RenderPassColorAttachment {
            view: msaa_framebuffer_view,
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
