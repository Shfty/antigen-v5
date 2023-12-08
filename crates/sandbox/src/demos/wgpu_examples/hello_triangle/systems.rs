use super::HelloTriangle;
use antigen_core::{ GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock};

use antigen_wgpu::{
    wgpu::{
        Color, CommandEncoderDescriptor, Device, FragmentState, LoadOp, MultisampleState,
        Operations, PipelineLayoutDescriptor, PrimitiveState, RenderPassColorAttachment,
        RenderPassDescriptor, RenderPipelineDescriptor, VertexState,
    },
    CommandBuffersComponent, RenderAttachmentTextureView, RenderPipelineComponent,
    ShaderModuleComponent, SurfaceConfigurationComponent,
};

use legion::IntoQuery;

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
pub fn hello_triangle_prepare(
    world: &legion::world::SubWorld,
    _: &HelloTriangle,
    shader_module: &ShaderModuleComponent,
    render_pipeline_component: &RenderPipelineComponent,
    surface_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    if !render_pipeline_component.read().is_pending() {
        return;
    }
    let device = <&Device>::query().iter(world).next().unwrap();

    let shader_module = shader_module.read();
    let shader_module = if let LazyComponent::Ready(shader_module) = &*shader_module {
        shader_module
    } else {
        return;
    };

    let surface_component = world.get_indirect(surface_component).unwrap();
    let config = surface_component.read();

    let pipeline_layout = device.create_pipeline_layout(&mut PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
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
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
    });

    render_pipeline_component.write().set_ready(render_pipeline);
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(RenderAttachmentTextureView)]
pub fn hello_triangle_render(
    world: &legion::world::SubWorld,
    _: &HelloTriangle,
    render_pipeline: &RenderPipelineComponent,
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
            view: texture_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::GREEN),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });
    rpass.set_pipeline(render_pipeline);
    rpass.draw(0..3, 0..1);
    drop(rpass);

    command_buffers.write().push(encoder.finish());
}
