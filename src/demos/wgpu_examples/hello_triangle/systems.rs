use super::HelloTriangle;
use crate::{
    CommandBuffersComponent, IndirectComponent, LazyComponent, ReadWriteLock,
    RenderPipelineComponent, SurfaceComponent, TextureViewComponent,
};

use legion::{Entity, IntoQuery};

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(wgpu::Device)]
#[read_component(SurfaceComponent)]
pub fn hello_triangle_prepare(
    world: &legion::world::SubWorld,
    entity: &Entity,
    _: &HelloTriangle,
    render_pipeline_component: &RenderPipelineComponent,
    surface_component: &IndirectComponent<SurfaceComponent>,
) {
    if render_pipeline_component.read().is_pending() {
        let device = <&wgpu::Device>::query().iter(world).next().unwrap();
        let surface_component = surface_component.get_sub_world(world, entity).unwrap();
        let format = ReadWriteLock::<wgpu::SurfaceConfiguration>::read(surface_component).format;

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shader.wgsl"
            ))),
        });

        let pipeline_layout = device.create_pipeline_layout(&mut wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[format.into()],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
        });

        render_pipeline_component.write().set_ready(render_pipeline);
    }
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(wgpu::Device)]
#[read_component(TextureViewComponent<'static>)]
pub fn hello_triangle_render(
    world: &legion::world::SubWorld,
    entity: &Entity,
    _: &HelloTriangle,
    render_pipeline: &RenderPipelineComponent,
    command_buffers: &CommandBuffersComponent,
    texture_view: &IndirectComponent<TextureViewComponent<'static>>,
) {
    let device = if let Some(components) = <&wgpu::Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    if let LazyComponent::Ready(render_pipeline) = &*render_pipeline.read() {
        let texture_view = texture_view.get_sub_world(world, entity).unwrap();

        if let LazyComponent::Ready(texture_view) = &*texture_view.read() {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });
                rpass.set_pipeline(render_pipeline);
                rpass.draw(0..3, 0..1);
            }

            command_buffers.write().push(encoder.finish());
        }
    }
}
