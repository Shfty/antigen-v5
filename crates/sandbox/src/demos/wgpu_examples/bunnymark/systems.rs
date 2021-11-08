use std::borrow::Cow;

use super::{Bunnymark, Globals, Locals, MAX_BUNNIES, BUNNY_SIZE};
use antigen_core::{GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock};

use antigen_wgpu::{CommandBuffersComponent, RenderAttachment, RenderPipelineComponent, SurfaceComponent, TextureViewComponent, wgpu::{AddressMode, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState, BufferAddress, BufferBinding, BufferBindingType, BufferDescriptor, BufferSize, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, Device, DynamicOffset, Extent3d, FilterMode, FragmentState, ImageDataLayout, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, SurfaceConfiguration, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState, util::{BufferInitDescriptor, DeviceExt}}};

use legion::IntoQuery;

// Initialize the hello triangle render pipeline
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(SurfaceComponent)]
pub fn bunnymark_prepare(
    world: &legion::world::SubWorld,
    _: &Bunnymark,
    render_pipeline_component: &RenderPipelineComponent<()>,
    surface_component: &IndirectComponent<SurfaceComponent>,
) {
    if render_pipeline_component.read().is_pending() {
        let device = <&Device>::query().iter(world).next().unwrap();
        let surface_component = world.get_indirect(surface_component).unwrap();
        let config = ReadWriteLock::<SurfaceConfiguration>::read(surface_component);
        let format = config.format;

        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let global_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(std::mem::size_of::<Globals>() as _),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler {
                            filtering: true,
                            comparison: false,
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let local_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(std::mem::size_of::<Locals>() as _),
                },
                count: None,
            }],
            label: None,
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&global_bind_group_layout, &local_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::default(),
                }],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let global_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &global_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });
        let local_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &local_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &local_buffer,
                    offset: 0,
                    size: BufferSize::new(std::mem::size_of::<Locals>() as _),
                }),
            }],
            label: None,
        });
    }
}

// Render the hello triangle pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(TextureViewComponent<'static, RenderAttachment>)]
pub fn bunnymark_render(
    world: &legion::world::SubWorld,
    _: &Bunnymark,
    render_pipeline: &RenderPipelineComponent<()>,
    command_buffers: &CommandBuffersComponent,
    texture_view: &IndirectComponent<TextureViewComponent<'static, RenderAttachment>>,
) {
    let device = if let Some(components) = <&Device>::query().iter(world).next() {
        components
    } else {
        return;
    };

    if let LazyComponent::Ready(render_pipeline) = &*render_pipeline.read() {
        let texture_view = world.get_indirect(texture_view).unwrap();

        if let LazyComponent::Ready(texture_view) = &*texture_view.read() {
            let delta = 0.01;
            for bunny in self.bunnies.iter_mut() {
                bunny.position[0] += bunny.velocity[0] * delta;
                bunny.position[1] += bunny.velocity[1] * delta;
                bunny.velocity[1] += GRAVITY * delta;
                if (bunny.velocity[0] > 0.0
                    && bunny.position[0] + 0.5 * BUNNY_SIZE > self.extent[0] as f32)
                    || (bunny.velocity[0] < 0.0 && bunny.position[0] - 0.5 * BUNNY_SIZE < 0.0)
                {
                    bunny.velocity[0] *= -1.0;
                }
                if bunny.velocity[1] < 0.0 && bunny.position[1] < 0.5 * BUNNY_SIZE {
                    bunny.velocity[1] *= -1.0;
                }
            }

            let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment;
            queue.write_buffer(&self.local_buffer, 0, unsafe {
                std::slice::from_raw_parts(
                    self.bunnies.as_ptr() as *const u8,
                    self.bunnies.len() * uniform_alignment as usize,
                )
            });

            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let clear_color = Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                };
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(clear_color),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });
                rpass.set_pipeline(&self.pipeline);
                rpass.set_bind_group(0, &self.global_group, &[]);
                for i in 0..self.bunnies.len() {
                    let offset =
                        (i as DynamicOffset) * (uniform_alignment as DynamicOffset);
                    rpass.set_bind_group(1, &self.local_group, &[offset]);
                    rpass.draw(0..4, 0..1);
                }
            }

            command_buffers.write().push(encoder.finish());
        }
    }
}
