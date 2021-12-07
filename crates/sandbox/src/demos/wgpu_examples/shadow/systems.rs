use std::ops::Range;

use crate::wgpu_examples::shadow::{
    BufferQuery, CubeMesh, GlobalUniforms, LightQuery, LightRaw, Mesh, ObjectQuery, ObjectUniforms,
    PlaneMesh, DEPTH_FORMAT, MAX_LIGHTS, SHADOW_FORMAT,
};

use super::{
    ForwardBindGroup, ForwardDepthView, ForwardRenderPipeline, ForwardUniformBuffer,
    IndexBufferComponent, IndexCountComponent, LightFovComponent, LightStorageBuffer,
    LightsAreDirtyComponent, ObjectBindGroup, ObjectMatrixComponent, ObjectUniformBuffer,
    RotationSpeed, Shadow, ShadowBindGroup, ShadowPass, ShadowRenderPipeline,
    ShadowSamplerComponent, ShadowTextureViewComponent, ShadowUniformBuffer, UniformOffset,
    VertexBufferComponent,
};
use antigen_core::{
    lazy_read_ready_else_return, ChangedFlag, GetIndirect, IndirectComponent, LazyComponent,
    ReadWriteLock, Usage,
};

use antigen_wgpu::{
    wgpu::{
        util::{BufferInitDescriptor, DeviceExt},
        vertex_attr_array, Adapter, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, BufferAddress, BufferBinding,
        BufferBindingType, BufferDescriptor, BufferSize, BufferUsages, Color,
        CommandEncoderDescriptor, CompareFunction, DepthBiasState, DepthStencilState, Device,
        DownlevelFlags, Extent3d, Face, FragmentState, FrontFace, IndexFormat, LoadOp,
        MultisampleState, Operations, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
        Queue, RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
        RenderPipelineDescriptor, ShaderStages, StencilState, SurfaceConfiguration,
        TextureDescriptor, TextureDimension, TextureSampleType, TextureUsages, TextureView,
        TextureViewDescriptor, TextureViewDimension, VertexBufferLayout, VertexState,
        VertexStepMode,
    },
    CommandBuffersComponent, RenderAttachmentTextureView, ShaderModuleComponent,
    SurfaceConfigurationComponent, TextureViewComponent,
};

use legion::{world::SubWorld, IntoQuery};

fn generate_matrix(aspect_ratio: f32) -> nalgebra::Matrix4<f32> {
    let projection = nalgebra_glm::perspective_rh_zo(aspect_ratio, 45.0, 1.0, 20.0);
    let view = nalgebra_glm::look_at_rh(
        &nalgebra::vector![3.0, -10.0, 6.0],
        &nalgebra::vector![0f32, 0.0, 0.0],
        &nalgebra::Vector3::z_axis(),
    );
    projection * view
}

fn create_depth_texture(config: &SurfaceConfiguration, device: &Device) -> Option<TextureView> {
    if config.width == 0 || config.height == 0 {
        return None;
    }
    let depth_texture = device.create_texture(&TextureDescriptor {
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        label: None,
    });

    Some(depth_texture.create_view(&TextureViewDescriptor::default()))
}

// Initialize the shadow render pipeline
#[legion::system(par_for_each)]
#[read_component(Adapter)]
#[read_component(Device)]
#[read_component(SurfaceConfigurationComponent)]
// Object query
#[read_component(Mesh)]
#[read_component(ObjectMatrixComponent)]
#[read_component(Usage<RotationSpeed, f32>)]
#[read_component(Color)]
#[read_component(Usage<UniformOffset, u32>)]
// Light query
#[read_component(nalgebra::Vector3<f32>)]
#[read_component(Color)]
#[read_component(LightFovComponent)]
#[read_component(Range<f32>)]
#[read_component(ShadowTextureViewComponent)]
pub fn shadow_prepare(
    world: &legion::world::SubWorld,
    _: &Shadow,
    shader_module: &ShaderModuleComponent,
    forward_render_pipeline_component: &ForwardRenderPipeline,
    forward_bind_group_component: &ForwardBindGroup,
    forward_uniform_buf_component: &ForwardUniformBuffer,
    forward_depth_view_component: &ForwardDepthView,
    shadow_render_pipeline_component: &ShadowRenderPipeline,
    shadow_bind_group_component: &ShadowBindGroup,
    shadow_uniform_buf_component: &ShadowUniformBuffer,
    shadow_view: &ShadowTextureViewComponent,
    shadow_sampler: &ShadowSamplerComponent,
    object_bind_group_component: &ObjectBindGroup,
    object_uniform_buf_component: &ObjectUniformBuffer,
    light_storage_buf_component: &LightStorageBuffer,
    surface_component: &IndirectComponent<SurfaceConfigurationComponent>,
) {
    if !forward_render_pipeline_component.read().is_pending() {
        return;
    }

    lazy_read_ready_else_return!(shader_module);
    lazy_read_ready_else_return!(shadow_view);
    lazy_read_ready_else_return!(shadow_sampler);

    let surface_component = world.get_indirect(surface_component).unwrap();
    let config = surface_component.read();

    let adapter = <&Adapter>::query().iter(world).next().unwrap();
    let device = <&Device>::query().iter(world).next().unwrap();

    let supports_storage_resources = adapter
        .get_downlevel_properties()
        .flags
        .contains(DownlevelFlags::VERTEX_STORAGE)
        && device.limits().max_storage_buffers_per_shader_stage > 0;

    let object_uniform_size = std::mem::size_of::<ObjectUniforms>() as BufferAddress;
    let num_objects = ObjectQuery::query().iter(world).count() as BufferAddress;
    assert!(object_uniform_size <= 256);
    // Note: dynamic uniform offsets also have to be aligned to `Limits::min_uniform_buffer_offset_alignment`.
    let object_uniform_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: num_objects * 256,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let local_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: BufferSize::new(object_uniform_size),
            },
            count: None,
        }],
        label: None,
    });

    let object_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &local_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &object_uniform_buf,
                offset: 0,
                size: BufferSize::new(object_uniform_size),
            }),
        }],
        label: None,
    });
    object_bind_group_component
        .write()
        .set_ready(object_bind_group);
    object_uniform_buf_component
        .write()
        .set_ready(object_uniform_buf);

    let light_uniform_size = (MAX_LIGHTS * std::mem::size_of::<LightRaw>()) as BufferAddress;
    let light_storage_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: light_uniform_size,
        usage: if supports_storage_resources {
            BufferUsages::STORAGE
        } else {
            BufferUsages::UNIFORM
        } | BufferUsages::COPY_SRC
            | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let vertex_attr = vertex_attr_array![0 => Sint8x4, 1 => Sint8x4];
    let vb_desc = VertexBufferLayout {
        array_stride: std::mem::size_of::<super::Vertex>() as BufferAddress,
        step_mode: VertexStepMode::Vertex,
        attributes: &vertex_attr,
    };

    // Shadow pass
    let uniform_size = std::mem::size_of::<GlobalUniforms>() as BufferAddress;
    // Create pipeline layout
    let shadow_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0, // global
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(uniform_size),
            },
            count: None,
        }],
    });

    let shadow_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("shadow"),
        bind_group_layouts: &[&shadow_bind_group_layout, &local_bind_group_layout],
        push_constant_ranges: &[],
    });

    let shadow_uniform_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: uniform_size,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create bind group
    let shadow_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &shadow_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: shadow_uniform_buf.as_entire_binding(),
        }],
        label: None,
    });

    // Create the render pipeline
    let shadow_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("shadow"),
        layout: Some(&shadow_pipeline_layout),
        vertex: VertexState {
            module: &shader_module,
            entry_point: "vs_bake",
            buffers: &[vb_desc.clone()],
        },
        fragment: None,
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: SHADOW_FORMAT,
            depth_write_enabled: true,
            depth_compare: CompareFunction::LessEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState {
                constant: 2, // corresponds to bilinear filtering
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: MultisampleState::default(),
    });

    shadow_render_pipeline_component
        .write()
        .set_ready(shadow_pipeline);
    shadow_bind_group_component
        .write()
        .set_ready(shadow_bind_group);
    shadow_uniform_buf_component
        .write()
        .set_ready(shadow_uniform_buf);

    // Forward pass
    // Create pipeline layout
    let forward_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0, // global
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<GlobalUniforms>() as _),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1, // lights
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: if supports_storage_resources {
                        BufferBindingType::Storage { read_only: true }
                    } else {
                        BufferBindingType::Uniform
                    },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(light_uniform_size),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2Array,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler {
                    comparison: true,
                    filtering: true,
                },
                count: None,
            },
        ],
        label: None,
    });

    let forward_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("main"),
        bind_group_layouts: &[&forward_bind_group_layout, &local_bind_group_layout],
        push_constant_ranges: &[],
    });

    let mx_total = generate_matrix(config.width as f32 / config.height as f32);
    let light_count = LightQuery::query().iter(world).count() as u32;
    let forward_uniforms = GlobalUniforms {
        proj: *mx_total.as_ref(),
        num_lights: [light_count, 0, 0, 0],
    };

    let forward_uniform_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(&forward_uniforms),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    // Create bind group
    let forward_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &forward_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: forward_uniform_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: light_storage_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&shadow_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Sampler(&shadow_sampler),
            },
        ],
        label: None,
    });

    light_storage_buf_component
        .write()
        .set_ready(light_storage_buf);

    // Create the render pipeline
    let forward_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("main"),
        layout: Some(&forward_pipeline_layout),
        vertex: VertexState {
            module: &shader_module,
            entry_point: "vs_main",
            buffers: &[vb_desc],
        },
        fragment: Some(FragmentState {
            module: &shader_module,
            entry_point: if supports_storage_resources {
                "fs_main"
            } else {
                "fs_main_without_storage"
            },
            targets: &[config.format.into()],
        }),
        primitive: PrimitiveState {
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
    });

    forward_render_pipeline_component
        .write()
        .set_ready(forward_pipeline);
    forward_bind_group_component
        .write()
        .set_ready(forward_bind_group);
    forward_uniform_buf_component
        .write()
        .set_ready(forward_uniform_buf);

    let forward_depth = create_depth_texture(&config, device).unwrap();
    forward_depth_view_component
        .write()
        .set_ready(forward_depth);
}

#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(Queue)]
#[read_component(SurfaceConfigurationComponent)]
#[read_component(ChangedFlag<SurfaceConfigurationComponent>)]
// Light query
#[read_component(nalgebra::Vector3<f32>)]
#[read_component(Color)]
#[read_component(LightFovComponent)]
#[read_component(Range<f32>)]
#[read_component(ShadowTextureViewComponent)]
pub fn shadow_resize(
    world: &SubWorld,
    _: &Shadow,
    surface_config: &IndirectComponent<SurfaceConfigurationComponent>,
    surface_config_changed: &IndirectComponent<ChangedFlag<SurfaceConfigurationComponent>>,
    forward_depth_view_component: &ForwardDepthView,
    forward_uniform_buf: &ForwardUniformBuffer,
) {
    let device = <&Device>::query().iter(world).next().unwrap();
    let queue = <&Queue>::query().iter(world).next().unwrap();

    lazy_read_ready_else_return!(forward_uniform_buf);

    let surface_config_changed = world.get_indirect(surface_config_changed).unwrap();
    let surface_config = world.get_indirect(surface_config).unwrap();

    if surface_config_changed.get() {
        let surface_config = surface_config.read();
        if let Some(depth_view) = create_depth_texture(&*surface_config, device) {
            forward_depth_view_component.write().set_ready(depth_view);
        }

        let mx_total = generate_matrix(surface_config.width as f32 / surface_config.height as f32);
        let light_count = LightQuery::query().iter(world).count() as u32;
        let forward_uniforms = GlobalUniforms {
            proj: *mx_total.as_ref(),
            num_lights: [light_count, 0, 0, 0],
        };

        queue.write_buffer(
            &*forward_uniform_buf,
            0,
            bytemuck::bytes_of(&forward_uniforms),
        );
    }
}

/// Render the shadow pipeline to the specified entity's surface
#[legion::system(par_for_each)]
#[read_component(Device)]
#[read_component(Queue)]
#[read_component(RenderAttachmentTextureView)]
// Mesh query
#[read_component(PlaneMesh)]
#[read_component(CubeMesh)]
#[read_component(VertexBufferComponent)]
#[read_component(IndexBufferComponent)]
#[read_component(IndexFormat)]
#[read_component(IndexCountComponent)]
// Object query
#[read_component(Mesh)]
#[read_component(ObjectMatrixComponent)]
#[read_component(Usage<RotationSpeed, f32>)]
#[read_component(Color)]
#[read_component(Usage<UniformOffset, u32>)]
// Light query
#[read_component(nalgebra::Vector3<f32>)]
#[read_component(Color)]
#[read_component(LightFovComponent)]
#[read_component(Range<f32>)]
#[read_component(ShadowTextureViewComponent)]
pub fn shadow_render(
    world: &legion::world::SubWorld,
    _: &Shadow,
    command_buffers: &CommandBuffersComponent,
    forward_render_pipeline: &ForwardRenderPipeline,
    forward_bind_group: &ForwardBindGroup,
    forward_depth_view: &ForwardDepthView,
    shadow_render_pipeline: &ShadowRenderPipeline,
    shadow_bind_group: &ShadowBindGroup,
    shadow_uniform_buf: &ShadowUniformBuffer,
    object_bind_group: &ObjectBindGroup,
    object_uniform_buf: &ObjectUniformBuffer,
    light_storage_buf: &LightStorageBuffer,
    lights_are_dirty: &LightsAreDirtyComponent,
    texture_view: &IndirectComponent<RenderAttachmentTextureView>,
) {
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

    let (_, plane_vertex_buffer, plane_index_buffer, plane_index_format, plane_index_count) =
        if let Some(components) = BufferQuery::<PlaneMesh>::query().iter(world).next() {
            components
        } else {
            return;
        };

    lazy_read_ready_else_return!(plane_vertex_buffer);
    lazy_read_ready_else_return!(plane_index_buffer);

    let (_, cube_vertex_buffer, cube_index_buffer, cube_index_format, cube_index_count) =
        if let Some(components) = BufferQuery::<CubeMesh>::query().iter(world).next() {
            components
        } else {
            return;
        };

    lazy_read_ready_else_return!(cube_vertex_buffer);
    lazy_read_ready_else_return!(cube_index_buffer);

    lazy_read_ready_else_return!(forward_render_pipeline);
    lazy_read_ready_else_return!(forward_bind_group);
    lazy_read_ready_else_return!(forward_depth_view);
    lazy_read_ready_else_return!(shadow_render_pipeline);
    lazy_read_ready_else_return!(shadow_bind_group);
    lazy_read_ready_else_return!(shadow_uniform_buf);
    lazy_read_ready_else_return!(object_uniform_buf);
    lazy_read_ready_else_return!(object_bind_group);
    lazy_read_ready_else_return!(light_storage_buf);

    let texture_view = world.get_indirect(texture_view).unwrap();
    lazy_read_ready_else_return!(texture_view);

    let objects = ObjectQuery::query().iter(world).collect::<Vec<_>>();
    let lights = LightQuery::query().iter(world).collect::<Vec<_>>();

    // update uniforms
    for (_, mx_world, rotation_speed, color, uniform_offset) in &objects {
        let rotation_speed = ***rotation_speed;
        let mut mx_world = mx_world.write();

        if rotation_speed != 0.0 {
            let rotation = nalgebra::Matrix4::new_rotation(nalgebra::vector![
                rotation_speed.to_radians(),
                0.0,
                0.0
            ]);
            *mx_world = *mx_world * rotation;
        }
        let data = ObjectUniforms {
            model: *mx_world.as_ref(),
            color: [
                color.r as f32,
                color.g as f32,
                color.b as f32,
                color.a as f32,
            ],
        };
        queue.write_buffer(
            &object_uniform_buf,
            (***uniform_offset) as u64,
            bytemuck::bytes_of(&data),
        );
    }

    if *lights_are_dirty.read() {
        *lights_are_dirty.write() = false;
        for (i, (position, color, fov, range, _)) in lights.iter().enumerate() {
            let mx_view = nalgebra::Matrix4::look_at_rh(
                &nalgebra::point![position.x, position.y, position.z],
                &nalgebra::point![0.0, 0.0, 0.0],
                &nalgebra::Vector3::z_axis(),
            );

            let projection =
                nalgebra_glm::perspective_rh_zo(1.0, (**fov).to_radians(), range.start, range.end);

            let mx_view_proj = projection * mx_view;

            queue.write_buffer(
                &light_storage_buf,
                (i * std::mem::size_of::<LightRaw>()) as BufferAddress,
                bytemuck::bytes_of(&LightRaw {
                    proj: *mx_view_proj.as_ref(),
                    pos: [position.x, position.y, position.z, 1.0],
                    color: [color.r as f32, color.g as f32, color.b as f32, 1.0],
                }),
            );
        }
    }

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    encoder.push_debug_group("shadow passes");
    for (i, (pos, _, _, _, target_view)) in lights.iter().enumerate() {
        let target_view = target_view.read();
        let target_view = if let LazyComponent::Ready(target_view) = &*target_view {
            target_view
        } else {
            continue;
        };

        encoder.push_debug_group(&format!("shadow pass {} (light at position {:?})", i, pos));

        // The light uniform buffer already has the projection,
        // let's just copy it over to the shadow uniform buffer.
        encoder.copy_buffer_to_buffer(
            &light_storage_buf,
            (i * std::mem::size_of::<LightRaw>()) as BufferAddress,
            &shadow_uniform_buf,
            0,
            64,
        );

        encoder.insert_debug_marker("render objects");
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: target_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            pass.set_pipeline(&shadow_render_pipeline);
            pass.set_bind_group(0, &shadow_bind_group, &[]);

            for (mesh, _, _, _, uniform_offset) in &objects {
                let vertex_buffer = match mesh {
                    Mesh::Plane => plane_vertex_buffer,
                    Mesh::Cube => cube_vertex_buffer,
                };

                let index_buffer = match mesh {
                    Mesh::Plane => plane_index_buffer,
                    Mesh::Cube => cube_index_buffer,
                };

                let index_format = match mesh {
                    Mesh::Plane => plane_index_format,
                    Mesh::Cube => cube_index_format,
                };

                let index_count = match mesh {
                    Mesh::Plane => plane_index_count,
                    Mesh::Cube => cube_index_count,
                };

                pass.set_bind_group(1, &object_bind_group, &[***uniform_offset]);
                pass.set_index_buffer(index_buffer.slice(..), *index_format);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw_indexed(0..**index_count as u32, 0, 0..1);
            }
        }

        encoder.pop_debug_group();
    }
    encoder.pop_debug_group();

    // forward pass
    encoder.push_debug_group("forward rendering pass");
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[RenderPassColorAttachment {
                view: &texture_view,
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
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &forward_depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: false,
                }),
                stencil_ops: None,
            }),
        });
        pass.set_pipeline(&forward_render_pipeline);
        pass.set_bind_group(0, &forward_bind_group, &[]);

        for (mesh, _, _, _, uniform_offset) in &objects {
            let vertex_buffer = match mesh {
                Mesh::Plane => plane_vertex_buffer,
                Mesh::Cube => cube_vertex_buffer,
            };

            let index_buffer = match mesh {
                Mesh::Plane => plane_index_buffer,
                Mesh::Cube => cube_index_buffer,
            };

            let index_format = match mesh {
                Mesh::Plane => plane_index_format,
                Mesh::Cube => cube_index_format,
            };

            let index_count = match mesh {
                Mesh::Plane => plane_index_count,
                Mesh::Cube => cube_index_count,
            };

            pass.set_bind_group(1, &object_bind_group, &[***uniform_offset]);
            pass.set_index_buffer(index_buffer.slice(..), *index_format);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.draw_indexed(0..**index_count as u32, 0, 0..1);
        }
    }
    encoder.pop_debug_group();

    command_buffers.write().push(encoder.finish());
}
