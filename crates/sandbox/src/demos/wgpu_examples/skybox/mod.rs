mod components;
mod systems;

use antigen_winit::{AssembleWinit, WindowComponent};
pub use components::*;
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddIndirectComponent, AsUsage, ImmutableSchedule, LazyComponent,
    RwLock, Serial, Single, Construct
};

use antigen_wgpu::{
    wgpu::{
        AddressMode, BufferAddress, BufferDescriptor, BufferSize, BufferUsages, Device, FilterMode,
        SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, TextureFormat,
    },
    AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent, TextureComponent,
    TextureViewComponent, ToBytes,
};

use bytemuck::{Pod, Zeroable};

const IMAGE_SIZE: u32 = 128;
const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth24Plus;

pub struct Camera {
    angle_y: f32,
    angle_xz: f32,
    dist: f32,
}

const MODEL_CENTER_Y: f32 = 2.0;

impl Camera {
    fn to_uniform_data(&self, aspect: f32) -> [f32; 52] {
        let projection = nalgebra_glm::perspective_lh_zo(aspect, 45.0, 1.0, 50.0);
        let cam_pos = nalgebra::vector![
            self.angle_y.cos() * self.angle_xz.sin() * self.dist,
            self.angle_y.sin() * self.dist + MODEL_CENTER_Y,
            self.angle_y.cos() * self.angle_xz.cos() * self.dist
        ];
        let view = nalgebra_glm::look_at_lh(
            &cam_pos,
            &nalgebra::vector![0.0, MODEL_CENTER_Y, 0.0],
            &nalgebra::Vector3::y_axis(),
        );
        let proj_inv = projection.try_inverse().unwrap();

        let mut raw = [0f32; 52];
        raw[..16].copy_from_slice(projection.as_slice());
        raw[16..32].copy_from_slice(proj_inv.as_slice());
        raw[32..48].copy_from_slice(view.as_slice());
        raw[48..51].copy_from_slice(cam_pos.as_slice());
        raw[51] = 1.0;
        raw
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

impl ToBytes for Vertex {
    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Skybox");

    // Renderer
    cmd.add_component(renderer_entity, Skybox);
    cmd.assemble_wgpu_render_pipeline_with_usage::<EntityTag>(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<Sky>(renderer_entity);
    cmd.assemble_wgpu_bind_group(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Window reference for input handling
    cmd.add_indirect_component::<WindowComponent>(renderer_entity, window_entity);

    // Shader
    cmd.assemble_wgpu_shader(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        },
    );

    // Object data
    let source = include_bytes!("models/teslacyberv3.0.obj");
    let data = obj::ObjData::load_buf(&source[..]).unwrap();
    for object in data.objects {
        for group in object.groups {
            let mut vertices = Vec::new();
            for poly in group.polys {
                for end_index in 2..poly.0.len() {
                    for &index in &[0, end_index - 1, end_index] {
                        let obj::IndexTuple(position_id, _texture_id, normal_id) = poly.0[index];
                        vertices.push(Vertex {
                            pos: data.position[position_id],
                            normal: data.normal[normal_id.unwrap()],
                        })
                    }
                }
            }

            let vertex_count = vertices.len();

            let object_entity = cmd.push(());
            cmd.assemble_wgpu_buffer_data_with_usage::<Vertex, _>(
                object_entity,
                VertexDataComponent::construct(vertices),
                0,
            );
            cmd.assemble_wgpu_buffer_with_usage::<Vertex>(
                object_entity,
                BufferDescriptor {
                    label: Some("Vertex"),
                    size: (std::mem::size_of::<Vertex>() * vertex_count) as BufferAddress,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );
            cmd.add_component(object_entity, VertexCount::as_usage(vertex_count));
        }
    }

    // Camera uniform
    let camera = Camera {
        angle_xz: 0.2,
        angle_y: 0.2,
        dist: 30.0,
    };

    let raw_uniforms = camera.to_uniform_data(1.0);
    antigen_wgpu::assemble_staging_belt_data_with_usage::<Uniform, _>(
        cmd,
        renderer_entity,
        UniformDataComponent::construct(raw_uniforms),
        0,
        BufferSize::new(std::mem::size_of::<[f32; 52]>() as u64).unwrap(),
    );
    antigen_wgpu::assemble_staging_belt(cmd, renderer_entity, 0x100);
    cmd.assemble_wgpu_buffer_with_usage::<Uniform>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Buffer"),
            size: std::mem::size_of::<[f32; 52]>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Texture
    cmd.add_component(
        renderer_entity,
        Texture::as_usage(TextureComponent::construct(LazyComponent::Pending)),
    );

    // Texture view
    cmd.add_component(
        renderer_entity,
        Texture::as_usage(TextureViewComponent::construct(LazyComponent::Pending)),
    );

    // Texture sampler
    cmd.assemble_wgpu_sampler(
        renderer_entity,
        SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        },
    );

    // Depth texture view
    cmd.add_component(
        renderer_entity,
        Depth::as_usage(TextureViewComponent::construct(LazyComponent::Pending)),
    );
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_shader_modules_system(),
            antigen_wgpu::create_buffers_system::<Vertex>(),
            antigen_wgpu::create_buffers_system::<Uniform>(),
            antigen_wgpu::create_samplers_system(),
        ],
        parallel![
            antigen_wgpu::buffer_write_system::<Vertex, RwLock<Vec<Vertex>>, Vec<Vertex>>(),
            antigen_wgpu::staging_belt_write_system::<Uniform, RwLock<[f32; 52]>, [f32; 52]>(),
        ],
        skybox_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![skybox_render_system()]
}

pub fn cursor_moved_schedule() -> ImmutableSchedule<Single> {
    single![skybox_cursor_moved_system()]
}
