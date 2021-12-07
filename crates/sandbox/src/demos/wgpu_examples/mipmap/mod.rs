mod components;
mod systems;

use std::num::NonZeroU32;

use antigen_winit::AssembleWinit;
pub use components::*;
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddIndirectComponent, AsUsage, ImmutableSchedule, RwLock, Serial,
    Single,
};

use antigen_wgpu::{
    wgpu::{
        AddressMode, BufferAddress, BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode,
        ImageCopyTextureBase, ImageDataLayout, Origin3d, SamplerDescriptor, ShaderModuleDescriptor,
        ShaderSource, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages,
    },
    AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent,
};

const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
const MIP_LEVEL_COUNT: u32 = 9;
const MIP_PASS_COUNT: u32 = MIP_LEVEL_COUNT - 1;

fn generate_matrix(aspect_ratio: f32) -> nalgebra::Matrix4<f32> {
    let projection = nalgebra_glm::perspective_lh_zo(aspect_ratio, 45.0, 1.0, 1000.0);

    let view = nalgebra_glm::look_at_lh(
        &nalgebra::vector![0.0f32, 10.0, 0.0],
        &nalgebra::vector![0f32, 0.0, 50.0],
        &nalgebra::Vector3::y_axis(),
    );

    projection * view
}

fn create_texels(size: usize, cx: f32, cy: f32) -> Vec<u8> {
    (0..size * size)
        .flat_map(|id| {
            // get high five for recognizing this ;)
            let mut x = 4.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let mut y = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let mut count = 0;
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }

            use std::iter::once;
            once(0xFF - (count * 2) as u8)
                .chain(once(0xFF - (count * 5) as u8))
                .chain(once(0xFF - (count * 13) as u8))
                .chain(once(std::u8::MAX))
        })
        .collect()
}

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Mipmap");

    // Renderer
    let renderer_entity = cmd.push(());

    cmd.add_component(renderer_entity, Mipmap);

    // Renderer resources
    cmd.assemble_wgpu_bind_group(renderer_entity);
    cmd.assemble_wgpu_render_pipeline_with_usage::<Draw>(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);

    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Draw shader
    cmd.assemble_wgpu_shader_with_usage::<Draw>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("draw.wgsl"))),
        },
    );

    // Matrix uniform
    let matrix = generate_matrix(1.0);
    let mut buf: [f32; 16] = [0.0; 16];
    buf.copy_from_slice(matrix.as_slice());
    cmd.assemble_wgpu_buffer_data_with_usage::<Uniform, _>(
        renderer_entity,
        ViewProjection::as_usage(RwLock::new(buf)),
        0,
    );

    // Uniform buffer
    cmd.assemble_wgpu_buffer_with_usage::<Uniform>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<nalgebra::Matrix4<f32>>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    // Julia set texture
    let size = 1 << MIP_LEVEL_COUNT;
    let texture_extent = Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
    };

    let texels = create_texels(size as usize, -0.8, 0.156);

    cmd.assemble_wgpu_texture_data_with_usage::<JuliaSet, _>(
        renderer_entity,
        RwLock::new(texels),
        ImageCopyTextureBase {
            texture: (),
            mip_level: 0,
            origin: Origin3d::default(),
            aspect: TextureAspect::All,
        },
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(4 * size).unwrap()),
            rows_per_image: None,
        },
    );

    cmd.assemble_wgpu_texture_with_usage::<JuliaSet>(
        renderer_entity,
        TextureDescriptor {
            size: texture_extent,
            mip_level_count: MIP_LEVEL_COUNT,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_DST,
            label: None,
        },
    );

    cmd.assemble_wgpu_texture_view_with_usage::<JuliaSet>(
        renderer_entity,
        renderer_entity,
        Default::default(),
    );

    // Texture sampler
    cmd.assemble_wgpu_sampler_with_usage::<JuliaSet>(
        renderer_entity,
        SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        },
    );
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_shader_modules_usage_system::<Draw>(),
            antigen_wgpu::create_buffers_system::<Uniform>(),
            antigen_wgpu::create_textures_system::<JuliaSet>(),
            antigen_wgpu::create_texture_views_system::<JuliaSet>(),
            antigen_wgpu::create_samplers_with_usage_system::<JuliaSet>(),
        ],
        parallel![
            antigen_wgpu::buffer_write_system::<Uniform, ViewProjectionMatrix, [f32; 16]>(),
            antigen_wgpu::texture_write_system::<JuliaSet, RwLock<Vec<u8>>, Vec<u8>>(),
        ],
        mipmap_prepare_system()
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![mipmap_render_system()]
}
