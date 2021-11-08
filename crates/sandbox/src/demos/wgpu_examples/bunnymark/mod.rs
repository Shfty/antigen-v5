mod components;
mod systems;

use std::num::NonZeroU32;

pub use components::*;
use legion::{world::SubWorld, IntoQuery};
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddIndirectComponent, ImmutableSchedule, RwLock, Serial, Single,
};

use antigen_wgpu::{
    assemble_buffer_data, assemble_texture_data,
    wgpu::{
        BufferAddress, BufferDescriptor, BufferUsages, Device, Extent3d, ImageCopyTextureBase,
        ImageDataLayout, SamplerDescriptor, TextureAspect, TextureDescriptor, TextureDimension,
        TextureFormat, TextureUsages, TextureViewDescriptor,
    },
    BufferComponent, CommandBuffersComponent, RenderAttachment, RenderPipelineComponent,
    SurfaceComponent, Texels, TextureComponent, TextureViewComponent,
};

const MAX_BUNNIES: usize = 1 << 20;
const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: f32 = 750.0;

pub enum Logo {}
pub enum Global {}
pub enum Local {}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    mvp: [[f32; 4]; 4],
    size: [f32; 2],
    pad: [f32; 2],
}

#[repr(C, align(256))]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    _pad: u32,
}

#[legion::system]
#[read_component(Device)]
pub fn assemble(world: &SubWorld, cmd: &mut legion::systems::CommandBuffer) {
    let device = <&Device>::query().iter(world).next().unwrap();

    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    antigen_winit::assemble_window(cmd, &(window_entity,));
    antigen_wgpu::assemble_window_surface(cmd, &(window_entity,));

    // Add title to window
    antigen_winit::assemble_window_title(cmd, &(window_entity,), &"Bunnymark");

    // Renderer
    cmd.add_component(renderer_entity, Bunnymark);
    cmd.add_component(renderer_entity, RenderPipelineComponent::<()>::pending());
    cmd.add_component(renderer_entity, CommandBuffersComponent::new());
    cmd.add_indirect_component::<SurfaceComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<TextureViewComponent<RenderAttachment>>(
        renderer_entity,
        window_entity,
    );

    // Buffer data
    let globals = Globals {
        mvp: *nalgebra::Orthographic3::new(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
            .into_inner()
            .as_ref(),
        size: [BUNNY_SIZE; 2],
        pad: [0.0; 2],
    };

    assemble_buffer_data::<Global, _>(cmd, renderer_entity, RwLock::new(globals), 0);

    // Texture data
    let img_data = include_bytes!("logo.png");
    let decoder = png::Decoder::new(std::io::Cursor::new(img_data));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let size = Extent3d {
        width: info.width,
        height: info.height,
        depth_or_array_layers: 1,
    };

    assemble_texture_data::<Logo, _>(
        cmd,
        renderer_entity,
        Texels::new(buf),
        ImageCopyTextureBase {
            texture: (),
            mip_level: 0,
            origin: Default::default(),
            aspect: TextureAspect::All,
        },
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(size.width).unwrap()),
            rows_per_image: Some(NonZeroU32::new(size.height).unwrap()),
        },
    );

    // Buffers
    cmd.add_component(
        renderer_entity,
        BufferComponent::<Global>::pending(BufferDescriptor {
            label: Some("Global"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            size: std::mem::size_of::<Globals>() as BufferAddress,
            mapped_at_creation: false,
        }),
    );

    let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as BufferAddress;

    cmd.add_component(
        renderer_entity,
        BufferComponent::<Local>::pending(BufferDescriptor {
            label: Some("Local"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            size: (MAX_BUNNIES as BufferAddress) * uniform_alignment,
            mapped_at_creation: false,
        }),
    );

    // Texture
    cmd.add_component(
        renderer_entity,
        TextureComponent::<Logo>::pending(TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        }),
    );

    // Texture view
    cmd.add_component(
        renderer_entity,
        TextureViewComponent::<Logo>::pending(TextureViewDescriptor::default()),
    );

    // Sampler
    cmd.add_component(
        renderer_entity,
        SamplerComponent::<Logo>::pending(SamplerDescriptor::default()),
    );
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_buffers_system::<Global>(),
            antigen_wgpu::create_buffers_system::<Local>(),
            antigen_wgpu::create_textures_system::<Logo>(),
            antigen_wgpu::create_texture_views_system::<Logo>(),
            antigen_wgpu::create_samplers_system::<Logo>(),
        ],
        parallel![
            antigen_wgpu::buffer_write_system::<Global, RwLock<Globals>, Globals> > (),
            antigen_wgpu::texture_write_system::<Logo, Texels<Vec<u8>>, Vec<u8>>(),
        ],
        bunnymark_prepare_system(),
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![bunnymark_render_system()]
}
