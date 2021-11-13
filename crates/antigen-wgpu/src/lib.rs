mod components;
mod systems;
mod to_bytes;

use std::path::Path;

pub use components::*;
pub use systems::*;
pub use to_bytes::*;

use legion::{storage::Component, systems::CommandBuffer, world::SubWorld, Entity, World};
use wgpu::{
    Adapter, Backends, BufferAddress, BufferDescriptor, Device, DeviceDescriptor,
    ImageCopyTextureBase, ImageDataLayout, Instance, Queue, ShaderModuleDescriptor, Surface,
};

pub use wgpu;

use antigen_core::{
    serial, single, AddComponentWithChangedFlag, AddIndirectComponent, ChangedFlag,
    ImmutableSchedule, ReadWriteLock, Serial, Single, Usage,
};
use antigen_winit::{WindowEntityMap, WindowEventComponent};

/// Create an entity to hold an Instance, Adapter, Device and Queue
pub fn assemble_wgpu_entity(
    world: &mut World,
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
) -> Entity {
    world.push((instance, adapter, device, queue))
}

pub fn assemble_wgpu_entity_from_env(
    world: &mut World,
    device_desc: &DeviceDescriptor,
    compatible_surface: Option<&Surface>,
    trace_path: Option<&Path>,
) {
    let backend_bits = wgpu::util::backend_bits_from_env().unwrap_or(Backends::PRIMARY);

    let instance = Instance::new(backend_bits);
    println!("Created WGPU instance: {:#?}\n", instance);

    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance,
        backend_bits,
        compatible_surface,
    ))
    .expect("Failed to acquire WGPU adapter");

    let adapter_info = adapter.get_info();
    println!("Acquired WGPU adapter: {:#?}\n", adapter_info);

    let (device, queue) =
        pollster::block_on(adapter.request_device(device_desc, trace_path)).unwrap();

    println!("Acquired WGPU device: {:#?}\n", device);
    println!("Acquired WGPU queue: {:#?}\n", queue);

    assemble_wgpu_entity(world, instance, adapter, device, queue);
}

/// Extends an existing window entity with the means to render to a WGPU surface
#[legion::system]
pub fn assemble_window_surface(cmd: &mut CommandBuffer, #[state] (entity,): &(Entity,)) {
    cmd.add_component(
        *entity,
        SurfaceComponent::pending(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: 100,
            height: 100,
            present_mode: wgpu::PresentMode::Mailbox,
        }),
    );
    cmd.add_component(*entity, SurfaceTextureComponent::pending());
    cmd.add_component(*entity, ChangedFlag::<SurfaceTextureComponent>::new_clean());
    cmd.add_component(
        *entity,
        Usage::<RenderAttachment, _>::new(TextureViewDescriptorComponent::new(Default::default())),
    );
    cmd.add_component(
        *entity,
        Usage::<RenderAttachment, _>::new(TextureViewComponent::pending()),
    );
}

#[legion::system]
pub fn assemble_surface_size(cmd: &mut CommandBuffer, #[state] (entity,): &(Entity,)) {
    cmd.add_component(*entity, SurfaceSizeComponent::new(Default::default()));
    cmd.add_component(*entity, ChangedFlag::<SurfaceSizeComponent>::new_clean());
}

#[legion::system]
pub fn assemble_texture_size(cmd: &mut CommandBuffer, #[state] (entity,): &(Entity,)) {
    cmd.add_component(*entity, TextureSizeComponent::new(Default::default()));
    cmd.add_component(*entity, ChangedFlag::<TextureSizeComponent>::new_clean());
}

pub fn assemble_shader(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: ShaderModuleDescriptor<'static>,
) {
    cmd.add_component(entity, ShaderModuleDescriptorComponent::new(desc));
    cmd.add_component(entity, ShaderModuleComponent::pending());
}

pub fn assemble_shader_usage<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: ShaderModuleDescriptor<'static>,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(ShaderModuleDescriptorComponent::new(desc)),
    );

    cmd.add_component(entity, Usage::<U, _>::new(ShaderModuleComponent::pending()));
}

pub fn assemble_buffer<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: BufferDescriptor<'static>,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(BufferDescriptorComponent::new(desc)),
    );

    cmd.add_component(entity, Usage::<U, _>::new(BufferComponent::pending()));
}

pub fn assemble_buffer_data<U, T>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    data: T,
    offset: BufferAddress,
) where
    U: Send + Sync + 'static,
    T: Component,
{
    cmd.add_component_with_changed_flag_dirty(entity, data);
    cmd.add_component(
        entity,
        Usage::<U, _>::new(BufferWriteComponent::<T>::new(offset)),
    );
    cmd.add_indirect_component_self::<Usage<U, BufferComponent>>(entity);
}

pub fn assemble_texture_data<U, T>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    data: T,
    image_copy_texture: ImageCopyTextureBase<()>,
    image_data_layout: ImageDataLayout,
) where
    T: Component,
    U: Send + Sync + 'static,
{
    cmd.add_component_with_changed_flag_dirty(entity, data);

    // Texture write
    cmd.add_component(
        entity,
        Usage::<U, _>::new(TextureWriteComponent::<T>::new(
            image_copy_texture,
            image_data_layout,
        )),
    );

    // Texture write indirect
    cmd.add_indirect_component_self::<Usage<U, TextureDescriptorComponent>>(entity);
    cmd.add_indirect_component_self::<Usage<U, TextureComponent>>(entity);
}

#[legion::system]
#[read_component(WindowEventComponent)]
#[read_component(WindowEntityMap)]
#[read_component(SurfaceComponent)]
#[read_component(SurfaceTextureComponent)]
#[read_component(ChangedFlag<SurfaceTextureComponent>)]
#[read_component(RenderAttachmentTextureViewDescriptor)]
#[read_component(RenderAttachmentTextureView)]
pub fn surface_textures_views(world: &SubWorld) {
    use legion::IntoQuery;

    let window_event = <&WindowEventComponent>::query()
        .iter(&*world)
        .next()
        .unwrap();
    let window_event = window_event.read().0.expect("No window for current event");

    let window_entity_map = <&WindowEntityMap>::query().iter(&*world).next().unwrap();
    let window_entity_map = window_entity_map.read();

    let entity = window_entity_map
        .get(&window_event)
        .expect("Redraw requested for window without entity");

    // Create surface textures and views
    // These will be rendered to and presented during RedrawEventsCleared
    surface_texture_query(&world, entity);
    surface_texture_view_query(&world, entity);
}

pub fn create_window_surfaces_schedule() -> ImmutableSchedule<Single> {
    single![create_window_surfaces_system()]
}

pub fn submit_and_present_schedule() -> ImmutableSchedule<Serial> {
    serial![
        submit_command_buffers_system(),
        surface_texture_present_system()
        surface_texture_view_drop_system()
    ]
}
