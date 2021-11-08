mod components;
mod systems;
mod to_bytes;

pub use components::*;
pub use systems::*;
pub use to_bytes::*;

use legion::{storage::Component, systems::CommandBuffer, world::SubWorld, Entity, World};
use wgpu::{
    Adapter, BufferAddress, Device, ImageCopyTextureBase, ImageDataLayout, Instance, Queue,
};

pub use wgpu;

use antigen_core::{
    serial, single, AddComponentWithChangedFlag, AddIndirectComponent, ChangedFlag, ImmutableSchedule, ReadWriteLock,
    Serial, Single, SizeComponent,
};
use antigen_winit::{EventWindow, WindowEntityMap};

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
        TextureViewComponent::<RenderAttachment>::pending(Default::default()),
    );
}

#[legion::system]
pub fn assemble_surface_size(cmd: &mut CommandBuffer, #[state] (entity,): &(Entity,)) {
    cmd.add_component(*entity, SizeComponent::<(u32, u32), SurfaceSize>::default());
    cmd.add_component(
        *entity,
        ChangedFlag::<SizeComponent<(u32, u32), SurfaceSize>>::new_clean(),
    );
}

#[legion::system]
pub fn assemble_texture_size(cmd: &mut CommandBuffer, #[state] (entity,): &(Entity,)) {
    cmd.add_component(*entity, SizeComponent::<(u32, u32), TextureSize>::default());
    cmd.add_component(
        *entity,
        ChangedFlag::<SizeComponent<(u32, u32), TextureSize>>::new_clean(),
    );
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
    cmd.add_component(entity, BufferWriteComponent::<U, T>::new(offset));
    cmd.add_indirect_component_self::<BufferComponent<U>>(entity);
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
        TextureWriteComponent::<U, T>::new(image_copy_texture, image_data_layout),
    );

    // Texture write indirect
    cmd.add_indirect_component_self::<TextureComponent<U>>(entity);
}

#[legion::system]
#[read_component(EventWindow)]
#[read_component(WindowEntityMap)]
#[read_component(SurfaceComponent)]
#[read_component(SurfaceTextureComponent)]
#[read_component(ChangedFlag<SurfaceTextureComponent>)]
#[read_component(TextureViewComponent<RenderAttachment>)]
pub fn surface_textures_views(world: &SubWorld) {
    use legion::IntoQuery;

    let event_window = <&EventWindow>::query().iter(&*world).next().unwrap();
    let event_window = event_window
        .get_window()
        .expect("No window for current event");

    let window_entity_map = <&WindowEntityMap>::query().iter(&*world).next().unwrap();
    let window_entity_map = window_entity_map.read();

    let entity = window_entity_map
        .get(&event_window)
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
