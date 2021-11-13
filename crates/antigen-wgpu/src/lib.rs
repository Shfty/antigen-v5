mod assemblage;
mod components;
mod systems;
mod to_bytes;

pub use assemblage::*;
pub use components::*;
pub use systems::*;
pub use to_bytes::*;
pub use wgpu;

use antigen_core::{
    serial, single, ChangedFlag, ImmutableSchedule, ReadWriteLock, Serial, Single,
};
use antigen_winit::{WindowEntityMap, WindowEventComponent};

use legion::world::SubWorld;

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
