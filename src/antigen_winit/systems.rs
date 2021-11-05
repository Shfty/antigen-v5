use super::{RedrawOnMainEventsCleared, WindowComponent};
use crate::{ImmutableWorld, LazyComponent, ReadWriteLock};

use legion::IntoQuery;
use rayon::iter::ParallelIterator;
use std::collections::BTreeMap;
use winit::event_loop::EventLoopWindowTarget;

// Create winit::Window for WindowComponent
pub fn create_windows_thread_local<T>(
    world: &ImmutableWorld,
    event_loop_proxy: &EventLoopWindowTarget<T>,
    window_entity_map: &mut BTreeMap<winit::window::WindowId, legion::Entity>,
) {
    let world_read = world.read();
    let mut iter = <(legion::Entity, &WindowComponent)>::query();

    let pending_entities = iter
        .par_iter(&*world_read)
        .flat_map(
            |(entity, window_component)| match *window_component.read() {
                LazyComponent::Pending => Some(entity),
                _ => None,
            },
        )
        .collect::<Vec<_>>();

    for entity in pending_entities {
        let window_component = <&WindowComponent>::query()
            .get(&*world_read, *entity)
            .unwrap();

        let window = winit::window::Window::new(event_loop_proxy).unwrap();
        window_entity_map.insert(window.id(), *entity);
        *window_component.write() = LazyComponent::Ready(window);
    }
}

// Request redraws for WindowComponents
#[legion::system(par_for_each)]
pub fn redraw_windows_on_main_events_cleared(
    window: &WindowComponent,
    _redraw: &RedrawOnMainEventsCleared,
) {
    match &*window.read() {
        LazyComponent::Ready(window) => window.request_redraw(),
        _ => (),
    }
}
