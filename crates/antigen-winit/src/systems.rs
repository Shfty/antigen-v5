use super::{RedrawUnconditionally, WindowComponent};
use crate::{WindowEntityMap, WindowEventComponent, WindowSizeComponent, WindowTitleComponent};

use antigen_core::{ChangedFlag, ImmutableWorld, LazyComponent, ReadWriteLock};

use legion::{world::SubWorld, IntoQuery};
use rayon::iter::ParallelIterator;
use winit::{event_loop::EventLoopWindowTarget};

// Create winit::Window for WindowComponent
pub fn create_windows_thread_local<T>(
    world: &ImmutableWorld,
    event_loop_proxy: &EventLoopWindowTarget<T>,
) {
    let world_read = world.read();

    let window_entity_map = <&WindowEntityMap>::query()
        .iter(&*world_read)
        .next()
        .unwrap();

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
        let (window_component, size_component, size_dirty) = <(
            &WindowComponent,
            Option<&WindowSizeComponent>,
            Option<&ChangedFlag<WindowSizeComponent>>,
        )>::query()
        .get(&*world_read, *entity)
        .unwrap();

        let window = winit::window::Window::new(event_loop_proxy).unwrap();
        let size = window.inner_size();

        window_entity_map.write().insert(window.id(), *entity);
        *window_component.write() = LazyComponent::Ready(window);

        if let (Some(window_size), Some(size_dirty)) = (size_component, size_dirty) {
            *window_size.write() = size;
            size_dirty.set(true);
        }
    }
}

// Request redraws for WindowComponents
#[legion::system(par_for_each)]
pub fn redraw_windows_on_main_events_cleared(
    window: &WindowComponent,
    _redraw: &RedrawUnconditionally,
) {
    match &*window.read() {
        LazyComponent::Ready(window) => window.request_redraw(),
        _ => (),
    }
}

#[legion::system]
#[read_component(WindowEventComponent)]
#[read_component(WindowEntityMap)]
#[read_component(WindowComponent)]
#[read_component(WindowSizeComponent)]
#[read_component(ChangedFlag<WindowSizeComponent>)]
pub fn resize_window(world: &SubWorld) {
    let event_window = <&WindowEventComponent>::query()
        .iter(&*world)
        .next()
        .unwrap();

    let window_id = event_window.read().0.expect("No window for current event");

    let window_entity_map = <&WindowEntityMap>::query().iter(world).next().unwrap();
    let window_entity_map = window_entity_map.read();

    let entity = window_entity_map
        .get(&window_id)
        .expect("Resize requested for window without entity");

    let (window_component, size_component, dirty_flag) = if let Ok(components) = <(
        &WindowComponent,
        &WindowSizeComponent,
        &ChangedFlag<WindowSizeComponent>,
    )>::query()
    .get(&*world, *entity)
    {
        components
    } else {
        return;
    };

    if let LazyComponent::Ready(window) = &*window_component.read() {
        *size_component.write() = window.inner_size();
        dirty_flag.set(true);
    }
}

#[legion::system(par_for_each)]
pub fn reset_resize_window_dirty_flags(dirty_flag: &ChangedFlag<WindowSizeComponent>) {
    if dirty_flag.get() {
        println!("Resetting window size changed flag");
        dirty_flag.set(false);
    }
}

#[legion::system]
#[read_component(WindowComponent)]
#[read_component(WindowTitleComponent)]
#[read_component(ChangedFlag<WindowTitleComponent>)]
pub fn window_title(world: &SubWorld) {
    <(
        &WindowComponent,
        &WindowTitleComponent,
        &ChangedFlag<WindowTitleComponent>,
    )>::query()
    .iter(world)
    .for_each(|(window, title, title_dirty)| {
        let window = window.read();
        if let LazyComponent::Ready(window) = &*window {
            if title_dirty.get() {
                window.set_title(&title.read());
                title_dirty.set(false);
            }
        }
    });
}

#[legion::system]
#[read_component(WindowEventComponent)]
#[read_component(WindowEntityMap)]
#[read_component(WindowComponent)]
pub fn close_window(world: &SubWorld) {
    let window_event = <&WindowEventComponent>::query()
        .iter(&*world)
        .next()
        .unwrap();

    let window_event = window_event.read();

    let window_id = if let (Some(window_id), _) = &*window_event {
        window_id
    } else {
        return;
    };

    let window_entity_map = <&WindowEntityMap>::query().iter(world).next().unwrap();
    let window_entity_map = window_entity_map.read();

    let entity = window_entity_map
        .get(&window_id)
        .expect("Close requested for window without entity");

    let window_component = <&WindowComponent>::query().get(&*world, *entity).unwrap();

    if window_component.read().is_ready() {
        window_component.write().set_dropped()
    } else {
        panic!("Close requested for a non-open window");
    }
}
