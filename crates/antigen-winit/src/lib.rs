mod assemblage;
mod components;
mod systems;

pub use assemblage::*;
pub use components::*;
pub use systems::*;

pub use winit;

use antigen_core::{single, ImmutableSchedule, ImmutableWorld, ReadWriteLock, Single};

use winit::{
    event::Event,
    event_loop::{ControlFlow, EventLoopWindowTarget},
};

use legion::IntoQuery;

/// Extend a winit event loop closure with a world reference and ECS event loop handling
pub fn event_loop_wrapper<T>(
    world: ImmutableWorld,
    mut f: impl FnMut(&ImmutableWorld, Event<T>, &EventLoopWindowTarget<T>, &mut ControlFlow),
) -> impl FnMut(Event<T>, &EventLoopWindowTarget<T>, &mut ControlFlow) {
    move |event, event_loop_window_target, control_flow: &mut winit::event_loop::ControlFlow| {
        let world_read = world.read();
        let window_event = <&WindowEventComponent>::query()
            .iter(&*world_read)
            .next()
            .unwrap();
        *window_event.write() = (None, None);

        let event = if let Some(event) = event.to_static() {
            event
        } else {
            return;
        };

        match &event {
            winit::event::Event::MainEventsCleared => {
                create_windows_thread_local(&world, event_loop_window_target);
            }
            winit::event::Event::RedrawRequested(window_id) => {
                window_event.write().0 = Some(*window_id);
            }
            winit::event::Event::WindowEvent { window_id, event } => {
                *window_event.write() = (Some(*window_id), Some(event.clone()));
            }

            _ => (),
        };

        f(&world, event, event_loop_window_target, control_flow)
    }
}

pub fn window_request_redraw_schedule() -> ImmutableSchedule<Single> {
    single![redraw_windows_on_main_events_cleared_system()]
}
