mod assemblage;
mod components;
mod systems;

pub use assemblage::*;
pub use components::*;
pub use systems::*;

pub use winit;

use antigen_core::{serial, single, ImmutableWorld, ReadWriteLock};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopWindowTarget},
};

use legion::IntoQuery;

/// A winit-compatible event loop closure
pub trait WinitEventLoopHandler<T>:
    FnMut(Event<T>, &EventLoopWindowTarget<T>, &mut ControlFlow)
{
}
impl<T, U> WinitEventLoopHandler<U> for T where
    T: FnMut(Event<U>, &EventLoopWindowTarget<U>, &mut ControlFlow)
{
}

/// A composable antigen_winit event loop closure
pub trait EventLoopHandler<T>:
    FnMut(&ImmutableWorld, Event<'static, T>, &EventLoopWindowTarget<T>, &mut ControlFlow)
{
}
impl<T, U> EventLoopHandler<U> for T where
    T: FnMut(&ImmutableWorld, Event<'static, U>, &EventLoopWindowTarget<U>, &mut ControlFlow)
{
}

/// Wrap [`EventLoopHandler`] into a [`WinitEventLoopHandler`]
pub fn wrap_event_loop<T>(
    world: ImmutableWorld,
    mut f: impl EventLoopHandler<T>,
) -> impl WinitEventLoopHandler<T> {
    move |event: Event<T>,
          event_loop_window_target: &EventLoopWindowTarget<T>,
          control_flow: &mut winit::event_loop::ControlFlow| {
        let event = if let Some(event) = event.to_static() {
            event
        } else {
            return;
        };

        f(&world, event, event_loop_window_target, control_flow)
    }
}

/// Extend an event loop closure with ECS event loop handling and window functionality
pub fn winit_event_handler<T: Clone>(mut f: impl EventLoopHandler<T>) -> impl EventLoopHandler<T> {
    let mut main_events_cleared_schedule = serial![
        window_title_system(),
        redraw_windows_on_main_events_cleared_system()
    ];

    let mut resize_window_schedule = single![resize_window_system()];
    let mut close_window_schedule = single![close_window_system()];
    let mut reset_resize_window_changed_flag_schedule =
        single![reset_resize_window_dirty_flags_system()];

    move |world: &ImmutableWorld,
          event: Event<'static, T>,
          event_loop_window_target: &EventLoopWindowTarget<T>,
          control_flow: &mut ControlFlow| {
        let world_read = world.read();
        let window_event = <&WindowEventComponent>::query()
            .iter(&*world_read)
            .next()
            .unwrap();
        *window_event.write() = (None, None);

        match &event {
            winit::event::Event::MainEventsCleared => {
                create_windows_thread_local(&world, event_loop_window_target);
                main_events_cleared_schedule.execute(world);
            }
            winit::event::Event::RedrawRequested(window_id) => {
                window_event.write().0 = Some(*window_id);
            }
            winit::event::Event::WindowEvent { window_id, event } => {
                *window_event.write() = (Some(*window_id), Some(event.clone()));
                match event {
                    WindowEvent::Resized(_) => {
                        resize_window_schedule.execute(world);
                    }
                    WindowEvent::CloseRequested => {
                        close_window_schedule.execute(&world);
                    }
                    _ => (),
                }
            }
            _ => (),
        }

        f(world, event.clone(), event_loop_window_target, control_flow);

        match &event {
            winit::event::Event::MainEventsCleared => {
                reset_resize_window_changed_flag_schedule.execute(world);
            }
            _ => (),
        }
    }
}

/// Unit winit event handler
pub fn winit_event_terminator<T>() -> impl EventLoopHandler<T> {
    move |_: &ImmutableWorld,
          _: Event<'static, T>,
          _: &EventLoopWindowTarget<T>,
          _: &mut ControlFlow| {}
}
