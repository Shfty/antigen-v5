mod components;
mod systems;

pub use components::*;
pub use systems::*;

pub use winit;

use antigen_core::{
    single, ChangedFlag, ImmutableSchedule, ImmutableWorld, ReadWriteLock, Single, SizeComponent,
};

use winit::{
    dpi::PhysicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoopWindowTarget},
};

use legion::IntoQuery;

pub fn assemble_winit_entity(world: &mut legion::World) {
    world.push((WindowEntityMap::new(), EventWindow::new()));
}

#[legion::system]
pub fn assemble_window(
    cmd: &mut legion::systems::CommandBuffer,
    #[state] (entity,): &(legion::Entity,),
) {
    cmd.add_component(*entity, WindowComponent::pending());
}

#[legion::system]
pub fn assemble_window_title(
    cmd: &mut legion::systems::CommandBuffer,
    #[state] (entity,): &(legion::Entity,),
    #[state] title: &&'static str,
) {
    cmd.add_component(
        *entity,
        antigen_core::NameComponent::<&str, WindowTitle>::new(title),
    );
    cmd.add_component(
        *entity,
        ChangedFlag::<antigen_core::NameComponent<&str, WindowTitle>>::new_dirty(),
    );
}

#[legion::system]
pub fn assemble_window_size(
    cmd: &mut legion::systems::CommandBuffer,
    #[state] (entity,): &(legion::Entity,),
) {
    // Add size tracking components to window
    cmd.add_component(
        *entity,
        SizeComponent::<PhysicalSize<u32>, WindowSize>::default(),
    );
    cmd.add_component(
        *entity,
        ChangedFlag::<SizeComponent<PhysicalSize<u32>, WindowSize>>::new_clean(),
    );
}

/// Extend a winit event loop closure with a world reference and ECS event loop handling
pub fn event_loop_wrapper<T>(
    world: ImmutableWorld,
    mut f: impl FnMut(&ImmutableWorld, Event<T>, &EventLoopWindowTarget<T>, &mut ControlFlow),
) -> impl FnMut(Event<T>, &EventLoopWindowTarget<T>, &mut ControlFlow) {
    move |event, event_loop_window_target, control_flow: &mut winit::event_loop::ControlFlow| {
        let world_read = world.read();
        let event_window = <&EventWindow>::query().iter(&*world_read).next().unwrap();
        event_window.set_window(None);

        match &event {
            winit::event::Event::MainEventsCleared => {
                create_windows_thread_local(&world, event_loop_window_target);
            }
            winit::event::Event::RedrawRequested(window_id) => {
                event_window.set_window(Some(*window_id));
            }
            winit::event::Event::WindowEvent { window_id, .. } => {
                event_window.set_window(Some(*window_id));
            }

            _ => (),
        };

        f(&world, event, event_loop_window_target, control_flow)
    }
}

pub fn window_request_redraw_schedule() -> ImmutableSchedule<Single> {
    single![redraw_windows_on_main_events_cleared_system()]
}
