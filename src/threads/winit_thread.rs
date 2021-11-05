use std::collections::BTreeMap;

use crate::{
    create_windows_thread_local, surface_texture_thread_local,
    surface_texture_view_thread_local, ImmutableSchedule, ImmutableWorld, ReadWriteLock,
    RunSchedule, WindowComponent,
};

use legion::{Entity, IntoQuery};
use winit::{event_loop::EventLoop, window::WindowId};

pub fn winit_thread<MEC: RunSchedule + 'static, REC: RunSchedule + 'static>(
    world: ImmutableWorld,
    mut main_events_cleared_schedule: ImmutableSchedule<MEC>,
    mut redraw_events_cleared_schedule: ImmutableSchedule<REC>,
) -> ! {
    let event_loop = EventLoop::new();
    let mut window_entity_map = BTreeMap::<WindowId, Entity>::default();

    event_loop.run(
        move |event: winit::event::Event<()>,
              event_loop_window_target: &winit::event_loop::EventLoopWindowTarget<()>,
              _control_flow: &mut winit::event_loop::ControlFlow| {
            match event {
                winit::event::Event::MainEventsCleared => {
                    create_windows_thread_local(
                        &world,
                        event_loop_window_target,
                        &mut window_entity_map,
                    );

                    main_events_cleared_schedule.execute(&world);
                }
                winit::event::Event::RedrawRequested(window_id) => {
                    let entity = window_entity_map
                        .get(&window_id)
                        .expect("Redraw requested for window without entity");

                    // Create surface textures and views
                    // These will be rendered to and presented during RedrawEventsCleared
                    surface_texture_thread_local(&world.read(), entity);
                    surface_texture_view_thread_local(&world.read(), entity);
                }
                winit::event::Event::WindowEvent { window_id, event } => match event {
                    winit::event::WindowEvent::CloseRequested => {
                        let world = world.read();

                        let window_component = <&WindowComponent>::query()
                            .get(&*world, window_entity_map[&window_id])
                            .unwrap();

                        if window_component.read().is_ready() {
                            window_component.write().set_dropped()
                        } else {
                            panic!("Close requested for a non-open window");
                        }
                    }
                    _ => (),
                },
                winit::event::Event::RedrawEventsCleared => {
                    redraw_events_cleared_schedule.execute(&world);
                }

                _ => (),
            }
        },
    )
}
