mod assemblage;
mod components;
mod staging_belt;
mod systems;
mod to_bytes;

use antigen_winit::{
    winit::{
        event::Event,
        event_loop::{ControlFlow, EventLoopWindowTarget},
    },
    EventLoopHandler,
};
pub use assemblage::*;
pub use components::*;
pub use staging_belt::*;
pub use systems::*;
pub use to_bytes::*;
pub use wgpu;

use antigen_core::{parallel, serial, single, ImmutableSchedule, ImmutableWorld, Serial};

pub fn submit_and_present_schedule() -> ImmutableSchedule<Serial> {
    serial![
        submit_command_buffers_system(),
        surface_texture_present_system()
        surface_texture_view_drop_system()
    ]
}

/// Extend an event loop closure with wgpu resource handling
pub fn winit_event_handler<T: Clone>(mut f: impl EventLoopHandler<T>) -> impl EventLoopHandler<T> {
    let mut window_surfaces_schedule = parallel![
        create_window_surfaces_system(),
        serial![
            surface_size_system()
            reconfigure_surfaces_system(),
        ]
    ];

    let mut surface_textures_views_schedule = single![surface_textures_views_system()];

    let mut reset_surface_config_changed_schedule = single![reset_surface_config_changed_system()];

    move |world: &ImmutableWorld,
          event: Event<'static, T>,
          event_loop_window_target: &EventLoopWindowTarget<T>,
          control_flow: &mut ControlFlow| {
        match event {
            Event::MainEventsCleared => {
                window_surfaces_schedule.execute(world);
            }
            Event::RedrawRequested(_) => {
                surface_textures_views_schedule.execute(world);
            }
            _ => (),
        }

        f(world, event.clone(), event_loop_window_target, control_flow);

        match event {
            Event::MainEventsCleared => {
                reset_surface_config_changed_schedule.execute(world);
            }
            _ => (),
        }
    }
}
