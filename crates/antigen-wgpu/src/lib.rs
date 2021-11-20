mod assemblage;
mod components;
mod systems;
mod to_bytes;
mod staging_belt;

pub use assemblage::*;
pub use components::*;
pub use systems::*;
pub use to_bytes::*;
pub use staging_belt::*;
pub use wgpu;

use antigen_core::{parallel, serial, ImmutableSchedule, Parallel, Serial};

pub fn window_surfaces_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        create_window_surfaces_system(),
        serial![
            surface_size_system()
            reconfigure_surfaces_system(),
        ]
    ]
}

pub fn submit_and_present_schedule() -> ImmutableSchedule<Serial> {
    serial![
        submit_command_buffers_system(),
        surface_texture_present_system()
        surface_texture_view_drop_system()
    ]
}
