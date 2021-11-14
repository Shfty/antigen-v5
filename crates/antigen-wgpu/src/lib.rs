mod assemblage;
mod components;
mod systems;
mod to_bytes;

pub use assemblage::*;
pub use components::*;
pub use systems::*;
pub use to_bytes::*;
pub use wgpu;

use antigen_core::{serial, single, ImmutableSchedule, Serial, Single};

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
