mod components;
mod systems;

use antigen_core::{ImmutableSchedule, Parallel, Serial, parallel, serial};
pub use components::*;
pub use systems::*;

#[legion::system]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    cmd.push((
        Position::default(),
        Rotation::default(),
        LinearVelocity::new((1.0, 1.0, 1.0)),
        AngularVelocity::new(0.5),
    ));
}

pub fn integrate_schedule() -> ImmutableSchedule<Parallel> {
    parallel![integrate_position_system(), integrate_rotation_system(),]
}

pub fn print_schedule() -> ImmutableSchedule<Serial> {
    serial![print_position_system(), print_rotation_system(),]
}
