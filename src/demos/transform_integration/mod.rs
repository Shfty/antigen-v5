mod components;
mod systems;

pub use components::*;
pub use systems::*;

#[legion::system]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    cmd.push((
        Position::default(),
        Rotation::default(),
        LinearVelocity::new((1.0, 1.0, 1.0)),
    ));
}
