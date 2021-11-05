mod components;
mod systems;

pub use components::*;
pub use systems::*;

#[legion::system]
pub fn assemble_window(
    cmd: &mut legion::systems::CommandBuffer,
    #[state] (entity,): &(legion::Entity,),
) {
    cmd.add_component(*entity, WindowComponent::pending());
}

