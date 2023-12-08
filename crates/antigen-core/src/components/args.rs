use legion::{Entity, World};

use crate::{Construct, Usage};

pub enum EnvArgs {}

pub type ArgsComponent = Usage<EnvArgs, Vec<String>>;

pub fn assemble_args(world: &mut World) -> Entity {
    world.push((ArgsComponent::construct(std::env::args().collect()),))
}
