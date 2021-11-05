use std::sync::Arc;

use legion::World;
use parking_lot::RwLock;

use crate::impl_read_write_lock;

#[derive(Debug, Clone)]
pub struct ImmutableWorld(Arc<RwLock<World>>);

impl ImmutableWorld {
    pub fn new(world: World) -> Self {
        ImmutableWorld(Arc::new(RwLock::new(world)))
    }
}

impl Default for ImmutableWorld {
    fn default() -> Self {
        ImmutableWorld::new(World::default())
    }
}

impl_read_write_lock!(ImmutableWorld, 0, World);
