use std::marker::PhantomData;

use legion::{Entity, IntoQuery, storage::Component, systems::CommandBuffer, world::EntityAccessError};

/// A component referenced by entity ID,
/// which can be fetched given a reference to a World or SubWorld
pub struct IndirectComponent<T> {
    target: Entity,
    _phantom: PhantomData<T>,
}

impl<T> IndirectComponent<T> {
    pub fn new(target: Entity) -> Self {
        IndirectComponent {
            target,
            _phantom: Default::default(),
        }
    }
}

pub trait GetIndirect<'a, T> {
    fn get_indirect(self, indirect: &IndirectComponent<T>) -> Result<&'a T, EntityAccessError>;
}

impl<'a, T, S> GetIndirect<'a, T> for &'a S 
where
    T: legion::storage::Component,
    S: legion::EntityStore,
{
    fn get_indirect(self, indirect: &IndirectComponent<T>) -> Result<&'a T, EntityAccessError> {
        <&T>::query().get(self, indirect.target)
    }
}

pub trait AddIndirectComponent {
    fn add_indirect_component<T: Component>(self, entity: Entity, target: Entity);

    fn add_indirect_component_self<T: Component>(self, entity: Entity) where Self: Sized {
        self.add_indirect_component::<T>(entity, entity)
    }

}

impl AddIndirectComponent for &mut CommandBuffer {
    fn add_indirect_component<T: Component>(self, entity: Entity, target: Entity) {
        self.add_component(entity, IndirectComponent::<T>::new(target))
    }
}

