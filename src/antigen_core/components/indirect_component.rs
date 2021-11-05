use std::{convert::Infallible, marker::PhantomData};

use legion::{Entity, IntoQuery};

/// A component referenced by entity ID,
/// which can be fetched given a reference to a World or SubWorld
pub enum IndirectComponent<T> {
    Local,
    Foreign(legion::Entity),
    #[non_exhaustive]
    _Phantom(Infallible, PhantomData<T>),
}

impl<T> IndirectComponent<T> {
    pub fn local() -> Self {
        IndirectComponent::Local
    }

    pub fn foreign(entity: legion::Entity) -> Self {
        IndirectComponent::Foreign(entity)
    }

    pub fn get_world<'a>(
        &self,
        world: &'a legion::World,
        entity: &Entity,
    ) -> Result<&'a T, legion::world::EntityAccessError>
    where
        T: legion::storage::Component,
    {
        match self {
            IndirectComponent::Local => <&T>::query().get(world, *entity),
            IndirectComponent::Foreign(entity) => <&T>::query().get(world, *entity),
            _ => unreachable!()
        }
    }

    pub fn get_sub_world<'a>(
        &self,
        world: &'a legion::world::SubWorld,
        entity: &Entity,
    ) -> Result<&'a T, legion::world::EntityAccessError>
    where
        T: legion::storage::Component,
    {
        match self {
            IndirectComponent::Local => <&T>::query().get(world, *entity),
            IndirectComponent::Foreign(entity) => <&T>::query().get(world, *entity),
            _ => unreachable!()
        }
    }
}
