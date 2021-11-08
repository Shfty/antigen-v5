use std::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, Ordering},
};

// Dirty flag
pub struct ChangedFlag<T> {
    flag: AtomicBool,
    _phantom: PhantomData<T>,
}

impl<T> ChangedFlag<T> {
    pub fn new_clean() -> Self {
        ChangedFlag {
            flag: AtomicBool::new(false),
            _phantom: Default::default(),
        }
    }

    pub fn new_dirty() -> Self {
        ChangedFlag {
            flag: AtomicBool::new(true),
            _phantom: Default::default(),
        }
    }

    pub fn set(&self, dirty: bool) {
        self.flag.store(dirty, Ordering::Relaxed);
    }

    pub fn get(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// Trait for adding some component T to a world alongside a DirtyFlag<T>
pub trait AddComponentWithChangedFlag<T> {
    fn add_component_with_changed_flag(self, entity: legion::Entity, component: T, dirty: bool);

    fn add_component_with_changed_flag_clean(self, entity: legion::Entity, component: T)
    where
        Self: Sized,
    {
        self.add_component_with_changed_flag(entity, component, false)
    }

    fn add_component_with_changed_flag_dirty(self, entity: legion::Entity, component: T)
    where
        Self: Sized,
    {
        self.add_component_with_changed_flag(entity, component, true)
    }
}

impl<T: legion::storage::Component> AddComponentWithChangedFlag<T>
    for &mut legion::systems::CommandBuffer
{
    fn add_component_with_changed_flag(self, entity: legion::Entity, component: T, dirty: bool) {
        self.add_component(entity, component);
        self.add_component(
            entity,
            if dirty {
                ChangedFlag::<T>::new_dirty()
            } else {
                ChangedFlag::<T>::new_clean()
            },
        );
    }
}
