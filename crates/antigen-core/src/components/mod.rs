mod lazy_component;
mod indirect_component;
mod dirty_flag;

pub use lazy_component::*;
pub use indirect_component::*;
pub use dirty_flag::*;

use crate::RwLock;

#[derive(Debug)]
pub struct NameComponent<T, U> {
    name: RwLock<T>,
    _phantom: std::marker::PhantomData<U>
}

impl<T, U> crate::ReadWriteLock<T> for NameComponent<T, U> {
    fn read(&self) -> parking_lot::RwLockReadGuard<T> {
        self.name.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<T> {
        self.name.write()
    }
}

impl<T, U> Default for NameComponent<T, U> where T: Default {
    fn default() -> Self {
        NameComponent {
            name: RwLock::new(Default::default()),
            _phantom: Default::default(),
        }
    }
}

impl<T, U> NameComponent<T, U> {
    pub fn new(name: T) -> Self {
        NameComponent {
            name: RwLock::new(name),
            _phantom: Default::default(),
        }
    }
}

#[derive(Debug)]
pub struct SizeComponent<T, U> {
    size: RwLock<T>,
    _phantom: std::marker::PhantomData<U>
}

impl<T, U> crate::ReadWriteLock<T> for SizeComponent<T, U> {
    fn read(&self) -> parking_lot::RwLockReadGuard<T> {
        self.size.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<T> {
        self.size.write()
    }
}

impl<T, U> Default for SizeComponent<T, U> where T: Default {
    fn default() -> Self {
        SizeComponent {
            size: RwLock::new(Default::default()),
            _phantom: Default::default(),
        }
    }
}

impl<T, U> SizeComponent<T, U> {
    pub fn new(size: T) -> Self {
        SizeComponent {
            size: RwLock::new(size),
            _phantom: Default::default(),
        }
    }
}

