mod indirect_component;
mod lazy_component;
mod usage;
mod changed;

pub use indirect_component::*;
pub use lazy_component::*;
pub use usage::*;
pub use changed::*;

use crate::ReadWriteLock;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NameComponent<T>(T);

impl<T, V> ReadWriteLock<V> for NameComponent<T>
where
    T: ReadWriteLock<V>,
{
    fn read(&self) -> parking_lot::RwLockReadGuard<V> {
        self.0.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<V> {
        self.0.write()
    }
}

impl<T> NameComponent<T> {
    pub fn new(name: T) -> Self {
        NameComponent(name)
    }
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SizeComponent<T>(T);

impl<T, V> ReadWriteLock<V> for SizeComponent<T>
where
    T: ReadWriteLock<V>,
{
    fn read(&self) -> parking_lot::RwLockReadGuard<V> {
        self.0.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<V> {
        self.0.write()
    }
}

impl<T> SizeComponent<T> {
    pub fn new(size: T) -> Self {
        SizeComponent(size)
    }
}
