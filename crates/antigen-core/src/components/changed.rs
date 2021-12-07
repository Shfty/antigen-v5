use std::{
    borrow::{Borrow, BorrowMut},
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicBool, Ordering},
};

use crate::ReadWriteLock;

// Changed flag
pub struct Changed<T> {
    pub data: T,
    flag: AtomicBool,
}

impl<T> Changed<T> {
    pub fn new(data: T, changed: bool) -> Self {
        Changed {
            data,
            flag: AtomicBool::new(changed),
        }
    }

    pub fn into_inner(self) -> T {
        self.data
    }
}

impl<T> Borrow<T> for Changed<T> {
    fn borrow(&self) -> &T {
        &self.data
    }
}

impl<T> BorrowMut<T> for Changed<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T> Deref for Changed<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Changed<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T, V> ReadWriteLock<V> for Changed<T>
where
    T: ReadWriteLock<V>,
{
    fn read(&self) -> parking_lot::RwLockReadGuard<V> {
        self.data.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<V> {
        self.data.write()
    }
}

/// A type that can get and set a changed flag
pub trait ChangedTrait {
    fn get_changed(&self) -> bool;
    fn set_changed(&self, dirty: bool);
}

impl<T> ChangedTrait for Changed<T> {
    fn get_changed(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    fn set_changed(&self, dirty: bool) {
        self.flag.store(dirty, Ordering::Relaxed);
    }
}

/// Utility trait for constructing a `Changed<T>` via `T::as_changed_*(T)`
pub trait AsChanged: Sized {
    fn as_changed(data: Self, changed: bool) -> Changed<Self> {
        Changed::new(data, changed)
    }
}

impl<T> AsChanged for T {}
