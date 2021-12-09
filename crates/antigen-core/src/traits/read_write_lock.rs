use std::{ops::Deref, sync::Arc};

pub use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Trait for newtypes that wrap a [`parking_lot::RwLock`]
pub trait ReadWriteLock<T> {
    fn read(&self) -> RwLockReadGuard<T>;
    fn write(&self) -> RwLockWriteGuard<T>;
}

impl<T> ReadWriteLock<T> for RwLock<T> {
    fn read(&self) -> RwLockReadGuard<T> {
        self.read()
    }

    fn write(&self) -> RwLockWriteGuard<T> {
        self.write()
    }
}

impl<T> ReadWriteLock<T> for Arc<RwLock<T>> {
    fn read(&self) -> RwLockReadGuard<T> {
        self.deref().read()
    }

    fn write(&self) -> RwLockWriteGuard<T> {
        self.deref().write()
    }
}

/// Construct implementation
impl<T> crate::Construct<T, crate::peano::Z> for RwLock<T> {
    fn construct(t: T) -> Self {
        RwLock::new(t)
    }
}

impl<T, I, N> crate::Construct<T, crate::peano::S<I>> for RwLock<N>
where
    N: crate::Construct<T, I>,
{
    fn construct(t: T) -> Self {
        RwLock::new(N::construct(t))
    }
}

/// Implement ReadWriteLock for a newtype struct
#[macro_export]
macro_rules! impl_read_write_lock {
    ($outer:ty, $field:tt, $inner:ty) => {
        impl $crate::ReadWriteLock<$inner> for $outer {
            fn read(&self) -> $crate::RwLockReadGuard<$inner> {
                self.$field.read()
            }

            fn write(&self) -> $crate::RwLockWriteGuard<$inner> {
                self.$field.write()
            }
        }
    };
}
