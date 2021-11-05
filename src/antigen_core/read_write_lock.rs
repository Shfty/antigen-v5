use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

/// Trait for newtypes that wrap a [`parking_lot::RwLock`]
pub trait ReadWriteLock<T> {
    fn read(&self) -> RwLockReadGuard<T>;
    fn write(&self) -> RwLockWriteGuard<T>;
}

/// Implement ReadWriteLock for a newtype struct
#[macro_export]
macro_rules! impl_read_write_lock {
    ($outer:ty, $field:tt, $inner:ty) => {
        impl $crate::ReadWriteLock<$inner> for $outer {
            fn read(&self) -> $crate::parking_lot::RwLockReadGuard<$inner> {
                self.$field.read()
            }

            fn write(&self) -> $crate::parking_lot::RwLockWriteGuard<$inner> {
                self.$field.write()
            }
        }
    };
}

