use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::ReadWriteLock;

/// Wrapper type for creating several distinct types out of the same underlying type
///
/// ex. You may have a SizeComponent type to represent size, but need multiple instances
/// of it on the same entity that represent different sizes.
///
/// Usage can be used to solve this problem by creating usage-specific variants of the base type:
/// ```
/// enum WindowSize = {};
/// enum SurfaceSize = {};
/// enum TextureSize = {};
///
/// type SizeComponent = (u32, u32);
///
/// type WindowSizeComponent = Usage<WindowSize, SizeComponent>;
/// type SurfaceSizeComponent = Usage<SurfaceSize, SizeComponent>;
/// type TextureSizeComponent = Usage<TextureSize, SizeComponent>;
/// ```
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Usage<U, T> {
    pub data: T,
    _phantom: PhantomData<U>,
}

impl<U, T> Usage<U, T> {
    pub fn new(data: T) -> Self {
        Usage {
            data,
            _phantom: Default::default(),
        }
    }
}

impl<U, T> From<T> for Usage<U, T> {
    fn from(t: T) -> Self {
        Usage::new(t)
    }
}

impl<U, T> Borrow<T> for Usage<U, T> {
    fn borrow(&self) -> &T {
        &self.data
    }
}

impl<U, T> BorrowMut<T> for Usage<U, T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<U, T> Deref for Usage<U, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<U, T> DerefMut for Usage<U, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<U, T, V> ReadWriteLock<V> for Usage<U, T>
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
