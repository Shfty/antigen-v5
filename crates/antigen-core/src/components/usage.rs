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
    pub fn into_inner(self) -> T {
        self.data
    }
}

impl<U, T> From<T> for Usage<U, T> {
    fn from(t: T) -> Self {
        U::as_usage(t)
    }
}

// Data access traits
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

// ReadWriteLock implementation
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

/// Trait for constructing a [`Usage<U, T>`] via `U::as_usage(T)`
pub trait AsUsage: Sized {
    fn as_usage<T>(data: T) -> Usage<Self, T> {
        Usage {
            data,
            _phantom: Default::default(),
        }
    }
}

impl<T> AsUsage for T {}

/// Construct implementation
impl<U, T> crate::Construct<T, crate::peano::Z> for Usage<U, T> {
    fn construct(t: T) -> Self {
        Usage {
            data: t,
            _phantom: Default::default(),
        }
    }
}

impl<T, I, U, N> crate::Construct<T, crate::peano::S<I>> for Usage<U, N>
where
    N: crate::Construct<T, I>,
{
    fn construct(t: T) -> Self {
        Usage {
            data: N::construct(t),
            _phantom: Default::default(),
        }
    }
}

/// With implementation
impl<T, I, U, N> crate::With<T, crate::peano::S<I>> for Usage<U, N>
where
    N: crate::With<T, I>,
{
    fn with(self, t: T) -> Self {
        Usage {
            data: self.data.with(t),
            ..self
        }
    }
}
