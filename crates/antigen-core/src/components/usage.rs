use std::marker::PhantomData;

use crate::ReadWriteLock;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Usage<U, T> {
    data: T,
    _phantom: PhantomData<U>,
}

impl<U, T> Usage<U, T> {
    pub fn new(data: T) -> Self {
        Usage {
            data,
            _phantom: Default::default()
        }
    }
}

impl<U, T, V> ReadWriteLock<V> for Usage<U, T> where T: ReadWriteLock<V> {
    fn read(&self) -> parking_lot::RwLockReadGuard<V> {
        self.data.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<V> {
        self.data.write()
    }
}
