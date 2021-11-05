use std::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, Ordering},
};

// Dirty flag
pub struct DirtyFlag<T> {
    flag: AtomicBool,
    _phantom: PhantomData<T>,
}

impl<T> DirtyFlag<T> {
    pub fn new_clean() -> Self {
        DirtyFlag {
            flag: AtomicBool::new(false),
            _phantom: Default::default(),
        }
    }

    pub fn new_dirty() -> Self {
        DirtyFlag {
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
