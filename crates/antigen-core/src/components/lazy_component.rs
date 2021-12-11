
/// A lazily-initialized component that can be pending, ready, or dropped 
#[derive(Debug)]
pub enum LazyComponent<T> {
    Pending,
    Ready(T),
    Dropped,
}

impl<T> LazyComponent<T> {
    pub fn is_pending(&self) -> bool {
        matches!(self, LazyComponent::Pending)
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, LazyComponent::Ready(_))
    }

    pub fn is_dropped(&self) -> bool {
        matches!(self, LazyComponent::Dropped)
    }

    pub fn set_pending(&mut self) {
        *self = LazyComponent::Pending;
    }

    pub fn set_ready(&mut self, inner: T) {
        *self = LazyComponent::Ready(inner);
    }

    pub fn set_dropped(&mut self) {
        *self = LazyComponent::Dropped;
    }
}

/// Read a RwLock<LazyComponent<T>> and match against LazyComponent::Ready(T), else panic
#[macro_export]
macro_rules! lazy_read_ready_else_panic {
    ($cmp:ident) => {
        let $cmp = $crate::ReadWriteLock::read($cmp);
        let $cmp = if let LazyComponent::Ready($cmp) = &*$cmp {
            $cmp
        } else {
            panic!("LazyComponent is not ready");
        };
    };
}

/// Read a RwLock<LazyComponent<T>> and match against LazyComponent::Ready(T), else continue
#[macro_export]
macro_rules! lazy_read_ready_else_continue {
    ($cmp:ident) => {
        let $cmp = $crate::ReadWriteLock::read($cmp);
        let $cmp = if let LazyComponent::Ready($cmp) = &*$cmp {
            $cmp
        } else {
            continue;
        };
    };
}

/// Read a RwLock<LazyComponent<T>> and match against LazyComponent::Ready(T), else return
#[macro_export]
macro_rules! lazy_read_ready_else_return {
    ($cmp:ident) => {
        let $cmp = $crate::ReadWriteLock::read($cmp);
        let $cmp = if let LazyComponent::Ready($cmp) = &*$cmp {
            $cmp
        } else {
            return;
        };
    };
}
