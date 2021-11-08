
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

