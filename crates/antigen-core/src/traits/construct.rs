// Utility trait for constructing nested newtypes
pub trait Construct<T, I> {
    fn construct(t: T) -> Self;
}

