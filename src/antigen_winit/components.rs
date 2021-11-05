use parking_lot::RwLock;

use crate::{LazyComponent, impl_read_write_lock};

// Winit window
pub struct WindowComponent(RwLock<LazyComponent<winit::window::Window>>);

impl WindowComponent {
    pub fn pending() -> WindowComponent {
        WindowComponent(RwLock::new(LazyComponent::Pending))
    }
}

impl_read_write_lock!(WindowComponent, 0, LazyComponent<winit::window::Window>);

// Tag component for a window that redraws unconditionally
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RedrawOnMainEventsCleared;

