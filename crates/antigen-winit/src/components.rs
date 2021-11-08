use antigen_core::{LazyComponent, RwLock, impl_read_write_lock};

use legion::Entity;
use winit::window::WindowId;

use std::collections::BTreeMap;

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
pub struct RedrawUnconditionally;

// Window ID -> Entity ID map for winit event handling
pub struct WindowEntityMap(RwLock<BTreeMap<WindowId, Entity>>);

impl_read_write_lock!(WindowEntityMap, 0, BTreeMap<WindowId, Entity>);

impl WindowEntityMap {
    pub fn new() -> Self {
        WindowEntityMap(RwLock::new(Default::default()))
    }
}

/// The window corresponding to a winit event
pub struct EventWindow(RwLock<Option<WindowId>>);

impl_read_write_lock!(EventWindow, 0, Option<WindowId>);

impl EventWindow {
    pub fn new() -> Self {
        EventWindow(RwLock::new(None))
    }

    pub fn set_window(&self, window: Option<WindowId>) {
        *self.0.write() = window;
    }

    pub fn get_window(&self) -> Option<WindowId> {
        *self.0.read()
    }
}

/// Usage tag for SizeComponent
pub enum WindowSize {}

/// Usage tag for NameComponent
pub enum WindowTitle {}
