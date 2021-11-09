use antigen_core::{LazyComponent, RwLock, Usage, impl_read_write_lock};

use legion::Entity;
use winit::{dpi::PhysicalSize, event::WindowEvent, window::WindowId};

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

/// Window event wrapper
pub struct WindowEventComponent(RwLock<(Option<WindowId>, Option<WindowEvent<'static>>)>);

impl_read_write_lock!(WindowEventComponent, 0, (Option<WindowId>, Option<WindowEvent<'static>>));

impl WindowEventComponent {
    pub fn new() -> Self {
        WindowEventComponent(RwLock::new((None, None)))
    }
}

/// Usage tag for SizeComponent
pub enum WindowSize {}

/// Usage tag for NameComponent
pub enum WindowTitle {}

pub type WindowTitleComponent = Usage<WindowTitle, RwLock<&'static str>>;
pub type WindowSizeComponent = Usage<WindowSize, RwLock<PhysicalSize<u32>>>;
