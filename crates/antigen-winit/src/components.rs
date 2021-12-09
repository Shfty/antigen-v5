use antigen_core::{Changed, LazyComponent, RwLock, Usage, impl_read_write_lock};

use legion::Entity;
use winit::{dpi::PhysicalSize, event::WindowEvent, window::WindowId};

use std::collections::BTreeMap;

// Winit window
pub type WindowComponent = RwLock<LazyComponent<winit::window::Window>>;

// Tag component for a window that redraws unconditionally
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RedrawUnconditionally;

// Window ID -> Entity ID map for winit event handling
pub type WindowEntityMap = RwLock<BTreeMap<WindowId, Entity>>;

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
pub type WindowSizeComponent = Usage<WindowSize, Changed<RwLock<PhysicalSize<u32>>>>;

/// Usage tag for NameComponent
pub enum WindowTitle {}
pub type WindowTitleComponent = Usage<WindowTitle, Changed<RwLock<&'static str>>>;
