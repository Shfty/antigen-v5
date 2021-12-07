use antigen_core::{AsUsage, Changed, LazyComponent, RwLock, SizeComponent};
use legion::{Entity, World};
use winit::dpi::PhysicalSize;

use crate::{WindowComponent, WindowEntityMap, WindowEventComponent, WindowSize, WindowTitle};

pub fn assemble_winit_backend(world: &mut World) -> Entity {
    world.push((
        WindowEntityMap::new(Default::default()),
        WindowEventComponent::new(),
    ))
}

pub trait AssembleWinit {
    fn assemble_winit_window(self, entity: Entity);
    fn assemble_winit_window_title(self, entity: Entity, title: &'static str);
}

impl AssembleWinit for &mut legion::systems::CommandBuffer {
    fn assemble_winit_window(self, entity: Entity) {
        self.add_component(entity, WindowComponent::new(LazyComponent::Pending));
        self.add_component(
            entity,
            WindowSize::as_usage(Changed::new(
                SizeComponent::new(RwLock::new(PhysicalSize::<u32>::default())),
                false,
            )),
        );
    }

    fn assemble_winit_window_title(self, entity: Entity, title: &'static str) {
        self.add_component(
            entity,
            WindowTitle::as_usage(Changed::new(RwLock::new(title), true)),
        );
    }
}
