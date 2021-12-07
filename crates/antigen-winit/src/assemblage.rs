use antigen_core::{Changed, LazyComponent};
use legion::{Entity, World};

use crate::{
    WindowComponent, WindowEntityMap, WindowEventComponent, WindowSizeComponent,
    WindowTitleComponent,
};

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
            WindowSizeComponent::new(Changed::new(Default::default(), false)),
        );
    }

    fn assemble_winit_window_title(self, entity: Entity, title: &'static str) {
        self.add_component(
            entity,
            WindowTitleComponent::new(Changed::new(title.into(), true)),
        );
    }
}
