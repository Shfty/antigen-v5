use antigen_core::{ChangedFlag, Construct, LazyComponent, With};
use legion::{Entity, World};
use winit::dpi::PhysicalSize;

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
        self.add_component(entity, WindowComponent::construct(LazyComponent::Pending));
        self.add_component(
            entity,
            WindowSizeComponent::construct(PhysicalSize::<u32>::default()).with(ChangedFlag(false)),
        );
    }

    fn assemble_winit_window_title(self, entity: Entity, title: &'static str) {
        self.add_component(
            entity,
            WindowTitleComponent::construct(title).with(ChangedFlag(true)),
        );
    }
}
