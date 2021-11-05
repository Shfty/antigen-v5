mod components;
mod systems;

pub use components::*;
pub use systems::*;

use crate::{
    CommandBuffersComponent, IndirectComponent, RenderPipelineComponent, SurfaceComponent,
    TextureViewComponent,
};

#[legion::system]
#[read_component(wgpu::Device)]
pub fn assemble(
    cmd: &mut legion::systems::CommandBuffer,
    #[state] (entity, window_entity): &(legion::Entity, legion::Entity),
) {
    cmd.add_component(*entity, HelloTriangle);
    cmd.add_component(*entity, RenderPipelineComponent::<()>::pending());
    cmd.add_component(*entity, CommandBuffersComponent::new());
    cmd.add_component(
        *entity,
        IndirectComponent::<SurfaceComponent>::foreign(*window_entity),
    );
    cmd.add_component(
        *entity,
        IndirectComponent::<TextureViewComponent>::foreign(*window_entity),
    );
}
