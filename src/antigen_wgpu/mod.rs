mod components;
mod systems;

pub use components::*;
pub use systems::*;

use crate::DirtyFlag;

/// Create an entity holding the WGPU Instance, Adapter, Device and Queue
#[legion::system]
pub fn assemble_wgpu(cmd: &mut legion::systems::CommandBuffer) {
    // WGPU backend
    let instance =
        wgpu::Instance::new(wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY));
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: Default::default(),
            limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap();

    cmd.push((instance, adapter, device, queue));
}

/// Extends an existing window entity with the means to render to a WGPU surface
#[legion::system]
pub fn assemble_window_surface(
    cmd: &mut legion::systems::CommandBuffer,
    #[state] (entity,): &(legion::Entity,),
) {
    cmd.add_component(
        *entity,
        SurfaceComponent::pending(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: 100,
            height: 100,
            present_mode: wgpu::PresentMode::Mailbox,
        }),
    );
    cmd.add_component(*entity, SurfaceTextureComponent::pending());
    cmd.add_component(*entity, DirtyFlag::<SurfaceTextureComponent>::new_clean());
    cmd.add_component(*entity, TextureViewComponent::pending(Default::default()));
}
