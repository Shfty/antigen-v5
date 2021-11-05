use super::{
    CommandBuffersComponent, SurfaceComponent, SurfaceTextureComponent, TextureViewComponent,
};
use crate::{DirtyFlag, LazyComponent, ReadWriteLock, WindowComponent};

use legion::{world::SubWorld, IntoQuery};
use wgpu::{Adapter, Device, Instance, Queue, Surface, SurfaceConfiguration};

// Initialize pending surfaces that share an entity with a window
#[legion::system(for_each)]
#[read_component(wgpu::Device)]
#[read_component(wgpu::Adapter)]
#[read_component(wgpu::Instance)]
pub fn create_window_surfaces(
    world: &SubWorld,
    window_component: &WindowComponent,
    surface_component: &SurfaceComponent,
) {
    if let LazyComponent::Ready(window) = &*window_component.read() {
        let adapter = <&Adapter>::query().iter(world).next().unwrap();
        let device = <&Device>::query().iter(world).next().unwrap();

        if ReadWriteLock::<LazyComponent<Surface>>::read(surface_component).is_pending() {
            let instance = <&Instance>::query().iter(world).next().unwrap();
            let surface = unsafe { instance.create_surface(window) };
            let mut config = ReadWriteLock::<SurfaceConfiguration>::write(surface_component);

            let window_size = window.inner_size();
            config.width = window_size.width;
            config.height = window_size.height;

            let format = surface
                .get_preferred_format(adapter)
                .expect("Surface is incompatible with adapter");
            config.format = format;

            surface.configure(device, &config);

            ReadWriteLock::<LazyComponent<Surface>>::write(surface_component).set_ready(surface);
        } else if ReadWriteLock::<LazyComponent<Surface>>::read(surface_component).is_ready() {
            let surface_read = surface_component.read();
            let surface = if let LazyComponent::Ready(surface) = &*surface_read {
                surface
            } else {
                unreachable!();
            };

            let mut reconfigure = false;
            let mut config = ReadWriteLock::<SurfaceConfiguration>::write(surface_component);

            let window_size = window.inner_size();
            if config.width != window_size.width {
                config.width = window_size.width;
                reconfigure = true;
            }

            if config.height != window_size.height {
                config.height = window_size.height;
                reconfigure = true;
            }

            let format = surface
                .get_preferred_format(adapter)
                .expect("Surface is incompatible with adapter");
            if config.format != format {
                config.format = format;
                reconfigure = true;
            }

            if config.width > 0 && config.height > 0 && reconfigure {
                surface.configure(device, &config);
            }
        }
    }
}

// Fetch the current surface texture for a given surface, and set its dirty flag
pub fn surface_texture_thread_local(world: &legion::World, entity: &legion::Entity) {
    let (surface, surface_texture, surface_texture_dirty) = if let Ok(components) = <(
        &SurfaceComponent,
        &SurfaceTextureComponent,
        &DirtyFlag<SurfaceTextureComponent>,
    )>::query()
    .get(world, *entity)
    {
        components
    } else {
        return;
    };

    if let LazyComponent::Ready(surface) = &*ReadWriteLock::<LazyComponent<Surface>>::read(surface)
    {
        if let Ok(current) = surface.get_current_texture() {
            *surface_texture.write() = Some(current);
            surface_texture_dirty.set(true);
        } else {
            if surface_texture.read().is_some() {
                surface_texture_dirty.set(true);
                *surface_texture.write() = None;
            }
        }
    }
}

// Create a texture view for a surface texture, unsetting its dirty flag
pub fn surface_texture_view_thread_local(world: &legion::World, entity: &legion::Entity) {
    let (surface_texture, surface_texture_dirty, texture_view) = if let Ok(components) = <(
        &SurfaceTextureComponent,
        &DirtyFlag<SurfaceTextureComponent>,
        &TextureViewComponent,
    )>::query(
    )
    .get(world, *entity)
    {
        components
    } else {
        return;
    };

    if surface_texture_dirty.get() {
        if let Some(surface_texture) = &*surface_texture.read() {
            let view = surface_texture
                .texture
                .create_view(&texture_view.descriptor());
            texture_view.write().set_ready(view);
            surface_texture_dirty.set(false);
        } else {
            texture_view.write().set_dropped();
            surface_texture_dirty.set(false);
        }
    }
}

// Present valid surface textures, setting their dirty flag
#[legion::system(par_for_each)]
pub fn surface_texture_present(
    surface_texture: &SurfaceTextureComponent,
    surface_texture_dirty: &DirtyFlag<SurfaceTextureComponent>,
) {
    if let Some(surface_texture) = surface_texture.write().take() {
        surface_texture.present();
        surface_texture_dirty.set(true);
    }
}

// Drop texture views whose surface textures have been invalidated, unsetting their dirty flag
#[legion::system(par_for_each)]
pub fn surface_texture_view_drop(
    surface_texture: &SurfaceTextureComponent,
    surface_texture_dirty: &DirtyFlag<SurfaceTextureComponent>,
    texture_view: &TextureViewComponent,
) {
    if surface_texture_dirty.get() {
        if surface_texture.read().is_none() {
            texture_view.write().set_dropped();
            surface_texture_dirty.set(false);
        }
    }
}

// Flush command buffers to the WGPU queue
#[legion::system(par_for_each)]
#[read_component(Queue)]
pub fn submit_command_buffers(world: &SubWorld, command_buffers: &CommandBuffersComponent) {
    let queue = if let Some(queue) = <&Queue>::query().iter(world).next() {
        queue
    } else {
        return;
    };

    queue.submit(command_buffers.write().drain(..));
}
