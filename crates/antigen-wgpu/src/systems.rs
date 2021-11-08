use super::{
    BufferWriteComponent, CommandBuffersComponent, RenderAttachment, SurfaceComponent,
    SurfaceTextureComponent, TextureViewComponent, TextureWriteComponent, ToBytes,
};
use crate::{BufferComponent, SurfaceSize, TextureComponent, TextureSize, SamplerComponent, ShaderModuleComponent};

use antigen_core::{ChangedFlag, GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock};
use antigen_winit::{winit::dpi::PhysicalSize, WindowComponent, WindowSize};

use legion::{world::SubWorld, IntoQuery};
use wgpu::{
    Adapter, Device, ImageCopyTextureBase, ImageDataLayout, Instance, Queue, Surface,
    SurfaceConfiguration,
};

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
pub fn surface_texture_query(world: &legion::world::SubWorld, entity: &legion::Entity) {
    let (surface, surface_texture, surface_texture_dirty) = if let Ok(components) = <(
        &SurfaceComponent,
        &SurfaceTextureComponent,
        &ChangedFlag<SurfaceTextureComponent>,
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
pub fn surface_texture_view_query(world: &legion::world::SubWorld, entity: &legion::Entity) {
    let (surface_texture, surface_texture_dirty, texture_view) = if let Ok(components) = <(
        &SurfaceTextureComponent,
        &ChangedFlag<SurfaceTextureComponent>,
        &TextureViewComponent<RenderAttachment>,
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

#[legion::system(par_for_each)]
pub fn surface_size(
    window_size: &antigen_core::SizeComponent<PhysicalSize<u32>, WindowSize>,
    window_size_dirty: &ChangedFlag<antigen_core::SizeComponent<PhysicalSize<u32>, WindowSize>>,
    surface_size: &antigen_core::SizeComponent<(u32, u32), SurfaceSize>,
    surface_size_dirty: &ChangedFlag<antigen_core::SizeComponent<(u32, u32), SurfaceSize>>,
) {
    if window_size_dirty.get() {
        let window_size = *window_size.read();
        *surface_size.write() = (window_size.width, window_size.height);
        surface_size_dirty.set(true);
    }
}

#[legion::system(par_for_each)]
pub fn surface_texture_size(
    surface_size: &antigen_core::SizeComponent<(u32, u32), SurfaceSize>,
    surface_size_dirty: &ChangedFlag<antigen_core::SizeComponent<(u32, u32), SurfaceSize>>,
    texture_size: &antigen_core::SizeComponent<(u32, u32), TextureSize>,
    texture_size_dirty: &ChangedFlag<antigen_core::SizeComponent<(u32, u32), TextureSize>>,
) {
    if surface_size_dirty.get() {
        let size = *surface_size.read();
        *texture_size.write() = size;
        texture_size_dirty.set(true);
    }
}

#[legion::system(par_for_each)]
pub fn reset_surface_size_dirty_flag(
    surface_size_dirty: &ChangedFlag<antigen_core::SizeComponent<(u32, u32), SurfaceSize>>,
) {
    if surface_size_dirty.get() {
        surface_size_dirty.set(false);
    }
}

#[legion::system(par_for_each)]
pub fn reset_texture_size_dirty_flag(
    texture_size_dirty: &ChangedFlag<antigen_core::SizeComponent<(u32, u32), TextureSize>>,
) {
    if texture_size_dirty.get() {
        texture_size_dirty.set(false);
    }
}

// Present valid surface textures, setting their dirty flag
#[legion::system(par_for_each)]
pub fn surface_texture_present(
    surface_texture: &SurfaceTextureComponent,
    surface_texture_dirty: &ChangedFlag<SurfaceTextureComponent>,
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
    surface_texture_dirty: &ChangedFlag<SurfaceTextureComponent>,
    texture_view: &TextureViewComponent<RenderAttachment>,
) {
    if surface_texture_dirty.get() {
        if surface_texture.read().is_none() {
            texture_view.write().set_dropped();
            surface_texture_dirty.set(false);
        }
    }
}

#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_buffers<T: Send + Sync + 'static>(world: &SubWorld, buffer: &BufferComponent<T>) {
    if buffer.read().is_pending() {
        let device = <&Device>::query().iter(world).next().unwrap();
        println!("Created {} buffer", std::any::type_name::<T>());
        buffer
            .write()
            .set_ready(device.create_buffer(buffer.desc()));
    }
}

#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_textures<T: Send + Sync + 'static>(world: &SubWorld, texture: &TextureComponent<T>) {
    if texture.read().is_pending() {
        let device = <&Device>::query().iter(world).next().unwrap();
        texture
            .write()
            .set_ready(device.create_texture(texture.desc()));
    }
}

#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_texture_views<T: Send + Sync + 'static>(
    texture: &TextureComponent<T>,
    texture_view: &TextureViewComponent<T>,
) {
    if !texture_view.read().is_pending() {
        return;
    }

    let texture = texture.read();
    let texture = if let LazyComponent::Ready(texture) = &*texture {
        texture
    } else {
        return;
    };

    println!("Creating texture view");
    texture_view
        .write()
        .set_ready(texture.create_view(texture_view.descriptor()));
}

#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_samplers<T: Send + Sync + 'static>(world: &SubWorld, sampler: &SamplerComponent<T>) {
    if sampler.read().is_pending() {
        let device = <&Device>::query().iter(world).next().unwrap();
        sampler
            .write()
            .set_ready(device.create_sampler(sampler.descriptor()));
    }
}

#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_shader_modules<T: Send + Sync + 'static>(world: &SubWorld, shader_module: &ShaderModuleComponent<T>) {
    if shader_module.read().is_pending() {
        let device = <&Device>::query().iter(world).next().unwrap();
        shader_module
            .write()
            .set_ready(device.create_shader_module(shader_module.descriptor()));
    }
}

// Write data to buffer
#[legion::system]
#[read_component(Queue)]
#[read_component(BufferWriteComponent<T, L>)]
#[read_component(L)]
#[read_component(ChangedFlag<L>)]
#[read_component(IndirectComponent<BufferComponent<T>>)]
#[read_component(BufferComponent<T>)]
pub fn buffer_write<
    T: Send + Sync + 'static,
    L: ReadWriteLock<V> + Send + Sync + 'static,
    V: ToBytes,
>(
    world: &SubWorld,
) {
    let queue = if let Some(queue) = <&Queue>::query().iter(world).next() {
        queue
    } else {
        return;
    };

    <(
        &BufferWriteComponent<T, L>,
        &L,
        &ChangedFlag<L>,
        &IndirectComponent<BufferComponent<T>>,
    )>::query()
    .par_for_each(
        world,
        |(buffer_write, value, dirty_flag, buffer)| {
            let buffer = world.get_indirect(buffer).unwrap();

            if dirty_flag.get() {
                let buffer = buffer.read();
                let buffer = if let LazyComponent::Ready(buffer) = &*buffer {
                    buffer
                } else {
                    return;
                };

                let value = value.read();
                let bytes = value.to_bytes();

                println!(
                    "Writing {} bytes to {} buffer at offset {}",
                    bytes.len(),
                    std::any::type_name::<T>(),
                    *buffer_write.read()
                );
                queue.write_buffer(buffer, *buffer_write.read(), bytes);

                dirty_flag.set(false);
            }
        },
    );
}

// Write data to texture
#[legion::system]
#[read_component(Queue)]
#[read_component(TextureWriteComponent<T, L>)]
#[read_component(L)]
#[read_component(ChangedFlag<L>)]
#[read_component(IndirectComponent<TextureComponent<T>>)]
#[read_component(TextureComponent<T>)]
pub fn texture_write<T, L, V>(world: &SubWorld)
where
    T: Send + Sync + 'static,
    L: ReadWriteLock<V> + Send + Sync + 'static,
    V: ToBytes,
{
    let queue = if let Some(queue) = <&Queue>::query().iter(world).next() {
        queue
    } else {
        return;
    };

    <(
        &TextureWriteComponent<T, L>,
        &L,
        &ChangedFlag<L>,
        &IndirectComponent<TextureComponent<T>>,
    )>::query()
    .par_for_each(
        world,
        |(texture_write, texels, dirty_flag, texture)| {
            let texture_component = world.get_indirect(texture).unwrap();

            if dirty_flag.get() {
                let texture = texture_component.read();
                let texture = if let LazyComponent::Ready(texture) = &*texture {
                    texture
                } else {
                    return;
                };

                let texels = texels.read();
                let bytes = texels.to_bytes();
                let desc = texture_component.desc();
                let image_copy_texture =
                    ReadWriteLock::<ImageCopyTextureBase<()>>::read(texture_write);
                let image_data_layout = ReadWriteLock::<ImageDataLayout>::read(texture_write);

                println!(
                    "Writing {} bytes to texture at offset {}",
                    bytes.len(),
                    ReadWriteLock::<wgpu::ImageDataLayout>::read(texture_write).offset,
                );

                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &*texture,
                        mip_level: image_copy_texture.mip_level,
                        origin: image_copy_texture.origin,
                        aspect: image_copy_texture.aspect,
                    },
                    bytes,
                    *image_data_layout,
                    desc.size,
                );

                dirty_flag.set(false);
            }
        },
    );
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
