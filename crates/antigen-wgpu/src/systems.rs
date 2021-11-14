use super::{
    BufferWriteComponent, CommandBuffersComponent, RenderAttachmentTextureViewDescriptor,
    SurfaceComponent, SurfaceTextureComponent, TextureDescriptorComponent, TextureViewComponent,
    TextureViewDescriptorComponent, TextureWriteComponent, ToBytes,
};
use crate::{
    BufferComponent, BufferDescriptorComponent, RenderAttachmentTextureView, SamplerComponent,
    SamplerDescriptorComponent, ShaderModuleComponent, ShaderModuleDescriptorComponent,
    SurfaceSizeComponent, TextureComponent, TextureSizeComponent,
};

use antigen_core::{
    ChangedFlag, GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock, Usage,
};
use antigen_winit::{WindowComponent, WindowEntityMap, WindowEventComponent, WindowSizeComponent};

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
    let (surface_texture, surface_texture_dirty, texture_view_desc, texture_view) =
        if let Ok(components) = <(
            &SurfaceTextureComponent,
            &ChangedFlag<SurfaceTextureComponent>,
            &RenderAttachmentTextureViewDescriptor,
            &RenderAttachmentTextureView,
        )>::query()
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
                .create_view(&texture_view_desc.read());
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
    window_size: &WindowSizeComponent,
    window_size_dirty: &ChangedFlag<WindowSizeComponent>,
    surface_size: &SurfaceSizeComponent,
    surface_size_dirty: &ChangedFlag<SurfaceSizeComponent>,
) {
    if window_size_dirty.get() {
        let window_size = *window_size.read();
        *surface_size.write() = (window_size.width, window_size.height);
        surface_size_dirty.set(true);
    }
}

#[legion::system(par_for_each)]
pub fn surface_texture_size(
    surface_size: &SurfaceSizeComponent,
    surface_size_dirty: &ChangedFlag<SurfaceSizeComponent>,
    texture_size: &TextureSizeComponent,
    texture_size_dirty: &ChangedFlag<TextureSizeComponent>,
) {
    if surface_size_dirty.get() {
        let size = *surface_size.read();
        *texture_size.write() = size;
        texture_size_dirty.set(true);
    }
}

#[legion::system(par_for_each)]
pub fn reset_surface_size_dirty_flag(surface_size_dirty: &ChangedFlag<SurfaceSizeComponent>) {
    if surface_size_dirty.get() {
        surface_size_dirty.set(false);
    }
}

#[legion::system(par_for_each)]
pub fn reset_texture_size_dirty_flag(texture_size_dirty: &ChangedFlag<TextureSizeComponent>) {
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
    surface_texture_changed: &ChangedFlag<SurfaceTextureComponent>,
    texture_view: &RenderAttachmentTextureView,
) {
    if surface_texture_changed.get() {
        if surface_texture.read().is_none() {
            texture_view.write().set_dropped();
            surface_texture_changed.set(false);
        }
    }
}

/// Create pending untagged shader modules,
/// optionally recreating them if a ChangedFlag is present and set
#[legion::system(for_each)]
#[read_component(Device)]
pub fn create_shader_modules(
    world: &SubWorld,
    shader_module_desc: &ShaderModuleDescriptorComponent,
    shader_module: &ShaderModuleComponent,
    shader_module_desc_changed: Option<&ChangedFlag<SamplerDescriptorComponent>>,
) {
    if shader_module.read().is_pending()
        || shader_module_desc_changed
            .map(|shader_module_desc_changed| shader_module_desc_changed.get())
            .unwrap_or(false)
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        shader_module
            .write()
            .set_ready(device.create_shader_module(&shader_module_desc.read()));
        println!("Created shader module");
    }
}

/// Create pending usage-tagged shader modules,
/// optionally recreating them if a ChangedFlag is present and set
#[legion::system(for_each)]
#[read_component(Device)]
pub fn create_shader_modules_usage<T: Send + Sync + 'static>(
    world: &SubWorld,
    shader_module_desc: &Usage<T, ShaderModuleDescriptorComponent>,
    shader_module: &Usage<T, ShaderModuleComponent>,
    shader_module_desc_changed: Option<&Usage<T, ChangedFlag<SamplerDescriptorComponent>>>,
) {
    if shader_module.read().is_pending()
        || shader_module_desc_changed
            .map(|shader_module_desc_changed| shader_module_desc_changed.get())
            .unwrap_or(false)
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        shader_module
            .write()
            .set_ready(device.create_shader_module(&shader_module_desc.read()));
        println!("Created {} shader module", std::any::type_name::<T>());
    }
}

/// Create pending usage-tagged buffers,
/// optionally recreating them if a ChangedFlag is present and set
#[legion::system(for_each)]
#[read_component(Device)]
pub fn create_buffers<T: Send + Sync + 'static>(
    world: &SubWorld,
    buffer_desc: &Usage<T, BufferDescriptorComponent>,
    buffer: &Usage<T, BufferComponent>,
    buffer_desc_changed: Option<&Usage<T, ChangedFlag<TextureDescriptorComponent>>>,
) {
    if buffer.read().is_pending()
        || buffer_desc_changed
            .map(|buffer_desc_changed| buffer_desc_changed.get())
            .unwrap_or(false)
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        println!("Created {} buffer", std::any::type_name::<T>());
        buffer
            .write()
            .set_ready(device.create_buffer(&buffer_desc.read()));
    }
}

/// Create pending usage-tagged textures,
/// optionally recreating them if a ChangedFlag is present and set
#[legion::system(for_each)]
#[read_component(Device)]
pub fn create_textures<T: Send + Sync + 'static>(
    world: &SubWorld,
    texture_descriptor: &Usage<T, TextureDescriptorComponent>,
    texture: &Usage<T, TextureComponent>,
    texture_descriptor_changed: Option<&Usage<T, ChangedFlag<TextureDescriptorComponent>>>,
) {
    if texture.read().is_pending()
        || texture_descriptor_changed
            .map(|texture_descriptor_changed| texture_descriptor_changed.get())
            .unwrap_or(false)
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        texture
            .write()
            .set_ready(device.create_texture(&*texture_descriptor.read()));
    }
}

/// Create pending usage-tagged texture views,
/// optionally recreating them if a ChangedFlag is present and set
#[legion::system(for_each)]
#[read_component(Device)]
pub fn create_texture_views<T: Send + Sync + 'static>(
    texture: &Usage<T, TextureComponent>,
    texture_view_desc: &Usage<T, TextureViewDescriptorComponent>,
    texture_view: &Usage<T, TextureViewComponent>,
    texture_view_desc_changed: Option<&Usage<T, ChangedFlag<TextureViewDescriptorComponent>>>,
) {
    if !texture_view.read().is_pending()
        && !texture_view_desc_changed
            .map(|texture_view_desc_changed| texture_view_desc_changed.get())
            .unwrap_or(false)
    {
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
        .set_ready(texture.create_view(&texture_view_desc.read()));
}

/// Create pending usage-tagged samplers,
/// optionally recreating them if a ChangedFlag is present and set
#[legion::system(for_each)]
#[read_component(Device)]
pub fn create_samplers<T: Send + Sync + 'static>(
    world: &SubWorld,
    sampler_desc: &Usage<T, SamplerDescriptorComponent>,
    sampler: &Usage<T, SamplerComponent>,
    sampler_desc_changed: Option<&Usage<T, ChangedFlag<SamplerDescriptorComponent>>>,
) {
    if sampler.read().is_pending()
        || sampler_desc_changed
            .map(|sampler_desc_changed| sampler_desc_changed.get())
            .unwrap_or(false)
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        sampler
            .write()
            .set_ready(device.create_sampler(&sampler_desc.read()));
    }
}

// Write data to buffer
#[legion::system]
#[read_component(Queue)]
#[read_component(Usage<T, BufferWriteComponent<L>>)]
#[read_component(L)]
#[read_component(ChangedFlag<L>)]
#[read_component(IndirectComponent<Usage<T, BufferComponent>>)]
#[read_component(Usage<T, BufferComponent>)]
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
        &Usage<T, BufferWriteComponent<L>>,
        &L,
        &ChangedFlag<L>,
        &IndirectComponent<Usage<T, BufferComponent>>,
    )>::query()
    .par_for_each(world, |(buffer_write, value, dirty_flag, buffer)| {
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
    });
}

// Write data to texture
#[legion::system]
#[read_component(Queue)]
#[read_component(Usage<T, TextureWriteComponent<L>>)]
#[read_component(L)]
#[read_component(ChangedFlag<L>)]
#[read_component(IndirectComponent<Usage<T, TextureDescriptorComponent>>)]
#[read_component(IndirectComponent<Usage<T, TextureComponent>>)]
#[read_component(Usage<T, TextureDescriptorComponent>)]
#[read_component(Usage<T, TextureComponent>)]
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
        &Usage<T, TextureWriteComponent<L>>,
        &L,
        &ChangedFlag<L>,
        &IndirectComponent<Usage<T, TextureDescriptorComponent>>,
        &IndirectComponent<Usage<T, TextureComponent>>,
    )>::query()
    .par_for_each(
        world,
        |(texture_write, texels, dirty_flag, texture_desc, texture)| {
            let texture_descriptor_component = world.get_indirect(texture_desc).unwrap();
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
                    texture_descriptor_component.read().size,
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

// Create textures and corresponding texture views for surfaces
#[legion::system]
#[read_component(WindowEventComponent)]
#[read_component(WindowEntityMap)]
#[read_component(SurfaceComponent)]
#[read_component(SurfaceTextureComponent)]
#[read_component(ChangedFlag<SurfaceTextureComponent>)]
#[read_component(RenderAttachmentTextureViewDescriptor)]
#[read_component(RenderAttachmentTextureView)]
pub fn surface_textures_views(world: &SubWorld) {
    use legion::IntoQuery;

    let window_event = <&WindowEventComponent>::query()
        .iter(&*world)
        .next()
        .unwrap();
    let window_event = window_event.read().0.expect("No window for current event");

    let window_entity_map = <&WindowEntityMap>::query().iter(&*world).next().unwrap();
    let window_entity_map = window_entity_map.read();

    let entity = window_entity_map
        .get(&window_event)
        .expect("Redraw requested for window without entity");

    // Create surface textures and views
    // These will be rendered to and presented during RedrawEventsCleared
    surface_texture_query(&world, entity);
    surface_texture_view_query(&world, entity);
}
