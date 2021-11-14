use super::{
    BufferWriteComponent, CommandBuffersComponent, RenderAttachmentTextureViewDescriptor,
    SurfaceComponent, SurfaceTextureComponent, TextureDescriptorComponent, TextureViewComponent,
    TextureViewDescriptorComponent, TextureWriteComponent, ToBytes,
};
use crate::{
    BufferComponent, BufferDescriptorComponent, RenderAttachmentTextureView, SamplerComponent,
    SamplerDescriptorComponent, ShaderModuleComponent, ShaderModuleDescriptorComponent,
    SurfaceConfigurationComponent, TextureComponent,
};

use antigen_core::{
    ChangedFlag, GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock, Usage,
};
use antigen_winit::{WindowComponent, WindowEntityMap, WindowEventComponent, WindowSizeComponent};

use legion::{world::SubWorld, IntoQuery};
use wgpu::{
    Adapter, Device, ImageCopyTextureBase, ImageDataLayout, Instance, Queue, Surface,
};

// Initialize pending surfaces that share an entity with a window
#[legion::system(for_each)]
#[read_component(wgpu::Device)]
#[read_component(wgpu::Adapter)]
#[read_component(wgpu::Instance)]
pub fn create_window_surfaces(
    world: &SubWorld,
    window_component: &WindowComponent,
    surface_configuration_component: &SurfaceConfigurationComponent,
    surface_component: &SurfaceComponent,
) {
    if let LazyComponent::Ready(window) = &*window_component.read() {
        let adapter = <&Adapter>::query().iter(world).next().unwrap();
        let device = <&Device>::query().iter(world).next().unwrap();

        if ReadWriteLock::<LazyComponent<Surface>>::read(surface_component).is_pending() {
            let instance = <&Instance>::query().iter(world).next().unwrap();
            let surface = unsafe { instance.create_surface(window) };
            let mut config = surface_configuration_component.write();

            let window_size = window.inner_size();
            config.width = window_size.width;
            config.height = window_size.height;

            config.format = surface
                .get_preferred_format(adapter)
                .expect("Surface is incompatible with adapter");

            surface.configure(device, &config);

            ReadWriteLock::<LazyComponent<Surface>>::write(surface_component).set_ready(surface);
        }
    }
}

// Initialize pending surfaces that share an entity with a window
#[legion::system(for_each)]
#[read_component(wgpu::Device)]
#[read_component(wgpu::Adapter)]
#[read_component(wgpu::Instance)]
pub fn reconfigure_surfaces(
    world: &SubWorld,
    surface_config: &SurfaceConfigurationComponent,
    surface_config_changed: &ChangedFlag<SurfaceConfigurationComponent>,
    surface_component: &SurfaceComponent,
) {
    let device = <&Device>::query().iter(world).next().unwrap();

    let surface_read = surface_component.read();
    let surface = if let LazyComponent::Ready(surface) = &*surface_read {
        surface
    } else {
        return;
    };

    if !surface_config_changed.get() {
        return
    }

    let config = surface_config.read();
    if config.width > 0 && config.height > 0 {
        surface.configure(device, &config);
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

    let surface = surface.read();
    let surface = if let LazyComponent::Ready(surface) = &*surface
    {
        surface
    }
    else {
        return;
    };

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
    window_size_changed: &ChangedFlag<WindowSizeComponent>,
    surface_configuration: &SurfaceConfigurationComponent,
    surface_configuration_changed: &ChangedFlag<SurfaceConfigurationComponent>,
) {
    if window_size_changed.get() {
        let window_size = *window_size.read();
        let mut surface_configuration = surface_configuration.write();
        surface_configuration.width = window_size.width;
        surface_configuration.height = window_size.height;
        surface_configuration_changed.set(true);
    }
}

#[legion::system(par_for_each)]
pub fn reset_surface_config_changed_flag(
    surface_config_changed: &ChangedFlag<SurfaceConfigurationComponent>,
) {
    if surface_config_changed.get() {
        surface_config_changed.set(false);
    }
}

#[legion::system(par_for_each)]
pub fn reset_texture_descriptor_changed_flag(
    texture_desc_changed: &ChangedFlag<TextureDescriptorComponent>,
) {
    if texture_desc_changed.get() {
        texture_desc_changed.set(false);
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

/// Create pending untagged shader modules, recreating them if a ChangedFlag is set
#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_shader_modules(
    world: &SubWorld,
    shader_module_desc: &ShaderModuleDescriptorComponent,
    shader_module: &ShaderModuleComponent,
    shader_module_desc_changed: &ChangedFlag<ShaderModuleDescriptorComponent>,
) {
    if shader_module.read().is_pending()
        || shader_module_desc_changed.get()
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        shader_module
            .write()
            .set_ready(device.create_shader_module(&shader_module_desc.read()));
        println!("Created shader module");
    }
}

/// Create pending usage-tagged shader modules, recreating them if a ChangedFlag is set
#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_shader_modules_usage<T: Send + Sync + 'static>(
    world: &SubWorld,
    shader_module_desc: &Usage<T, ShaderModuleDescriptorComponent>,
    shader_module: &Usage<T, ShaderModuleComponent>,
    shader_module_desc_changed: &Usage<T, ChangedFlag<ShaderModuleDescriptorComponent>>,
) {
    if shader_module.read().is_pending()
        || shader_module_desc_changed.get()
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        shader_module
            .write()
            .set_ready(device.create_shader_module(&shader_module_desc.read()));
        println!("Created {} shader module", std::any::type_name::<T>());
    }
}

/// Create pending usage-tagged buffers, recreating them if a ChangedFlag is set
#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_buffers<T: Send + Sync + 'static>(
    world: &SubWorld,
    buffer_desc: &Usage<T, BufferDescriptorComponent>,
    buffer: &Usage<T, BufferComponent>,
    buffer_desc_changed: &Usage<T, ChangedFlag<BufferDescriptorComponent>>,
) {
    if buffer.read().is_pending()
        || buffer_desc_changed.get()
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        println!("Created {} buffer", std::any::type_name::<T>());
        buffer
            .write()
            .set_ready(device.create_buffer(&buffer_desc.read()));
    }
}

/// Create pending usage-tagged textures, recreating them if a ChangedFlag is set
#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_textures<T: Send + Sync + 'static>(
    world: &SubWorld,
    texture_descriptor: &Usage<T, TextureDescriptorComponent>,
    texture: &Usage<T, TextureComponent>,
    texture_descriptor_changed: &Usage<T, ChangedFlag<TextureDescriptorComponent>>,
) {
    if texture.read().is_pending()
        || texture_descriptor_changed.get()
    {
        let device = <&Device>::query().iter(world).next().unwrap();
        texture
            .write()
            .set_ready(device.create_texture(&*texture_descriptor.read()));
    }
}

/// Create pending usage-tagged texture views, recreating them if a ChangedFlag is set
#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_texture_views<T: Send + Sync + 'static>(
    texture: &Usage<T, TextureComponent>,
    texture_view_desc: &Usage<T, TextureViewDescriptorComponent>,
    texture_view: &Usage<T, TextureViewComponent>,
    texture_view_desc_changed: &Usage<T, ChangedFlag<TextureViewDescriptorComponent>>,
) {
    if !texture_view.read().is_pending()
        && !texture_view_desc_changed.get()
    {
        return;
    }

    let texture = texture.read();
    let texture = if let LazyComponent::Ready(texture) = &*texture {
        texture
    } else {
        return;
    };

    texture_view
        .write()
        .set_ready(texture.create_view(&texture_view_desc.read()));
}

/// Create pending usage-tagged samplers, recreating them if a ChangedFlag is set
#[legion::system(par_for_each)]
#[read_component(Device)]
pub fn create_samplers<T: Send + Sync + 'static>(
    world: &SubWorld,
    sampler_desc: &Usage<T, SamplerDescriptorComponent>,
    sampler: &Usage<T, SamplerComponent>,
    sampler_desc_changed: &Usage<T, ChangedFlag<SamplerDescriptorComponent>>,
) {
    if sampler.read().is_pending()
        || sampler_desc_changed.get()
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
