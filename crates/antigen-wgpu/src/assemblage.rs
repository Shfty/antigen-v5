use antigen_core::{
    AddComponentWithChangedFlag, AddIndirectComponent, ChangedFlag, LazyComponent, Usage,
};

use legion::{storage::Component, systems::CommandBuffer, Entity, World};
use wgpu::{
    util::BufferInitDescriptor, Adapter, Backends, BufferAddress, BufferDescriptor, Device,
    DeviceDescriptor, ImageCopyTextureBase, ImageDataLayout, Instance, Queue, SamplerDescriptor,
    ShaderModuleDescriptor, Surface, SurfaceConfiguration, TextureDescriptor,
    TextureViewDescriptor,
};

use std::path::Path;

use crate::{BindGroupComponent, BufferComponent, BufferDescriptorComponent, BufferInitDescriptorComponent, BufferWriteComponent, CommandBuffersComponent, ComputePipelineComponent, PipelineLayoutComponent, RenderAttachment, RenderBundleComponent, RenderPipelineComponent, SamplerComponent, SamplerDescriptorComponent, ShaderModuleComponent, ShaderModuleDescriptorComponent, SurfaceComponent, SurfaceConfigurationComponent, SurfaceTextureComponent, TextureComponent, TextureDescriptorComponent, TextureViewComponent, TextureViewDescriptorComponent, TextureWriteComponent};

/// Create an entity to hold an Instance, Adapter, Device and Queue
pub fn assemble_wgpu_entity(
    world: &mut World,
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
) -> Entity {
    world.push((instance, adapter, device, queue))
}

/// Retrieve WGPU settings from environment variables, and use them to create an entity
/// holding an Instance, Adapter, Device, and Queue
pub fn assemble_wgpu_entity_from_env(
    world: &mut World,
    device_desc: &DeviceDescriptor,
    compatible_surface: Option<&Surface>,
    trace_path: Option<&Path>,
) {
    let backend_bits = wgpu::util::backend_bits_from_env().unwrap_or(Backends::PRIMARY);

    let instance = Instance::new(backend_bits);
    println!("Created WGPU instance: {:#?}\n", instance);

    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance,
        backend_bits,
        compatible_surface,
    ))
    .expect("Failed to acquire WGPU adapter");

    let adapter_info = adapter.get_info();
    println!("Acquired WGPU adapter: {:#?}\n", adapter_info);

    let (device, queue) =
        pollster::block_on(adapter.request_device(device_desc, trace_path)).unwrap();

    println!("Acquired WGPU device: {:#?}\n", device);
    println!("Acquired WGPU queue: {:#?}\n", queue);

    assemble_wgpu_entity(world, instance, adapter, device, queue);
}

/// Extends an existing window entity with the means to render to a WGPU surface
#[legion::system]
pub fn assemble_window_surface(cmd: &mut CommandBuffer, #[state] (entity,): &(Entity,)) {
    cmd.add_component_with_changed_flag_clean(
        *entity,
        SurfaceConfigurationComponent::new(SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: 0,
            height: 0,
            present_mode: wgpu::PresentMode::Mailbox,
        }),
    );
    cmd.add_component(*entity, SurfaceComponent::new(LazyComponent::Pending));

    cmd.add_component_with_changed_flag_clean(*entity, SurfaceTextureComponent::new(None));

    cmd.add_component(
        *entity,
        Usage::<RenderAttachment, _>::new(TextureViewDescriptorComponent::new(Default::default())),
    );
    cmd.add_component(
        *entity,
        Usage::<RenderAttachment, _>::new(
            ChangedFlag::<TextureViewDescriptorComponent>::new_clean(),
        ),
    );
    cmd.add_component(
        *entity,
        Usage::<RenderAttachment, _>::new(TextureViewComponent::new(LazyComponent::Pending)),
    );
}

pub fn assemble_render_pipeline(cmd: &mut CommandBuffer, entity: Entity) {
    cmd.add_component(entity, RenderPipelineComponent::new(LazyComponent::Pending));
}

pub fn assemble_pipeline_layout(cmd: &mut CommandBuffer, entity: Entity) {
    cmd.add_component(entity, PipelineLayoutComponent::new(LazyComponent::Pending));
}

pub fn assemble_render_bundle(cmd: &mut CommandBuffer, entity: Entity) {
    cmd.add_component(entity, RenderBundleComponent::new(LazyComponent::Pending));
}

pub fn assemble_render_pipeline_usage<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(RenderPipelineComponent::new(LazyComponent::Pending)),
    );
}

pub fn assemble_compute_pipeline(cmd: &mut CommandBuffer, entity: Entity) {
    cmd.add_component(
        entity,
        ComputePipelineComponent::new(LazyComponent::Pending),
    );
}

pub fn assemble_bind_group(cmd: &mut CommandBuffer, entity: Entity) {
    cmd.add_component(entity, BindGroupComponent::new(LazyComponent::Pending));
}

pub fn assemble_bind_group_usage<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(BindGroupComponent::new(LazyComponent::Pending)),
    );
}

pub fn assemble_command_buffers(cmd: &mut CommandBuffer, entity: Entity) {
    cmd.add_component(entity, CommandBuffersComponent::new(Default::default()));
}

/// Adds an untagged shader to an entity
pub fn assemble_shader(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: ShaderModuleDescriptor<'static>,
) {
    cmd.add_component_with_changed_flag_clean(entity, ShaderModuleDescriptorComponent::new(desc));
    cmd.add_component(entity, ShaderModuleComponent::new(LazyComponent::Pending));
}

/// Adds a usage-tagged shader to an entity
pub fn assemble_shader_usage<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: ShaderModuleDescriptor<'static>,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(ShaderModuleDescriptorComponent::new(desc)),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(ChangedFlag::<ShaderModuleDescriptorComponent>::new_clean()),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(ShaderModuleComponent::new(LazyComponent::Pending)),
    );
}

/// Adds a usage-tagged buffer to an entity
pub fn assemble_buffer<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: BufferDescriptor<'static>,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(BufferDescriptorComponent::new(desc)),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(ChangedFlag::<BufferDescriptorComponent>::new_clean()),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(BufferComponent::new(LazyComponent::Pending)),
    );
}

/// Adds a usage-tagged buffer to an entity with initial data
pub fn assemble_buffer_init<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: BufferInitDescriptor<'static>,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(BufferInitDescriptorComponent::new(desc)),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(ChangedFlag::<BufferInitDescriptorComponent>::new_clean()),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(BufferComponent::new(LazyComponent::Pending)),
    );
}

/// Adds some usage-tagged data to be written to a buffer when its change flag is set
pub fn assemble_buffer_data<U, T>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    data: T,
    offset: BufferAddress,
) where
    U: Send + Sync + 'static,
    T: Component,
{
    cmd.add_component_with_changed_flag_dirty(entity, data);
    cmd.add_component(
        entity,
        Usage::<U, _>::new(BufferWriteComponent::<T>::new(offset)),
    );
    cmd.add_indirect_component_self::<Usage<U, BufferComponent>>(entity);
}

/// Adds a usage-tagged texture to an entity
pub fn assemble_texture<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: TextureDescriptor<'static>,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(TextureDescriptorComponent::new(desc)),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(ChangedFlag::<TextureDescriptorComponent>::new_clean()),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(TextureComponent::new(LazyComponent::Pending)),
    );
}

/// Adds some usage-tagged data to be written to a texture when its change flag is set
pub fn assemble_texture_data<U, T>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    data: T,
    image_copy_texture: ImageCopyTextureBase<()>,
    image_data_layout: ImageDataLayout,
) where
    T: Component,
    U: Send + Sync + 'static,
{
    cmd.add_component_with_changed_flag_dirty(entity, data);

    // Texture write
    cmd.add_component(
        entity,
        Usage::<U, _>::new(TextureWriteComponent::<T>::new(
            image_copy_texture,
            image_data_layout,
        )),
    );

    // Texture write indirect
    cmd.add_indirect_component_self::<Usage<U, TextureDescriptorComponent>>(entity);
    cmd.add_indirect_component_self::<Usage<U, TextureComponent>>(entity);
}

/// Adds a usage-tagged texture view to an entity
pub fn assemble_texture_view<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: TextureViewDescriptor<'static>,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(TextureViewDescriptorComponent::new(desc)),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(ChangedFlag::<TextureViewDescriptorComponent>::new_clean()),
    );

    cmd.add_component(
        entity,
        Usage::<U, _>::new(TextureViewComponent::new(LazyComponent::Pending)),
    );
}

/// Adds a usage-tagged sampler to an entity
pub fn assemble_sampler<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    desc: SamplerDescriptor<'static>,
) {
    cmd.add_component(
        entity,
        Usage::<U, _>::new(SamplerDescriptorComponent::new(desc)),
    );
    cmd.add_component(
        entity,
        Usage::<U, _>::new(ChangedFlag::<SamplerDescriptorComponent>::new_clean()),
    );
    cmd.add_component(
        entity,
        Usage::<U, _>::new(SamplerComponent::new(LazyComponent::Pending)),
    );
}
