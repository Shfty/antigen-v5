use antigen_core::{AddIndirectComponent, AsUsage, Changed, LazyComponent, Usage};

use legion::{storage::Component, Entity, World};
use wgpu::{
    util::BufferInitDescriptor, Adapter, Backends, BufferAddress, BufferDescriptor, Device,
    DeviceDescriptor, ImageCopyTextureBase, ImageDataLayout, Instance, Queue, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderModuleDescriptorSpirV, Surface, SurfaceConfiguration,
    TextureDescriptor, TextureViewDescriptor,
};

use std::path::Path;

use crate::{
    BindGroupComponent, BindGroupLayoutComponent, BufferComponent, BufferDescriptorComponent,
    BufferInitDescriptorComponent, BufferWriteComponent, CommandBuffersComponent,
    ComputePipelineComponent, PipelineLayoutComponent, RenderAttachment, RenderBundleComponent,
    RenderPipelineComponent, SamplerComponent, SamplerDescriptorComponent, ShaderModuleComponent,
    ShaderModuleDescriptorComponent, ShaderModuleDescriptorSpirVComponent, SurfaceComponent,
    SurfaceConfigurationComponent, SurfaceTextureComponent, TextureComponent,
    TextureDescriptorComponent, TextureViewComponent, TextureViewDescriptorComponent,
    TextureWriteComponent,
};

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

/// [`CommandBuffer`] extension trait containing WGPU-specific assembly methods
pub trait AssembleWgpu {
    /// Extends an existing window entity with the means to render to a WGPU surface
    fn assemble_wgpu_window_surface(self, entity: Entity);

    /// Adds a render pipeline to an entity
    fn assemble_wgpu_render_pipeline(self, entity: Entity);

    /// Adds a pipeline layout to an entity
    fn assemble_wgpu_pipeline_layout(self, entity: Entity);

    /// Adds a render bundle to and entity
    fn assemble_wgpu_render_bundle(self, entity: Entity);

    /// Adds a usage-tagged render pipeline to an entity
    fn assemble_wgpu_render_pipeline_with_usage<U: Send + Sync + 'static>(self, entity: Entity);

    /// Adds a compute pipeline to an entity
    fn assemble_wgpu_compute_pipeline(self, entity: Entity);

    /// Adds a bind group layout to an entity
    fn assemble_wgpu_bind_group_layout(self, entity: Entity);

    /// Adds a usage-tagged bind group to an entity
    fn assemble_wgpu_bind_group_layout_with_usage<U: Send + Sync + 'static>(self, entity: Entity);

    /// Adds a bind group to an entity
    fn assemble_wgpu_bind_group(self, entity: Entity);

    /// Adds a usage-tagged bind group to an entity
    fn assemble_wgpu_bind_group_with_usage<U: Send + Sync + 'static>(self, entity: Entity);

    /// Adds command buffer storage to an entity
    fn assemble_wgpu_command_buffers(self, entity: Entity);

    /// Adds an untagged shader to an entity
    fn assemble_wgpu_shader(self, entity: Entity, desc: ShaderModuleDescriptor<'static>);

    /// Adds a usage-tagged shader to an entity
    fn assemble_wgpu_shader_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: ShaderModuleDescriptor<'static>,
    );

    /// Adds an untagged shader to an entity
    fn assemble_wgpu_shader_spirv(self, entity: Entity, desc: ShaderModuleDescriptorSpirV<'static>);

    /// Adds a usage-tagged shader to an entity
    fn assemble_wgpu_shader_spirv_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: ShaderModuleDescriptorSpirV<'static>,
    );

    /// Adds a usage-tagged buffer to an entity
    fn assemble_wgpu_buffer_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: BufferDescriptor<'static>,
    );

    /// Adds a usage-tagged buffer to an entity with initial data
    fn assemble_wgpu_buffer_init_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: BufferInitDescriptor<'static>,
    );

    /// Adds some usage-tagged data to be written to a buffer when its change flag is set
    fn assemble_wgpu_buffer_data_with_usage<U, T>(
        self,
        entity: Entity,
        data: T,
        offset: BufferAddress,
    ) where
        U: Send + Sync + 'static,
        T: Component;

    /// Adds a usage-tagged texture to an entity
    fn assemble_wgpu_texture_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: TextureDescriptor<'static>,
    );

    /// Adds some usage-tagged data to be written to a texture when its change flag is set
    fn assemble_wgpu_texture_data_with_usage<U, T>(
        self,
        entity: Entity,
        data: T,
        image_copy_texture: ImageCopyTextureBase<()>,
        image_data_layout: ImageDataLayout,
    ) where
        T: Component,
        U: Send + Sync + 'static;

    /// Adds a usage-tagged texture view to an entity
    fn assemble_wgpu_texture_view_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        target: Entity,
        desc: TextureViewDescriptor<'static>,
    );

    /// Adds a usage-tagged sampler to an entity
    fn assemble_wgpu_sampler(self, entity: Entity, desc: SamplerDescriptor<'static>);

    /// Adds a usage-tagged sampler to an entity
    fn assemble_wgpu_sampler_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: SamplerDescriptor<'static>,
    );
}

impl AssembleWgpu for &mut legion::systems::CommandBuffer {
    fn assemble_wgpu_window_surface(self, entity: Entity) {
        self.add_component(
            entity,
            Changed::new(
                SurfaceConfigurationComponent::new(SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: wgpu::TextureFormat::Bgra8Unorm,
                    width: 0,
                    height: 0,
                    present_mode: wgpu::PresentMode::Mailbox,
                }),
                false,
            ),
        );
        self.add_component(entity, SurfaceComponent::new(LazyComponent::Pending));

        self.add_component(entity, SurfaceTextureComponent::new(None.into(), false));

        self.add_component(
            entity,
            RenderAttachment::as_usage(TextureViewDescriptorComponent::new(
                Default::default(),
                false,
            )),
        );
        self.add_component(
            entity,
            RenderAttachment::as_usage(TextureViewComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_render_pipeline(self, entity: Entity) {
        self.add_component(entity, RenderPipelineComponent::new(LazyComponent::Pending));
    }

    fn assemble_wgpu_pipeline_layout(self, entity: Entity) {
        self.add_component(entity, PipelineLayoutComponent::new(LazyComponent::Pending));
    }

    fn assemble_wgpu_render_bundle(self, entity: Entity) {
        self.add_component(entity, RenderBundleComponent::new(LazyComponent::Pending));
    }

    fn assemble_wgpu_render_pipeline_with_usage<U: Send + Sync + 'static>(self, entity: Entity) {
        self.add_component(
            entity,
            U::as_usage(RenderPipelineComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_compute_pipeline(self, entity: Entity) {
        self.add_component(
            entity,
            ComputePipelineComponent::new(LazyComponent::Pending),
        );
    }

    fn assemble_wgpu_bind_group_layout(self, entity: Entity) {
        self.add_component(
            entity,
            BindGroupLayoutComponent::new(LazyComponent::Pending),
        );
    }

    fn assemble_wgpu_bind_group_layout_with_usage<U: Send + Sync + 'static>(self, entity: Entity) {
        self.add_component(
            entity,
            U::as_usage(BindGroupLayoutComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_bind_group(self, entity: Entity) {
        self.add_component(entity, BindGroupComponent::new(LazyComponent::Pending));
    }

    fn assemble_wgpu_bind_group_with_usage<U: Send + Sync + 'static>(self, entity: Entity) {
        self.add_component(
            entity,
            U::as_usage(BindGroupComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_command_buffers(self, entity: Entity) {
        self.add_component(entity, CommandBuffersComponent::new(Default::default()));
    }

    fn assemble_wgpu_shader(self, entity: Entity, desc: ShaderModuleDescriptor<'static>) {
        self.add_component(
            entity,
            ShaderModuleDescriptorComponent::new(desc.into(), false),
        );
        self.add_component(entity, ShaderModuleComponent::new(LazyComponent::Pending));
    }

    fn assemble_wgpu_shader_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: ShaderModuleDescriptor<'static>,
    ) {
        self.add_component(
            entity,
            U::as_usage(ShaderModuleDescriptorComponent::new(desc.into(), false)),
        );

        self.add_component(
            entity,
            U::as_usage(ShaderModuleComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_shader_spirv(
        self,
        entity: Entity,
        desc: ShaderModuleDescriptorSpirV<'static>,
    ) {
        self.add_component(
            entity,
            ShaderModuleDescriptorSpirVComponent::new(desc.into(), false),
        );
        self.add_component(entity, ShaderModuleComponent::new(LazyComponent::Pending));
    }

    fn assemble_wgpu_shader_spirv_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: ShaderModuleDescriptorSpirV<'static>,
    ) {
        self.add_component(
            entity,
            U::as_usage(ShaderModuleDescriptorSpirVComponent::new(
                desc.into(),
                false,
            )),
        );

        self.add_component(
            entity,
            U::as_usage(ShaderModuleComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_buffer_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: BufferDescriptor<'static>,
    ) {
        self.add_component(
            entity,
            U::as_usage(BufferDescriptorComponent::new(desc.into(), false)),
        );

        self.add_component(
            entity,
            U::as_usage(BufferComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_buffer_init_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: BufferInitDescriptor<'static>,
    ) {
        self.add_component(
            entity,
            U::as_usage(BufferInitDescriptorComponent::new(desc.into(), false)),
        );

        self.add_component(
            entity,
            U::as_usage(BufferComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_buffer_data_with_usage<U, T>(
        self,
        entity: Entity,
        data: T,
        offset: BufferAddress,
    ) where
        U: Send + Sync + 'static,
        T: Component,
    {
        self.add_component(entity, Changed::new(data, true));
        self.add_component(entity, U::as_usage(BufferWriteComponent::<T>::new(offset)));
        self.add_indirect_component_self::<Usage<U, BufferComponent>>(entity);
    }

    fn assemble_wgpu_texture_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: TextureDescriptor<'static>,
    ) {
        self.add_component(
            entity,
            U::as_usage(TextureDescriptorComponent::new(desc.into(), false)),
        );

        self.add_component(
            entity,
            U::as_usage(TextureComponent::new(LazyComponent::Pending)),
        );
    }

    fn assemble_wgpu_texture_data_with_usage<U, T>(
        self,
        entity: Entity,
        data: T,
        image_copy_texture: ImageCopyTextureBase<()>,
        image_data_layout: ImageDataLayout,
    ) where
        T: Component,
        U: Send + Sync + 'static,
    {
        self.add_component(entity, Changed::new(data, true));

        // Texture write
        self.add_component(
            entity,
            U::as_usage(TextureWriteComponent::<T>::new(
                image_copy_texture,
                image_data_layout,
            )),
        );

        // Texture write indirect
        self.add_indirect_component_self::<Usage<U, TextureDescriptorComponent>>(entity);
        self.add_indirect_component_self::<Usage<U, TextureComponent>>(entity);
    }

    fn assemble_wgpu_texture_view_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        target: Entity,
        desc: TextureViewDescriptor<'static>,
    ) {
        self.add_component(
            entity,
            U::as_usage(TextureViewDescriptorComponent::new(desc.into(), false)),
        );

        self.add_component(
            entity,
            U::as_usage(TextureViewComponent::new(LazyComponent::Pending)),
        );

        self.add_indirect_component::<Usage<U, TextureComponent>>(entity, target);
    }

    fn assemble_wgpu_sampler(self, entity: Entity, desc: SamplerDescriptor<'static>) {
        self.add_component(entity, SamplerDescriptorComponent::new(desc.into(), false));
        self.add_component(entity, SamplerComponent::new(LazyComponent::Pending));
    }

    fn assemble_wgpu_sampler_with_usage<U: Send + Sync + 'static>(
        self,
        entity: Entity,
        desc: SamplerDescriptor<'static>,
    ) {
        self.add_component(
            entity,
            U::as_usage(SamplerDescriptorComponent::new(desc.into(), false)),
        );
        self.add_component(
            entity,
            U::as_usage(SamplerComponent::new(LazyComponent::Pending)),
        );
    }
}
