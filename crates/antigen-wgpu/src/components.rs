use antigen_core::{
    impl_read_write_lock, LazyComponent, ReadWriteLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
    Usage,
};

use wgpu::{
    util::BufferInitDescriptor, BindGroup, Buffer, BufferDescriptor, ComputePipeline,
    ImageCopyTextureBase, ImageDataLayout, PipelineLayout, RenderBundle, RenderPipeline, Sampler,
    SamplerDescriptor, ShaderModule, ShaderModuleDescriptor, Surface, SurfaceConfiguration,
    Texture, TextureDescriptor, TextureViewDescriptor,
};

use std::marker::PhantomData;

// WGPU surface configuration
pub struct SurfaceConfigurationComponent(RwLock<SurfaceConfiguration>);

impl SurfaceConfigurationComponent {
    pub fn new(config: SurfaceConfiguration) -> SurfaceConfigurationComponent {
        SurfaceConfigurationComponent(RwLock::new(config))
    }
}

impl_read_write_lock!(SurfaceConfigurationComponent, 0, SurfaceConfiguration);

// WGPU surface
pub struct SurfaceComponent(RwLock<LazyComponent<Surface>>);

impl SurfaceComponent {
    pub fn pending() -> SurfaceComponent {
        SurfaceComponent(RwLock::new(LazyComponent::Pending))
    }

    pub fn ready(surface: Surface) -> SurfaceComponent {
        SurfaceComponent(RwLock::new(LazyComponent::Ready(surface)))
    }
}

impl_read_write_lock!(SurfaceComponent, 0, LazyComponent<Surface>);

// WGPU texture descriptor
#[derive(Debug)]
pub struct TextureDescriptorComponent<'a>(RwLock<TextureDescriptor<'a>>);

impl<'a> ReadWriteLock<TextureDescriptor<'a>> for TextureDescriptorComponent<'a> {
    fn read(&self) -> RwLockReadGuard<TextureDescriptor<'a>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<TextureDescriptor<'a>> {
        self.0.write()
    }
}

impl<'a> TextureDescriptorComponent<'a> {
    pub fn new(texture_descriptor: TextureDescriptor<'a>) -> Self {
        TextureDescriptorComponent(RwLock::new(texture_descriptor))
    }
}

// WGPU texture
pub struct TextureComponent(RwLock<LazyComponent<Texture>>);

impl_read_write_lock!(TextureComponent, 0, LazyComponent<Texture>);

impl TextureComponent {
    pub fn pending() -> Self {
        TextureComponent(RwLock::new(LazyComponent::Pending))
    }
}

// Render attachment usage flag for TextureComponent
pub enum RenderAttachment {}

pub type RenderAttachmentTextureViewDescriptor<'a> =
    Usage<RenderAttachment, TextureViewDescriptorComponent<'a>>;
pub type RenderAttachmentTexture = Usage<RenderAttachment, TextureComponent>;
pub type RenderAttachmentTextureView = Usage<RenderAttachment, TextureViewComponent>;

// MSAA frambuffer usage flag for TextureComponent
pub enum MsaaFramebuffer {}

pub type MsaaFramebufferTextureDescriptor<'a> =
    Usage<MsaaFramebuffer, TextureDescriptorComponent<'a>>;
pub type MsaaFramebufferTexture = Usage<MsaaFramebuffer, TextureComponent>;

pub type MsaaFramebufferTextureViewDescriptor<'a> =
    Usage<MsaaFramebuffer, TextureViewDescriptorComponent<'a>>;
pub type MsaaFramebufferTextureView = Usage<MsaaFramebuffer, TextureViewComponent>;

// WGPU surface texture
pub struct SurfaceTextureComponent(RwLock<Option<wgpu::SurfaceTexture>>);

impl_read_write_lock!(SurfaceTextureComponent, 0, Option<wgpu::SurfaceTexture>);

impl SurfaceTextureComponent {
    pub fn pending() -> Self {
        SurfaceTextureComponent(RwLock::new(None))
    }
}

// WPGU texture view descriptor
pub struct TextureViewDescriptorComponent<'a>(RwLock<TextureViewDescriptor<'a>>);

impl<'a> ReadWriteLock<TextureViewDescriptor<'a>> for TextureViewDescriptorComponent<'a> {
    fn read(&self) -> RwLockReadGuard<TextureViewDescriptor<'a>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<TextureViewDescriptor<'a>> {
        self.0.write()
    }
}

impl<'a> TextureViewDescriptorComponent<'a> {
    pub fn new(desc: TextureViewDescriptor<'a>) -> Self {
        TextureViewDescriptorComponent(RwLock::new(desc))
    }
}

// WGPU texture view
pub struct TextureViewComponent(RwLock<LazyComponent<wgpu::TextureView>>);

impl_read_write_lock!(TextureViewComponent, 0, LazyComponent<wgpu::TextureView>);

impl TextureViewComponent {
    pub fn pending() -> Self {
        TextureViewComponent(RwLock::new(LazyComponent::Pending))
    }
}

// WGPU sampler descriptor
pub struct SamplerDescriptorComponent<'a>(RwLock<SamplerDescriptor<'a>>);

impl<'a> ReadWriteLock<SamplerDescriptor<'a>> for SamplerDescriptorComponent<'a> {
    fn read(&self) -> RwLockReadGuard<SamplerDescriptor<'a>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<SamplerDescriptor<'a>> {
        self.0.write()
    }
}

impl<'a> SamplerDescriptorComponent<'a> {
    pub fn new(desc: SamplerDescriptor<'a>) -> Self {
        SamplerDescriptorComponent(RwLock::new(desc))
    }
}

// WGPU sampler
pub struct SamplerComponent(RwLock<LazyComponent<wgpu::Sampler>>);

impl_read_write_lock!(SamplerComponent, 0, LazyComponent<Sampler>);

impl<'a> SamplerComponent {
    pub fn pending() -> Self {
        SamplerComponent(RwLock::new(LazyComponent::Pending))
    }
}

// WGPU pipeline layout
pub struct PipelineLayoutComponent(RwLock<LazyComponent<PipelineLayout>>);

impl_read_write_lock!(PipelineLayoutComponent, 0, LazyComponent<PipelineLayout>);

impl PipelineLayoutComponent {
    pub fn pending() -> Self {
        PipelineLayoutComponent(RwLock::new(LazyComponent::Pending))
    }
}

// WGPU render pipeline
pub struct RenderPipelineComponent(RwLock<LazyComponent<RenderPipeline>>);

impl_read_write_lock!(RenderPipelineComponent, 0, LazyComponent<RenderPipeline>);

impl RenderPipelineComponent {
    pub fn pending() -> Self {
        RenderPipelineComponent(RwLock::new(LazyComponent::Pending))
    }
}

// WGPU compute pipeline
pub struct ComputePipelineComponent(RwLock<LazyComponent<ComputePipeline>>);

impl_read_write_lock!(ComputePipelineComponent, 0, LazyComponent<ComputePipeline>);

impl ComputePipelineComponent {
    pub fn pending() -> Self {
        ComputePipelineComponent(RwLock::new(LazyComponent::Pending))
    }
}

// WGPU render bundle
pub struct RenderBundleComponent(RwLock<LazyComponent<RenderBundle>>);

impl_read_write_lock!(RenderBundleComponent, 0, LazyComponent<RenderBundle>);

impl RenderBundleComponent {
    pub fn pending() -> Self {
        RenderBundleComponent(RwLock::new(LazyComponent::Pending))
    }
}

// WGPU bind group
pub struct BindGroupComponent {
    bind_group: RwLock<LazyComponent<BindGroup>>,
}

impl ReadWriteLock<LazyComponent<BindGroup>> for BindGroupComponent {
    fn read(&self) -> RwLockReadGuard<LazyComponent<BindGroup>> {
        self.bind_group.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<BindGroup>> {
        self.bind_group.write()
    }
}

impl BindGroupComponent {
    pub fn pending() -> Self {
        BindGroupComponent {
            bind_group: RwLock::new(LazyComponent::Pending),
        }
    }
}

// WGPU command buffers
pub struct CommandBuffersComponent(RwLock<Vec<wgpu::CommandBuffer>>);

impl_read_write_lock!(CommandBuffersComponent, 0, Vec<wgpu::CommandBuffer>);

impl CommandBuffersComponent {
    pub fn new() -> Self {
        CommandBuffersComponent(Default::default())
    }
}

// WGPU buffer descriptor
#[derive(Debug)]
pub struct BufferDescriptorComponent<'a>(RwLock<BufferDescriptor<'a>>);

impl<'a> ReadWriteLock<BufferDescriptor<'a>> for BufferDescriptorComponent<'a> {
    fn read(&self) -> RwLockReadGuard<BufferDescriptor<'a>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<BufferDescriptor<'a>> {
        self.0.write()
    }
}

impl<'a> BufferDescriptorComponent<'a> {
    pub fn new(desc: BufferDescriptor<'a>) -> Self {
        BufferDescriptorComponent(RwLock::new(desc))
    }
}

// WGPU buffer init descriptor
#[derive(Debug)]
pub struct BufferInitDescriptorComponent<'a>(RwLock<BufferInitDescriptor<'a>>);

impl<'a> ReadWriteLock<BufferInitDescriptor<'a>> for BufferInitDescriptorComponent<'a> {
    fn read(&self) -> RwLockReadGuard<BufferInitDescriptor<'a>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<BufferInitDescriptor<'a>> {
        self.0.write()
    }
}

impl<'a> BufferInitDescriptorComponent<'a> {
    pub fn new(desc: BufferInitDescriptor<'a>) -> Self {
        BufferInitDescriptorComponent(RwLock::new(desc))
    }
}

// WGPU buffer
#[derive(Debug)]
pub struct BufferComponent(RwLock<LazyComponent<Buffer>>);

impl_read_write_lock!(BufferComponent, 0, LazyComponent<Buffer>);

impl BufferComponent {
    pub fn pending() -> Self {
        BufferComponent(RwLock::new(LazyComponent::Pending))
    }
}

// Buffer write operation
pub struct BufferWriteComponent<T> {
    offset: RwLock<wgpu::BufferAddress>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<wgpu::BufferAddress> for BufferWriteComponent<T> {
    fn read(&self) -> RwLockReadGuard<wgpu::BufferAddress> {
        self.offset.read()
    }

    fn write(&self) -> RwLockWriteGuard<wgpu::BufferAddress> {
        self.offset.write()
    }
}

impl<T> BufferWriteComponent<T> {
    pub fn new(offset: wgpu::BufferAddress) -> Self {
        BufferWriteComponent {
            offset: RwLock::new(offset),
            _phantom: Default::default(),
        }
    }
}

// Texture write operation
pub struct TextureWriteComponent<T> {
    image_copy_texture: RwLock<ImageCopyTextureBase<()>>,
    image_data_layout: RwLock<wgpu::ImageDataLayout>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<wgpu::ImageDataLayout> for TextureWriteComponent<T> {
    fn read(&self) -> RwLockReadGuard<ImageDataLayout> {
        self.image_data_layout.read()
    }

    fn write(&self) -> RwLockWriteGuard<ImageDataLayout> {
        self.image_data_layout.write()
    }
}

impl<T> ReadWriteLock<ImageCopyTextureBase<()>> for TextureWriteComponent<T> {
    fn read(&self) -> RwLockReadGuard<ImageCopyTextureBase<()>> {
        self.image_copy_texture.read()
    }

    fn write(&self) -> RwLockWriteGuard<ImageCopyTextureBase<()>> {
        self.image_copy_texture.write()
    }
}

impl<T> TextureWriteComponent<T> {
    pub fn new(
        image_copy_texture: ImageCopyTextureBase<()>,
        image_data_layout: ImageDataLayout,
    ) -> Self {
        TextureWriteComponent {
            image_copy_texture: RwLock::new(image_copy_texture),
            image_data_layout: RwLock::new(image_data_layout),
            _phantom: Default::default(),
        }
    }
}

// WGPU shader module descriptor
pub struct ShaderModuleDescriptorComponent<'a>(RwLock<ShaderModuleDescriptor<'a>>);

impl<'a> ReadWriteLock<ShaderModuleDescriptor<'a>> for ShaderModuleDescriptorComponent<'a> {
    fn read(&self) -> RwLockReadGuard<ShaderModuleDescriptor<'a>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<ShaderModuleDescriptor<'a>> {
        self.0.write()
    }
}

impl<'a> ShaderModuleDescriptorComponent<'a> {
    pub fn new(descriptor: ShaderModuleDescriptor<'a>) -> Self {
        ShaderModuleDescriptorComponent(RwLock::new(descriptor))
    }
}

// WGPU shader module
pub struct ShaderModuleComponent(RwLock<LazyComponent<ShaderModule>>);

impl_read_write_lock!(ShaderModuleComponent, 0, LazyComponent<ShaderModule>);

impl ShaderModuleComponent {
    pub fn pending() -> Self {
        ShaderModuleComponent(RwLock::new(LazyComponent::Pending))
    }
}

// Texture texels
pub struct Texels<T>(RwLock<T>);

impl<T> ReadWriteLock<T> for Texels<T> {
    fn read(&self) -> RwLockReadGuard<T> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<T> {
        self.0.write()
    }
}

impl<T> Texels<T> {
    pub fn new(texels: T) -> Self {
        Texels(RwLock::new(texels))
    }
}

// Mesh vertices
pub struct MeshVertices<T>(RwLock<Vec<T>>);

impl<T> ReadWriteLock<Vec<T>> for MeshVertices<T> {
    fn read(&self) -> RwLockReadGuard<Vec<T>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<Vec<T>> {
        self.0.write()
    }
}

impl<T> MeshVertices<T> {
    pub fn new(vertices: Vec<T>) -> Self {
        MeshVertices(RwLock::new(vertices))
    }
}

// Mesh UVs
pub struct MeshUvs<T>(RwLock<Vec<T>>);

impl<T> ReadWriteLock<Vec<T>> for MeshUvs<T> {
    fn read(&self) -> RwLockReadGuard<Vec<T>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<Vec<T>> {
        self.0.write()
    }
}

impl<T> MeshUvs<T> {
    pub fn new(uvs: Vec<T>) -> Self {
        MeshUvs(RwLock::new(uvs))
    }
}

// Mesh indices
pub struct MeshIndices<T>(RwLock<Vec<T>>);

impl<T> ReadWriteLock<Vec<T>> for MeshIndices<T> {
    fn read(&self) -> RwLockReadGuard<Vec<T>> {
        self.0.read()
    }

    fn write(&self) -> RwLockWriteGuard<Vec<T>> {
        self.0.write()
    }
}

impl<T> MeshIndices<T> {
    pub fn new(indices: Vec<T>) -> Self {
        MeshIndices(RwLock::new(indices))
    }
}
