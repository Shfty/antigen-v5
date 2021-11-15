use antigen_core::{
    impl_read_write_lock, LazyComponent, ReadWriteLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
    Usage,
};

use wgpu::{
    util::BufferInitDescriptor, BindGroup, Buffer, BufferDescriptor, CommandBuffer,
    ComputePipeline, ImageCopyTextureBase, ImageDataLayout, PipelineLayout, RenderBundle,
    RenderPipeline, Sampler, SamplerDescriptor, ShaderModule, ShaderModuleDescriptor, Surface,
    SurfaceConfiguration, SurfaceTexture, Texture, TextureDescriptor, TextureView,
    TextureViewDescriptor,
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
pub type SurfaceComponent = RwLock<LazyComponent<Surface>>;

// WGPU texture descriptor
pub type TextureDescriptorComponent<'a> = RwLock<TextureDescriptor<'a>>;

// WGPU texture
pub type TextureComponent = RwLock<LazyComponent<Texture>>;

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
pub type SurfaceTextureComponent = RwLock<Option<SurfaceTexture>>;

// WPGU texture view descriptor
pub type TextureViewDescriptorComponent<'a> = RwLock<TextureViewDescriptor<'a>>;

// WGPU texture view
pub type TextureViewComponent = RwLock<LazyComponent<TextureView>>;

// WGPU sampler descriptor
pub type SamplerDescriptorComponent<'a> = RwLock<SamplerDescriptor<'a>>;

// WGPU sampler
pub type SamplerComponent = RwLock<LazyComponent<Sampler>>;

// WGPU pipeline layout
pub type PipelineLayoutComponent = RwLock<LazyComponent<PipelineLayout>>;

// WGPU render pipeline
pub type RenderPipelineComponent = RwLock<LazyComponent<RenderPipeline>>;

// WGPU compute pipeline
pub type ComputePipelineComponent = RwLock<LazyComponent<ComputePipeline>>;

// WGPU render bundle
pub type RenderBundleComponent = RwLock<LazyComponent<RenderBundle>>;

// WGPU bind group
pub type BindGroupComponent = RwLock<LazyComponent<BindGroup>>;

// WGPU command buffers
pub type CommandBuffersComponent = RwLock<Vec<CommandBuffer>>;

// WGPU buffer descriptor
pub type BufferDescriptorComponent<'a> = RwLock<BufferDescriptor<'a>>;

// WGPU buffer init descriptor
pub type BufferInitDescriptorComponent<'a> = RwLock<BufferInitDescriptor<'a>>;

// WGPU buffer
pub type BufferComponent = RwLock<LazyComponent<Buffer>>;

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
pub type ShaderModuleDescriptorComponent<'a> = RwLock<ShaderModuleDescriptor<'a>>;

// WGPU shader module
pub type ShaderModuleComponent = RwLock<LazyComponent<ShaderModule>>;

// Texture texels usage tag
pub enum Texels {}

// Mesh vertices usage tag
pub enum MeshVertices {}

// Mesh UVs usage tag
pub enum MeshUvs {}

// Mesh indices usage tag
pub enum MeshIndices {}
