use antigen_core::{
    Changed, LazyComponent, ReadWriteLock, RwLock, RwLockReadGuard, RwLockWriteGuard, Usage,
};

use wgpu::{
    util::BufferInitDescriptor, BindGroup, BindGroupLayout, Buffer, BufferAddress,
    BufferDescriptor, CommandBuffer, ComputePipeline, ImageCopyTextureBase, ImageDataLayout,
    PipelineLayout, RenderBundle, RenderPipeline, Sampler, SamplerDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderModuleDescriptorSpirV, Surface, SurfaceConfiguration,
    SurfaceTexture, Texture, TextureDescriptor, TextureView, TextureViewDescriptor,
};

use std::marker::PhantomData;

// WGPU surface configuration
pub type SurfaceConfigurationComponent = Changed<RwLock<SurfaceConfiguration>>;

// WGPU surface
pub type SurfaceComponent = RwLock<LazyComponent<Surface>>;

// WGPU texture descriptor
pub type TextureDescriptorComponent<'a> = Changed<RwLock<TextureDescriptor<'a>>>;

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
pub type SurfaceTextureComponent = Changed<RwLock<Option<SurfaceTexture>>>;

// WPGU texture view descriptor
pub type TextureViewDescriptorComponent<'a> = Changed<RwLock<TextureViewDescriptor<'a>>>;

// WGPU texture view
pub type TextureViewComponent = RwLock<LazyComponent<TextureView>>;

// WGPU sampler descriptor
pub type SamplerDescriptorComponent<'a> = Changed<RwLock<SamplerDescriptor<'a>>>;

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

// WGPU bind group layout
pub type BindGroupLayoutComponent = RwLock<LazyComponent<BindGroupLayout>>;

// WGPU bind group
pub type BindGroupComponent = RwLock<LazyComponent<BindGroup>>;

// WGPU command buffers
pub type CommandBuffersComponent = RwLock<Vec<CommandBuffer>>;

// WGPU buffer descriptor
pub type BufferDescriptorComponent<'a> = Changed<RwLock<BufferDescriptor<'a>>>;

// WGPU buffer init descriptor
pub type BufferInitDescriptorComponent<'a> = Changed<RwLock<BufferInitDescriptor<'a>>>;

// WGPU buffer
pub type BufferComponent = RwLock<LazyComponent<Buffer>>;

// Buffer write operation
pub struct BufferWriteComponent<T> {
    offset: RwLock<BufferAddress>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<BufferAddress> for BufferWriteComponent<T> {
    fn read(&self) -> RwLockReadGuard<BufferAddress> {
        self.offset.read()
    }

    fn write(&self) -> RwLockWriteGuard<BufferAddress> {
        self.offset.write()
    }
}

impl<T> BufferWriteComponent<T> {
    pub fn new(offset: BufferAddress) -> Self {
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
pub type ShaderModuleDescriptorComponent<'a> = Changed<RwLock<ShaderModuleDescriptor<'a>>>;

// WGPU shader module descriptor
pub type ShaderModuleDescriptorSpirVComponent<'a> =
    Changed<RwLock<ShaderModuleDescriptorSpirV<'a>>>;

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
