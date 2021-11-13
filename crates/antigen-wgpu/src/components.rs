use antigen_core::{
    impl_read_write_lock, LazyComponent, ReadWriteLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
    SizeComponent, Usage,
};

use wgpu::{
    BindGroup, BufferDescriptor, ComputePipeline, ImageCopyTextureBase, ImageDataLayout,
    PipelineLayout, RenderBundle, RenderPipeline, ShaderModule, ShaderModuleDescriptor, Surface,
    SurfaceConfiguration, Texture, TextureDescriptor,
};

use std::marker::PhantomData;

// WGPU surface
pub struct SurfaceComponent {
    config: RwLock<SurfaceConfiguration>,
    surface: RwLock<LazyComponent<Surface>>,
}

impl SurfaceComponent {
    pub fn pending(config: SurfaceConfiguration) -> SurfaceComponent {
        SurfaceComponent {
            config: RwLock::new(config),
            surface: RwLock::new(LazyComponent::Pending),
        }
    }

    pub fn ready(config: SurfaceConfiguration, surface: Surface) -> SurfaceComponent {
        SurfaceComponent {
            config: RwLock::new(config),
            surface: RwLock::new(LazyComponent::Ready(surface)),
        }
    }
}

impl_read_write_lock!(SurfaceComponent, surface, LazyComponent<wgpu::Surface>);
impl_read_write_lock!(SurfaceComponent, config, wgpu::SurfaceConfiguration);

// WGPU texture descriptor
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
pub struct TextureComponent<'a> {
    desc: TextureDescriptor<'a>,
    texture: RwLock<LazyComponent<Texture>>,
}

impl ReadWriteLock<LazyComponent<Texture>> for TextureComponent<'_> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<Texture>> {
        self.texture.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<Texture>> {
        self.texture.write()
    }
}

impl<'a> TextureComponent<'a> {
    pub fn pending(desc: TextureDescriptor<'a>) -> Self {
        TextureComponent {
            desc,
            texture: RwLock::new(LazyComponent::Pending),
        }
    }

    pub fn desc(&self) -> &TextureDescriptor<'_> {
        &self.desc
    }
}

// Render attachment usage flag for TextureComponent
pub enum RenderAttachment {}

pub type RenderAttachmentTexture<'a> = Usage<RenderAttachment, TextureComponent<'a>>;
pub type RenderAttachmentTextureView<'a> = Usage<RenderAttachment, TextureViewComponent<'a>>;

// MSAA frambuffer usage flag for TextureComponent
pub enum MsaaFramebuffer {}

pub type MsaaFramebufferTexture<'a> = Usage<MsaaFramebuffer, TextureComponent<'a>>;
pub type MsaaFramebufferTextureView<'a> = Usage<MsaaFramebuffer, TextureViewComponent<'a>>;

// WGPU surface texture
pub struct SurfaceTextureComponent(RwLock<Option<wgpu::SurfaceTexture>>);

impl_read_write_lock!(SurfaceTextureComponent, 0, Option<wgpu::SurfaceTexture>);

impl SurfaceTextureComponent {
    pub fn pending() -> Self {
        SurfaceTextureComponent(RwLock::new(None))
    }
}

// WGPU texture view
pub struct TextureViewComponent<'a> {
    desc: wgpu::TextureViewDescriptor<'a>,
    view: RwLock<LazyComponent<wgpu::TextureView>>,
}

impl<'a> ReadWriteLock<LazyComponent<wgpu::TextureView>> for TextureViewComponent<'a> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<wgpu::TextureView>> {
        self.view.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<wgpu::TextureView>> {
        self.view.write()
    }
}

impl<'a> TextureViewComponent<'a> {
    pub fn pending(desc: wgpu::TextureViewDescriptor<'a>) -> Self {
        TextureViewComponent {
            desc,
            view: RwLock::new(LazyComponent::Pending),
        }
    }

    pub fn descriptor(&self) -> &wgpu::TextureViewDescriptor<'a> {
        &self.desc
    }
}

// WGPU sampler
pub struct SamplerComponent<'a> {
    desc: wgpu::SamplerDescriptor<'a>,
    view: RwLock<LazyComponent<wgpu::Sampler>>,
}

impl<'a> ReadWriteLock<LazyComponent<wgpu::Sampler>> for SamplerComponent<'a> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<wgpu::Sampler>> {
        self.view.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<wgpu::Sampler>> {
        self.view.write()
    }
}

impl<'a> SamplerComponent<'a> {
    pub fn pending(desc: wgpu::SamplerDescriptor<'a>) -> Self {
        SamplerComponent {
            desc,
            view: RwLock::new(LazyComponent::Pending),
        }
    }

    pub fn descriptor(&self) -> &wgpu::SamplerDescriptor<'a> {
        &self.desc
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

// WGPU buffer
#[derive(Debug)]
pub struct BufferComponent<'a> {
    desc: wgpu::BufferDescriptor<'a>,
    buffer: RwLock<LazyComponent<wgpu::Buffer>>,
}

impl ReadWriteLock<LazyComponent<wgpu::Buffer>> for BufferComponent<'_> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<wgpu::Buffer>> {
        self.buffer.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<wgpu::Buffer>> {
        self.buffer.write()
    }
}

impl<'a> BufferComponent<'a> {
    pub fn pending(desc: BufferDescriptor<'a>) -> Self {
        BufferComponent {
            desc,
            buffer: RwLock::new(LazyComponent::Pending),
        }
    }

    pub fn desc(&self) -> &wgpu::BufferDescriptor<'a> {
        &self.desc
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

// WGPU shader module
pub struct ShaderModuleComponent<'a> {
    descriptor: ShaderModuleDescriptor<'a>,
    bind_group: RwLock<LazyComponent<ShaderModule>>,
}

impl ReadWriteLock<LazyComponent<ShaderModule>> for ShaderModuleComponent<'_> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<ShaderModule>> {
        self.bind_group.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<ShaderModule>> {
        self.bind_group.write()
    }
}

impl<'a> ShaderModuleComponent<'a> {
    pub fn pending(descriptor: ShaderModuleDescriptor<'a>) -> Self {
        ShaderModuleComponent {
            descriptor,
            bind_group: RwLock::new(LazyComponent::Pending),
        }
    }

    pub fn descriptor(&self) -> &ShaderModuleDescriptor<'a> {
        &self.descriptor
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

// Surface size
pub enum SurfaceSize {}
pub type SurfaceSizeComponent = Usage<TextureSize, SizeComponent<RwLock<(u32, u32)>>>;

// Texture size
pub enum TextureSize {}
pub type TextureSizeComponent = Usage<TextureSize, SizeComponent<RwLock<(u32, u32)>>>;
