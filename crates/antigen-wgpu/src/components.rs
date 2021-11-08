use antigen_core::{
    impl_read_write_lock, LazyComponent, ReadWriteLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

use wgpu::{BindGroup, BufferDescriptor, ComputePipeline, ImageCopyTextureBase, ImageDataLayout, RenderPipeline, ShaderModule, ShaderModuleDescriptor, Surface, SurfaceConfiguration, Texture, TextureDescriptor};

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

// WGPU texture
pub struct TextureComponent<'a, T> {
    desc: TextureDescriptor<'a>,
    texture: RwLock<LazyComponent<Texture>>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<LazyComponent<Texture>> for TextureComponent<'_, T> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<Texture>> {
        self.texture.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<Texture>> {
        self.texture.write()
    }
}

impl<'a, T> TextureComponent<'a, T> {
    pub fn pending(desc: TextureDescriptor<'a>) -> Self {
        TextureComponent {
            desc,
            texture: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
        }
    }

    pub fn desc(&self) -> &TextureDescriptor<'_> {
        &self.desc
    }
}

// Render attachment usage flag for TextureComponent
pub enum RenderAttachment {}

// WGPU surface texture
pub struct SurfaceTextureComponent(RwLock<Option<wgpu::SurfaceTexture>>);

impl_read_write_lock!(SurfaceTextureComponent, 0, Option<wgpu::SurfaceTexture>);

impl SurfaceTextureComponent {
    pub fn pending() -> Self {
        SurfaceTextureComponent(RwLock::new(None))
    }
}

// WGPU texture view
pub struct TextureViewComponent<'a, T> {
    desc: wgpu::TextureViewDescriptor<'a>,
    view: RwLock<LazyComponent<wgpu::TextureView>>,
    _phantom: PhantomData<T>,
}

impl<'a, T> ReadWriteLock<LazyComponent<wgpu::TextureView>> for TextureViewComponent<'a, T> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<wgpu::TextureView>> {
        self.view.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<wgpu::TextureView>> {
        self.view.write()
    }
}

impl<'a, T> TextureViewComponent<'a, T> {
    pub fn pending(desc: wgpu::TextureViewDescriptor<'a>) -> Self {
        TextureViewComponent {
            desc,
            view: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
        }
    }

    pub fn descriptor(&self) -> &wgpu::TextureViewDescriptor<'a> {
        &self.desc
    }
}

// WGPU sampler
pub struct SamplerComponent<'a, T> {
    desc: wgpu::SamplerDescriptor<'a>,
    view: RwLock<LazyComponent<wgpu::Sampler>>,
    _phantom: PhantomData<T>,
}

impl<'a, T> ReadWriteLock<LazyComponent<wgpu::Sampler>> for SamplerComponent<'a, T> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<wgpu::Sampler>> {
        self.view.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<wgpu::Sampler>> {
        self.view.write()
    }
}

impl<'a, T> SamplerComponent<'a, T> {
    pub fn pending(desc: wgpu::SamplerDescriptor<'a>) -> Self {
        SamplerComponent {
            desc,
            view: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
        }
    }

    pub fn descriptor(&self) -> &wgpu::SamplerDescriptor<'a> {
        &self.desc
    }
}

// WGPU render pipeline
pub struct RenderPipelineComponent<T> {
    pipeline: RwLock<LazyComponent<RenderPipeline>>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<LazyComponent<RenderPipeline>> for RenderPipelineComponent<T> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<RenderPipeline>> {
        self.pipeline.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<RenderPipeline>> {
        self.pipeline.write()
    }
}

impl<T> RenderPipelineComponent<T> {
    pub fn pending() -> Self {
        RenderPipelineComponent {
            pipeline: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
        }
    }
}

// WGPU compute pipeline
pub struct ComputePipelineComponent<T> {
    pipeline: RwLock<LazyComponent<ComputePipeline>>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<LazyComponent<ComputePipeline>> for ComputePipelineComponent<T> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<ComputePipeline>> {
        self.pipeline.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<ComputePipeline>> {
        self.pipeline.write()
    }
}

impl<T> ComputePipelineComponent<T> {
    pub fn pending() -> Self {
        ComputePipelineComponent {
            pipeline: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
        }
    }
}

// WGPU bind group
pub struct BindGroupComponent<T> {
    bind_group: RwLock<LazyComponent<BindGroup>>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<LazyComponent<BindGroup>> for BindGroupComponent<T> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<BindGroup>> {
        self.bind_group.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<BindGroup>> {
        self.bind_group.write()
    }
}

impl<T> BindGroupComponent<T> {
    pub fn pending() -> Self {
        BindGroupComponent {
            bind_group: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
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
pub struct BufferComponent<'a, T> {
    desc: wgpu::BufferDescriptor<'a>,
    buffer: RwLock<LazyComponent<wgpu::Buffer>>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<LazyComponent<wgpu::Buffer>> for BufferComponent<'_, T> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<wgpu::Buffer>> {
        self.buffer.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<wgpu::Buffer>> {
        self.buffer.write()
    }
}

impl<'a, T> BufferComponent<'a, T> {
    pub fn pending(desc: BufferDescriptor<'a>) -> Self {
        BufferComponent {
            desc,
            buffer: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
        }
    }

    pub fn desc(&self) -> &wgpu::BufferDescriptor<'a> {
        &self.desc
    }
}

// Buffer write operation
pub struct BufferWriteComponent<T, V> {
    offset: RwLock<wgpu::BufferAddress>,
    _phantom: PhantomData<(T, V)>,
}

impl<T, V> ReadWriteLock<wgpu::BufferAddress> for BufferWriteComponent<T, V> {
    fn read(&self) -> RwLockReadGuard<wgpu::BufferAddress> {
        self.offset.read()
    }

    fn write(&self) -> RwLockWriteGuard<wgpu::BufferAddress> {
        self.offset.write()
    }
}

impl<T, V> BufferWriteComponent<T, V> {
    pub fn new(offset: wgpu::BufferAddress) -> Self {
        BufferWriteComponent {
            offset: RwLock::new(offset),
            _phantom: Default::default(),
        }
    }
}

// Texture write operation
pub struct TextureWriteComponent<T, U> {
    image_copy_texture: RwLock<ImageCopyTextureBase<()>>,
    image_data_layout: RwLock<wgpu::ImageDataLayout>,
    _phantom: PhantomData<(T, U)>,
}

impl<T, U> ReadWriteLock<wgpu::ImageDataLayout> for TextureWriteComponent<T, U> {
    fn read(&self) -> RwLockReadGuard<ImageDataLayout> {
        self.image_data_layout.read()
    }

    fn write(&self) -> RwLockWriteGuard<ImageDataLayout> {
        self.image_data_layout.write()
    }
}

impl<T, U> ReadWriteLock<ImageCopyTextureBase<()>> for TextureWriteComponent<T, U> {
    fn read(&self) -> RwLockReadGuard<ImageCopyTextureBase<()>> {
        self.image_copy_texture.read()
    }

    fn write(&self) -> RwLockWriteGuard<ImageCopyTextureBase<()>> {
        self.image_copy_texture.write()
    }
}

impl<T, U> TextureWriteComponent<T, U> {
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
pub struct ShaderModuleComponent<'a, T> {
    descriptor: ShaderModuleDescriptor<'a>,
    bind_group: RwLock<LazyComponent<ShaderModule>>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<LazyComponent<ShaderModule>> for ShaderModuleComponent<'_, T> {
    fn read(&self) -> RwLockReadGuard<LazyComponent<ShaderModule>> {
        self.bind_group.read()
    }

    fn write(&self) -> RwLockWriteGuard<LazyComponent<ShaderModule>> {
        self.bind_group.write()
    }
}

impl<'a, T> ShaderModuleComponent<'a, T> {
    pub fn pending(descriptor: ShaderModuleDescriptor<'a>) -> Self {
        ShaderModuleComponent {
            descriptor,
            bind_group: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
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

// Matrix
pub enum Model {}
pub enum View {}
pub enum Projection {}

pub struct MatrixComponent<M, U> {
    matrix: RwLock<M>,
    _phantom: PhantomData<U>,
}

impl<M, U> ReadWriteLock<M> for MatrixComponent<M, U> {
    fn read(&self) -> RwLockReadGuard<M> {
        self.matrix.read()
    }

    fn write(&self) -> RwLockWriteGuard<M> {
        self.matrix.write()
    }
}

impl<M, U> MatrixComponent<M, U> {
    pub fn new(matrix: M) -> Self {
        MatrixComponent {
            matrix: RwLock::new(matrix),
            _phantom: Default::default(),
        }
    }
}

// Surface usage tag for SizeComponent
pub enum SurfaceSize {}

// Texture usage tag for SizeComponent
pub enum TextureSize {}
