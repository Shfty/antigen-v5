use std::marker::PhantomData;

use parking_lot::RwLock;
use wgpu::{BindGroup, BufferDescriptor, RenderPipeline, Surface, SurfaceConfiguration, Texture, TextureDescriptor};

use crate::{impl_read_write_lock, LazyComponent, ReadWriteLock};

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
    texels: Vec<T>,
    texture: RwLock<LazyComponent<Texture>>,
}

impl<T> ReadWriteLock<LazyComponent<Texture>> for TextureComponent<'_, T> {
    fn read(&self) -> parking_lot::RwLockReadGuard<LazyComponent<Texture>> {
        self.texture.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<LazyComponent<Texture>> {
        self.texture.write()
    }
}

impl<'a, T> TextureComponent<'a, T> {
    pub fn pending(desc: TextureDescriptor<'a>, texels: Vec<T>) -> Self {
        TextureComponent {
            desc,
            texels,
            texture: RwLock::new(LazyComponent::Pending),
        }
    }
}

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
    fn read(&self) -> parking_lot::RwLockReadGuard<LazyComponent<wgpu::TextureView>> {
        self.view.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<LazyComponent<wgpu::TextureView>> {
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

// WGPU render pipeline
pub struct RenderPipelineComponent<T = ()> {
    pipeline: RwLock<LazyComponent<RenderPipeline>>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<LazyComponent<RenderPipeline>> for RenderPipelineComponent<T> {
    fn read(&self) -> parking_lot::RwLockReadGuard<LazyComponent<RenderPipeline>> {
        self.pipeline.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<LazyComponent<RenderPipeline>> {
        self.pipeline.write()
    }
}

impl<T> RenderPipelineComponent<T> {
    pub fn pending() -> RenderPipelineComponent {
        RenderPipelineComponent {
            pipeline: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
        }
    }
}

// WGPU bind group
pub struct BindGroupComponent(RwLock<LazyComponent<BindGroup>>);

impl_read_write_lock!(BindGroupComponent, 0, LazyComponent<BindGroup>);

impl BindGroupComponent {
    pub fn pending() -> BindGroupComponent {
        BindGroupComponent(RwLock::new(LazyComponent::Pending))
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
pub struct BufferComponent<'a, T = ()> {
    desc: wgpu::BufferDescriptor<'a>,
    pipeline: RwLock<LazyComponent<wgpu::Buffer>>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<LazyComponent<wgpu::Buffer>> for BufferComponent<'_, T> {
    fn read(&self) -> parking_lot::RwLockReadGuard<LazyComponent<wgpu::Buffer>> {
        self.pipeline.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<LazyComponent<wgpu::Buffer>> {
        self.pipeline.write()
    }
}

impl<T> BufferComponent<'_, T> {
    pub fn pending(desc: BufferDescriptor) -> BufferComponent {
        BufferComponent {
            desc,
            pipeline: RwLock::new(LazyComponent::Pending),
            _phantom: Default::default(),
        }
    }
}

// Mesh vertices
pub struct MeshVertices<T>(RwLock<Vec<T>>);

impl<T> ReadWriteLock<Vec<T>> for MeshVertices<T> {
    fn read(&self) -> parking_lot::RwLockReadGuard<Vec<T>> {
        self.0.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<Vec<T>> {
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
    fn read(&self) -> parking_lot::RwLockReadGuard<Vec<T>> {
        self.0.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<Vec<T>> {
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
    fn read(&self) -> parking_lot::RwLockReadGuard<Vec<T>> {
        self.0.read()
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<Vec<T>> {
        self.0.write()
    }
}

impl<T> MeshIndices<T> {
    pub fn new(indices: Vec<T>) -> Self {
        MeshIndices(RwLock::new(indices))
    }
}
