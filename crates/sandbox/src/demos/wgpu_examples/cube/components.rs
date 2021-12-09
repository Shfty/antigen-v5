use antigen_core::{RwLock, Usage};
use antigen_wgpu::{BufferComponent, MeshUvs, MeshVertices, RenderPipelineComponent, Texels, TextureComponent, TextureDescriptorComponent, TextureViewComponent};

// Cube renderer tag
pub struct Cube;

// Usage tags
#[derive(Debug)]
pub enum OpaquePass {}

#[derive(Debug)]
pub enum WirePass {}

#[derive(Debug)]
pub enum Vertex {}

#[derive(Debug)]
pub enum Index {}

#[derive(Debug)]
pub enum Uniform {}

#[derive(Debug)]
pub enum Mandelbrot {}

#[derive(Debug)]
pub enum ViewProjection {}

// Usage-tagged components
pub type OpaquePassRenderPipelineComponent = Usage<OpaquePass, RenderPipelineComponent>;
pub type WirePassRenderPipelineComponent = Usage<WirePass, RenderPipelineComponent>;

pub type ViewProjectionMatrix = Usage<ViewProjection, RwLock<[f32; 16]>>;

pub type VertexBufferComponent = Usage<Vertex, BufferComponent>;
pub type IndexBufferComponent = Usage<Index, BufferComponent>;
pub type UniformBufferComponent = Usage<Uniform, BufferComponent>;

pub type MandelbrotTextureDescriptorComponent<'a> =
    Usage<Mandelbrot, TextureDescriptorComponent<'a>>;
pub type MandelbrotTextureComponent = Usage<Mandelbrot, TextureComponent>;
pub type MandelbrotTextureViewComponent = Usage<Mandelbrot, TextureViewComponent>;

pub type MeshVerticesComponent = Usage<MeshVertices, RwLock<Vec<[f32; 3]>>>;
pub type MeshUvsComponent = Usage<MeshUvs, RwLock<Vec<[f32; 2]>>>;

pub type TexelsComponent = Usage<Texels, RwLock<Vec<u8>>>;
