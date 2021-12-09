use super::Vertex;
use antigen_core::{RwLock, Usage};
use antigen_wgpu::{
    BufferComponent, RenderPipelineComponent, TextureComponent, TextureViewComponent,
};

// Skybox renderer tag
pub struct Skybox;

// Usage tags
pub enum EntityTag {}
pub enum Sky {}
pub enum Depth {}
pub enum Objects {}
pub enum VertexCount {}
pub enum Uniform {}
pub enum Texture {}

// Usage-tagged components
pub type EntityPipelineComponent = Usage<EntityTag, RenderPipelineComponent>;
pub type SkyPipelineComponent = Usage<Sky, RenderPipelineComponent>;
pub type DepthTextureView = Usage<Depth, TextureViewComponent>;
pub type UniformDataComponent = RwLock<[f32; 52]>;
pub type UniformBufferComponent = Usage<Uniform, BufferComponent>;

pub type VertexDataComponent = RwLock<Vec<Vertex>>;
pub type VertexBufferComponent = Usage<Vertex, BufferComponent>;
pub type VertexCountComponent = Usage<VertexCount, usize>;

pub type SkyboxTextureComponent = Usage<Texture, TextureComponent>;
pub type SkyboxTextureViewComponent = Usage<Texture, TextureViewComponent>;
