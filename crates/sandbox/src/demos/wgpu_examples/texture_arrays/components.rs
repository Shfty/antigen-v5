use antigen_core::{RwLock, Usage};
use antigen_wgpu::{
    BufferComponent, ShaderModuleComponent, TextureComponent, TextureViewComponent,
};

use super::Vertex;

// Texture arrays renderer tag
pub struct TextureArrays;

// Usage tags
pub enum Fragment {}

pub type Index = u16;

pub enum UniformWorkaround {}

pub enum Red {}
pub enum Green {}

// Usage-tagged components
pub type VertexBufferComponent = Usage<Vertex, BufferComponent>;
pub type IndexBufferComponent = Usage<Index, BufferComponent>;

pub type VertexShaderComponent = Usage<Vertex, ShaderModuleComponent>;
pub type FragmentShaderComponent = Usage<Fragment, ShaderModuleComponent>;

pub type UniformWorkaroundComponent = Usage<UniformWorkaround, RwLock<bool>>;

pub type RedTexelComponent = Usage<Red, RwLock<[u8; 4]>>;
pub type RedTextureComponent = Usage<Red, TextureComponent>;
pub type RedTextureViewComponent = Usage<Red, TextureViewComponent>;

pub type GreenTexelComponent = Usage<Green, RwLock<[u8; 4]>>;
pub type GreenTextureComponent = Usage<Green, TextureComponent>;
pub type GreenTextureViewComponent = Usage<Green, TextureViewComponent>;
