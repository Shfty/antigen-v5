use antigen_core::{RwLock, Usage};
use antigen_wgpu::{BufferComponent, RenderPipelineComponent, SamplerComponent, ShaderModuleComponent, TextureComponent, TextureViewComponent};

// Hello triangle renderer tag
pub struct Mipmap;

// Usage tags
pub enum Texels {}
pub enum Draw {}
pub enum Blit {}
pub enum Uniform {}
pub enum JuliaSet {}
pub enum ViewProjection {}

// Usage-tagged components
pub type DrawShaderComponent = Usage<Draw, ShaderModuleComponent>;
pub type DrawPipelineComponent = Usage<Draw, RenderPipelineComponent>;

pub type ViewProjectionMatrix = Usage<ViewProjection, RwLock<[f32; 16]>>;

pub type UniformBufferComponent = Usage<Uniform, BufferComponent>;

pub type JuliaSetTextureComponent = Usage<JuliaSet, TextureComponent>;
pub type JuliaSetTextureViewComponent = Usage<JuliaSet, TextureViewComponent>;
pub type JuliaSetSamplerComponent = Usage<JuliaSet, SamplerComponent>;
