use antigen_core::Usage;
use antigen_wgpu::{BindGroupComponent, BufferComponent, ShaderModuleComponent};

// Hello triangle renderer tag
pub struct Boids;

// Usage flags
pub enum Compute {}
pub enum Draw {}
pub enum Vertex {}
pub enum Uniform {}
pub enum FrontBuffer {}
pub enum BackBuffer {}

// Usage-tagged components
pub type VertexBufferComponent = Usage<Vertex, BufferComponent>;
pub type UniformBufferComponent = Usage<Uniform, BufferComponent>;

pub type FrontBufferComponent = Usage<FrontBuffer, BufferComponent>;
pub type BackBufferComponent = Usage<BackBuffer, BufferComponent>;

pub type FrontBufferBindGroupComponent = Usage<FrontBuffer, BindGroupComponent>;
pub type BackBufferBindGroupComponent = Usage<BackBuffer, BindGroupComponent>;

pub type ComputeShaderModuleComponent = Usage<Compute, ShaderModuleComponent>;
pub type DrawShaderModuleComponent = Usage<Draw, ShaderModuleComponent>;
