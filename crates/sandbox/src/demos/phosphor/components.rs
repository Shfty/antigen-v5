use bytemuck::{Pod, Zeroable};
use std::time::Instant;

use antigen_core::{RwLock, Usage};
use antigen_wgpu::{
    BindGroupComponent, BufferComponent, RenderPipelineComponent, SamplerComponent,
    ShaderModuleComponent, TextureComponent, TextureViewComponent, ToBytes,
};

// Phosphor renderer tag
pub struct Phosphor;

// Usage tags
pub enum Position {}

pub enum StartTime {}
pub enum Timestamp {}
pub enum TotalTime {}
pub enum DeltaTime {}

pub enum Uniform {}

pub enum Hdr {}
pub enum HdrDecay {}
pub enum HdrRaster {}
pub enum HdrFrontBuffer {}
pub enum HdrBackBuffer {}
pub enum Blit {}

pub enum Linear {}

pub enum Gradients {}

pub enum Vertex {}
pub enum Instance {}

pub enum Projection {}

pub enum FlipFlop {}

pub enum Origin {}

// Usage-tagged components
pub type PositionComponent = Usage<Position, RwLock<(f32, f32)>>;

pub type StartTimeComponent = Usage<StartTime, Instant>;
pub type TimestampComponent = Usage<Timestamp, RwLock<Instant>>;
pub type TotalTimeComponent = Usage<TotalTime, RwLock<f32>>;
pub type DeltaTimeComponent = Usage<DeltaTime, RwLock<f32>>;
pub type ProjectionMatrixComponent = Usage<Projection, RwLock<[[f32; 4]; 4]>>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct UniformData {
    total_time: f32,
    delta_time: f32,
    _pad: [f32; 2],
    projection: [[f32; 4]; 4],
}

pub type UniformDataComponent = Usage<Uniform, RwLock<UniformData>>;
pub type UniformBufferComponent = Usage<Uniform, BufferComponent>;

pub type HdrDecayShaderComponent = Usage<HdrDecay, ShaderModuleComponent>;
pub type HdrRasterShaderComponent = Usage<HdrRaster, ShaderModuleComponent>;
pub type HdrFrontBufferComponent = Usage<HdrFrontBuffer, TextureComponent>;
pub type HdrBackBufferComponent = Usage<HdrBackBuffer, TextureComponent>;
pub type HdrFrontBufferViewComponent = Usage<HdrFrontBuffer, TextureViewComponent>;
pub type HdrBackBufferViewComponent = Usage<HdrBackBuffer, TextureViewComponent>;

pub type LinearSamplerComponent = Usage<Linear, SamplerComponent>;
pub type HdrBlitPipelineComponent = Usage<HdrDecay, RenderPipelineComponent>;
pub type HdrRasterPipelineComponent = Usage<HdrRaster, RenderPipelineComponent>;
pub type FrontBindGroupComponent = Usage<HdrFrontBuffer, BindGroupComponent>;
pub type BackBindGroupComponent = Usage<HdrBackBuffer, BindGroupComponent>;
pub type UniformBindGroupComponent = Usage<Uniform, BindGroupComponent>;

pub type BlitShaderComponent = Usage<Blit, ShaderModuleComponent>;
pub type BlitPipelineComponent = Usage<Blit, RenderPipelineComponent>;

pub type GradientTextureComponent = Usage<Gradients, TextureComponent>;
pub type GradientTextureViewComponent = Usage<Gradients, TextureViewComponent>;

pub type OriginComponent = Usage<Origin, RwLock<(f32, f32)>>;

pub type GradientData = Vec<u8>;
pub type GradientDataComponent = Usage<Gradients, RwLock<GradientData>>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct VertexData {
    pub position: [f32; 4],
    pub end: f32,
    pub _pad: [f32; 3],
}

pub type VertexDataComponent = Usage<Vertex, RwLock<Vec<VertexData>>>;
pub type VertexBufferComponent = Usage<Vertex, BufferComponent>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct InstanceData {
    pub position: [f32; 4],
    pub prev_position: [f32; 4],
    pub intensity: f32,
    pub delta_intensity: f32,
    pub delta_delta: f32,
    pub gradient: f32,
}

impl ToBytes for InstanceData {
    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

pub type InstanceDataComponent = Usage<Instance, RwLock<InstanceData>>;
pub type InstanceBufferComponent = Usage<Instance, BufferComponent>;

pub type BufferFlipFlopComponent = Usage<FlipFlop, RwLock<bool>>;

pub struct Oscilloscope {
    f: Box<dyn Fn(f32) -> (f32, f32) + Send + Sync>,
    speed: f32,
    magnitude: f32,
}

impl Oscilloscope {
    pub fn new<F>(speed: f32, magnitude: f32, f: F) -> Self
    where
        F: Fn(f32) -> (f32, f32) + Send + Sync + 'static,
    {
        Oscilloscope {
            speed,
            magnitude,
            f: Box::new(f),
        }
    }

    pub fn eval(&self, f: f32) -> (f32, f32) {
        let (x, y) = (self.f)(f * self.speed);
        (x * self.magnitude, y * self.magnitude)
    }
}
