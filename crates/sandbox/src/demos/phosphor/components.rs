use bytemuck::{Pod, Zeroable};
use std::time::Instant;

use antigen_core::{Changed, RwLock, Usage};
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
pub enum HdrLine {}
pub enum HdrMesh {}
pub enum HdrFrontBuffer {}
pub enum HdrBackBuffer {}
pub enum HdrDepthBuffer {}
pub enum Tonemap {}

pub enum Linear {}

pub enum Gradients {}

pub enum LineVertex {}
pub enum LineInstance {}

pub enum MeshVertex {}
pub enum MeshIndex {}

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
pub type HdrLineShaderComponent = Usage<HdrLine, ShaderModuleComponent>;
pub type HdrMeshShaderComponent = Usage<HdrMesh, ShaderModuleComponent>;
pub type HdrFrontBufferComponent = Usage<HdrFrontBuffer, TextureComponent>;
pub type HdrBackBufferComponent = Usage<HdrBackBuffer, TextureComponent>;
pub type HdrDepthBufferComponent = Usage<HdrDepthBuffer, TextureComponent>;
pub type HdrFrontBufferViewComponent = Usage<HdrFrontBuffer, TextureViewComponent>;
pub type HdrBackBufferViewComponent = Usage<HdrBackBuffer, TextureViewComponent>;
pub type HdrDepthBufferViewComponent = Usage<HdrDepthBuffer, TextureViewComponent>;

pub type LinearSamplerComponent = Usage<Linear, SamplerComponent>;
pub type HdrDecayPipelineComponent = Usage<HdrDecay, RenderPipelineComponent>;
pub type HdrLinePipelineComponent = Usage<HdrLine, RenderPipelineComponent>;
pub type HdrMeshPipelineComponent = Usage<HdrMesh, RenderPipelineComponent>;
pub type FrontBindGroupComponent = Usage<HdrFrontBuffer, BindGroupComponent>;
pub type BackBindGroupComponent = Usage<HdrBackBuffer, BindGroupComponent>;
pub type UniformBindGroupComponent = Usage<Uniform, BindGroupComponent>;

pub type TonemapShaderComponent = Usage<Tonemap, ShaderModuleComponent>;
pub type TonemapPipelineComponent = Usage<Tonemap, RenderPipelineComponent>;

pub type GradientTextureComponent = Usage<Gradients, TextureComponent>;
pub type GradientTextureViewComponent = Usage<Gradients, TextureViewComponent>;

pub type OriginComponent = Usage<Origin, RwLock<(f32, f32, f32)>>;

pub type GradientData = Vec<u8>;
pub type GradientDataComponent = Usage<Gradients, RwLock<GradientData>>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct LineVertexData {
    pub position: [f32; 4],
    pub end: f32,
    pub _pad: [f32; 3],
}

pub type LineVertexDataComponent = Usage<LineVertex, RwLock<Vec<LineVertexData>>>;
pub type LineVertexBufferComponent = Usage<LineVertex, BufferComponent>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct LineInstanceData {
    pub v0: MeshVertexData,
    pub v1: MeshVertexData,
}

impl ToBytes for LineInstanceData {
    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

pub type LineInstanceDataComponent = Usage<LineInstance, RwLock<Vec<LineInstanceData>>>;
pub type LineInstanceBufferComponent = Usage<LineInstance, BufferComponent>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct MeshVertexData {
    pub position: [f32; 4],
    pub intensity: f32,
    pub delta_intensity: f32,
    pub delta_delta: f32,
    pub gradient: f32,
}

impl MeshVertexData {
    pub fn new(
        position: (f32, f32, f32),
        intensity: f32,
        delta_intensity: f32,
        delta_delta: f32,
        gradient: f32,
    ) -> Self {
        MeshVertexData {
            position: [position.0, position.1, position.2, 1.0],
            intensity,
            delta_intensity,
            delta_delta,
            gradient,
        }
    }
}

pub type MeshVertexDataComponent = Usage<MeshVertex, RwLock<Vec<MeshVertexData>>>;
pub type MeshVertexBufferComponent = Usage<MeshVertex, BufferComponent>;

pub type MeshIndexDataComponent = Usage<MeshIndex, RwLock<Vec<u16>>>;
pub type MeshIndexBufferComponent = Usage<MeshIndex, BufferComponent>;

pub type BufferFlipFlopComponent = Usage<FlipFlop, RwLock<bool>>;

pub struct Oscilloscope {
    f: Box<dyn Fn(f32) -> (f32, f32, f32) + Send + Sync>,
    speed: f32,
    magnitude: f32,
}

impl Oscilloscope {
    pub fn new<F>(speed: f32, magnitude: f32, f: F) -> Self
    where
        F: Fn(f32) -> (f32, f32, f32) + Send + Sync + 'static,
    {
        Oscilloscope {
            speed,
            magnitude,
            f: Box::new(f),
        }
    }

    pub fn eval(&self, f: f32) -> (f32, f32, f32) {
        let (x, y, z) = (self.f)(f * self.speed);
        (x * self.magnitude, y * self.magnitude, z * self.magnitude)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Timer {
    pub timestamp: std::time::Instant,
    pub duration: std::time::Duration,
}

pub type TimerComponent = Changed<RwLock<Timer>>;
