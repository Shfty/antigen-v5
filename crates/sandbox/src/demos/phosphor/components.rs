use bytemuck::{Pod, Zeroable};
use std::time::Instant;

use antigen_core::{Changed, RwLock, Usage};
use antigen_wgpu::{
    BindGroupComponent, BufferComponent, ComputePipelineComponent, RenderPipelineComponent,
    SamplerComponent, ShaderModuleComponent, TextureComponent, TextureViewComponent, ToBytes,
};

// Phosphor renderer tag
pub struct PhosphorRenderer;

// Usage tags
pub enum Position {}

pub enum StartTime {}
pub enum Timestamp {}
pub enum TotalTime {}
pub enum DeltaTime {}

pub enum Uniform {}

pub enum PhosphorDecay {}
pub enum BeamLine {}
pub enum BeamMesh {}
pub enum BeamBuffer {}
pub enum BeamMultisample {}
pub enum BeamDepthBuffer {}
pub enum PhosphorFrontBuffer {}
pub enum PhosphorBackBuffer {}
pub enum Tonemap {}

pub enum Linear {}

pub enum LineVertex {}
pub enum LineIndex {}
pub enum LineInstance {}

pub enum MeshVertex {}
pub enum MeshIndex {}

pub enum Perspective {}
pub enum Orthographic {}

pub enum FlipFlop {}

pub enum Origin {}

pub enum ComputeLineInstances {}

pub enum MapFile {}

#[derive(Debug)]
pub enum VertexCount {}

#[derive(Debug)]
pub enum MeshIndexCount {}

#[derive(Debug)]
pub enum LineIndexCount {}

// Usage-tagged components
pub type PositionComponent = Usage<Position, RwLock<(f32, f32)>>;

pub type StartTimeComponent = Usage<StartTime, Instant>;
pub type TimestampComponent = Usage<Timestamp, RwLock<Instant>>;
pub type TotalTimeComponent = Usage<TotalTime, RwLock<f32>>;
pub type DeltaTimeComponent = Usage<DeltaTime, RwLock<f32>>;
pub type PerspectiveMatrixComponent = Usage<Perspective, RwLock<[[f32; 4]; 4]>>;
pub type OrthographicMatrixComponent = Usage<Orthographic, RwLock<[[f32; 4]; 4]>>;
pub type VertexCountComponent = Usage<VertexCount, RwLock<u64>>;
pub type MeshIndexCountComponent = Usage<MeshIndexCount, RwLock<u64>>;
pub type LineIndexCountComponent = Usage<LineIndexCount, RwLock<u64>>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct UniformData {
    perspective: [[f32; 4]; 4],
    orthographic: [[f32; 4]; 4],
    total_time: f32,
    delta_time: f32,
    _pad: [f32; 2],
}

pub type UniformDataComponent = Usage<Uniform, RwLock<UniformData>>;
pub type UniformBufferComponent = Usage<Uniform, BufferComponent>;

pub type ComputeLineInstancesShader = Usage<ComputeLineInstances, ShaderModuleComponent>;
pub type PhosphorDecayShaderComponent = Usage<PhosphorDecay, ShaderModuleComponent>;
pub type BeamLineShaderComponent = Usage<BeamLine, ShaderModuleComponent>;
pub type BeamMeshShaderComponent = Usage<BeamMesh, ShaderModuleComponent>;

pub type BeamBufferComponent = Usage<BeamBuffer, TextureComponent>;
pub type BeamDepthBufferComponent = Usage<BeamDepthBuffer, TextureComponent>;
pub type BeamMultisampleComponent = Usage<BeamMultisample, TextureComponent>;
pub type PhosphorFrontBufferComponent = Usage<PhosphorFrontBuffer, TextureComponent>;
pub type PhosphorBackBufferComponent = Usage<PhosphorBackBuffer, TextureComponent>;

pub type BeamBufferViewComponent = Usage<BeamBuffer, TextureViewComponent>;
pub type BeamDepthBufferViewComponent = Usage<BeamDepthBuffer, TextureViewComponent>;
pub type BeamMultisampleViewComponent = Usage<BeamMultisample, TextureViewComponent>;
pub type PhosphorFrontBufferViewComponent = Usage<PhosphorFrontBuffer, TextureViewComponent>;
pub type PhosphorBackBufferViewComponent = Usage<PhosphorBackBuffer, TextureViewComponent>;

pub type LinearSamplerComponent = Usage<Linear, SamplerComponent>;

pub type ComputeLineInstancesPipeline = Usage<ComputeLineInstances, ComputePipelineComponent>;
pub type BeamLinePipelineComponent = Usage<BeamLine, RenderPipelineComponent>;
pub type BeamMeshPipelineComponent = Usage<BeamMesh, RenderPipelineComponent>;
pub type PhosphorDecayPipelineComponent = Usage<PhosphorDecay, RenderPipelineComponent>;

pub type ComputeBindGroupComponent = Usage<ComputeLineInstances, BindGroupComponent>;
pub type UniformBindGroupComponent = Usage<Uniform, BindGroupComponent>;
pub type FrontBindGroupComponent = Usage<PhosphorFrontBuffer, BindGroupComponent>;
pub type BackBindGroupComponent = Usage<PhosphorBackBuffer, BindGroupComponent>;

pub type TonemapShaderComponent = Usage<Tonemap, ShaderModuleComponent>;
pub type TonemapPipelineComponent = Usage<Tonemap, RenderPipelineComponent>;

pub type OriginComponent = Usage<Origin, RwLock<(f32, f32, f32)>>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct LineVertexData {
    pub position: [f32; 3],
    pub end: f32,
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

pub type LineIndexDataComponent = Usage<LineIndex, RwLock<Vec<u32>>>;
pub type LineIndexBufferComponent = Usage<LineIndex, BufferComponent>;

pub type LineInstanceBufferComponent = Usage<LineInstance, BufferComponent>;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct MeshVertexData {
    pub position: [f32; 3],
    pub surface_color: [f32; 3],
    pub line_color: [f32; 3],
    pub intensity: f32,
    pub delta_intensity: f32,
    pub _pad: f32,
}

impl MeshVertexData {
    pub fn new(
        position: (f32, f32, f32),
        surface_color: (f32, f32, f32),
        line_color: (f32, f32, f32),
        intensity: f32,
        delta_intensity: f32,
    ) -> Self {
        MeshVertexData {
            position: [position.0, position.1, position.2],
            surface_color: [surface_color.0, surface_color.1, surface_color.2],
            line_color: [line_color.0, line_color.1, line_color.2],
            intensity,
            delta_intensity,
            ..Default::default()
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
