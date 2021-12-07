use std::ops::Range;

use antigen_core::{IndirectComponent, RwLock, Usage};
use antigen_wgpu::{
    wgpu::{BufferAddress, Color, IndexFormat},
    BindGroupComponent, BufferComponent, RenderPipelineComponent, SamplerComponent,
    TextureComponent, TextureViewComponent,
};
use legion::Entity;
use nalgebra::Matrix4;

// Shadow renderer tag
pub struct Shadow;

// Usage tags
#[derive(Debug)]
pub enum VertexTag {}
#[derive(Debug)]
pub enum IndexTag {}

#[derive(Debug)]
pub enum ObjectTag {}

#[derive(Debug)]
pub enum ForwardPass {}
#[derive(Debug)]
pub enum ShadowPass {}

#[derive(Debug)]
pub enum LightTag {}

#[derive(Debug)]
pub enum RotationSpeed {}

#[derive(Debug)]
pub enum IndexCount {}

#[derive(Debug)]
pub enum UniformOffset {}

#[derive(Debug)]
pub struct FieldOfView;

#[derive(Debug)]
pub enum LightsAreDirty {}

// Usage-tagged components
#[derive(Default)]
pub struct PlaneMesh;

#[derive(Default)]
pub struct CubeMesh;

pub enum Mesh {
    Plane,
    Cube
}

pub type VertexBufferComponent = Usage<VertexTag, BufferComponent>;
pub type IndexBufferComponent = Usage<IndexTag, BufferComponent>;

pub type ForwardRenderPipeline = Usage<ForwardPass, RenderPipelineComponent>;
pub type ForwardBindGroup = Usage<ForwardPass, BindGroupComponent>;
pub type ForwardUniformBuffer = Usage<ForwardPass, BufferComponent>;

pub type ForwardDepthView = Usage<ForwardPass, TextureViewComponent>;

pub type ShadowRenderPipeline = Usage<ShadowPass, RenderPipelineComponent>;
pub type ShadowBindGroup = Usage<ShadowPass, BindGroupComponent>;
pub type ShadowUniformBuffer = Usage<ShadowPass, BufferComponent>;

pub type ShadowTextureViewComponent = Usage<ShadowPass, TextureViewComponent>;
pub type ShadowSamplerComponent = Usage<ShadowPass, SamplerComponent>;

pub type LightStorageBuffer = Usage<LightTag, BufferComponent>;

pub type ObjectMatrixComponent = Usage<ObjectTag, RwLock<Matrix4<f32>>>;
pub type ObjectBindGroup = Usage<ObjectTag, BindGroupComponent>;
pub type ObjectUniformBuffer = Usage<ObjectTag, BufferComponent>;

pub type RotationSpeedComponent = Usage<RotationSpeed, f32>;
pub type LightFovComponent = Usage<FieldOfView, f32>;
pub type UniformOffsetComponent = Usage<UniformOffset, u32>;
pub type IndexCountComponent = Usage<IndexCount, BufferAddress>;
pub type LightsAreDirtyComponent = Usage<LightsAreDirty, RwLock<bool>>;

// Queries
pub type BufferQuery<'a, T> = (
    &'a T,
    &'a VertexBufferComponent,
    &'a IndexBufferComponent,
    &'a IndexFormat,
    &'a IndexCountComponent,
);

pub type ObjectQuery<'a> = (
    &'a Mesh,
    &'a ObjectMatrixComponent,
    &'a RotationSpeedComponent,
    &'a Color,
    &'a UniformOffsetComponent,
);

pub type LightQuery<'a> = (
    &'a nalgebra::Vector3<f32>,
    &'a Color,
    &'a LightFovComponent,
    &'a Range<f32>,
    &'a ShadowTextureViewComponent,
);
