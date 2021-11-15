use antigen_core::Usage;
use antigen_wgpu::{BindGroupComponent, BindGroupLayoutComponent, RenderPipelineComponent, SamplerComponent, ShaderModuleComponent, TextureDescriptorComponent, TextureViewComponent};

// Conservative raster renderer tag
pub struct ConservativeRaster;

// Usage tags
pub enum TriangleAndLines {}

pub enum TriangleConservative {}
pub enum TriangleRegular {}
pub enum Upscale {}
pub enum Lines {}

pub enum LowResTarget {}

pub type LowResTextureDescriptorComponent<'a> = Usage<LowResTarget, TextureDescriptorComponent<'a>>;
pub type LowResTextureViewComponent = Usage<LowResTarget, TextureViewComponent>;

pub type TriangleConservativePipelineComponent = Usage<TriangleConservative, RenderPipelineComponent>;
pub type TriangleRegularPipelineComponent = Usage<TriangleRegular, RenderPipelineComponent>;
pub type UpscalePipelineComponent = Usage<Upscale, RenderPipelineComponent>;
pub type LinesPipelineComponent = Usage<Lines, RenderPipelineComponent>;

pub type UpscaleBindGroupLayoutComponent = Usage<Upscale, BindGroupLayoutComponent>;
pub type UpscaleBindGroupComponent = Usage<Upscale, BindGroupComponent>;

pub type LowResSamplerComponent = Usage<LowResTarget, SamplerComponent>;

pub type TriangleAndLinesShaderComponent = Usage<TriangleAndLines, ShaderModuleComponent>;
pub type UpscaleShaderComponent = Usage<Upscale, ShaderModuleComponent>;
