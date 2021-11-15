use antigen_core::Usage;
use antigen_wgpu::{TextureViewComponent, RenderPipelineComponent, BindGroupComponent};

// Conservative raster renderer tag
pub struct ConservativeRaster;

// Usage tags
pub enum TriangleAndLines {}

pub enum TriangleConservative {}
pub enum TriangleRegular {}
pub enum Upscale {}
pub enum Lines {}

pub enum LowResTarget {}

pub type LowResTextureViewComponent = Usage<LowResTarget, TextureViewComponent>;

pub type TriangleConservativePipelineComponent = Usage<TriangleConservative, RenderPipelineComponent>;
pub type TriangleRegularPipelineComponent = Usage<TriangleRegular, RenderPipelineComponent>;
pub type UpscalePipelineComponent = Usage<Upscale, RenderPipelineComponent>;
pub type LinesPipelineComponent = Usage<Lines, RenderPipelineComponent>;
pub type UpscaleBindGroupComponent = Usage<Upscale, BindGroupComponent>;
