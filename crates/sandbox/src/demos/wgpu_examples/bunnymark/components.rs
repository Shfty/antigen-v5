use antigen_core::{RwLock, Usage};
use antigen_wgpu::{BindGroupComponent, BufferComponent, SamplerComponent, Texels, TextureComponent, TextureDescriptorComponent, TextureViewComponent};

use super::{Globals, Locals};

// Hello triangle renderer tag
pub struct Bunnymark;

// Usage tags
pub enum Logo {}
pub enum Global {}
pub enum Local {}
pub enum PlayfieldExtent {}

// Usage-tagged components
pub type GlobalDataComponent = RwLock<Globals>;
pub type GlobalBufferComponent = Usage<Global, BufferComponent>;
pub type LocalBufferComponent = Usage<Local, BufferComponent>;

pub type GlobalBindGroupComponent<'a> = Usage<Global, BindGroupComponent>;
pub type LocalBindGroupComponent<'a> = Usage<Local, BindGroupComponent>;

pub type PlayfieldExtentComponent = Usage<PlayfieldExtent, RwLock<(u32, u32)>>;

pub type LogoTextureDescriptorComponent<'a> = Usage<Logo, TextureDescriptorComponent<'a>>;
pub type LogoTextureComponent = Usage<Logo, TextureComponent>;
pub type LogoTextureViewComponent = Usage<Logo, TextureViewComponent>;
pub type LogoSamplerComponent = Usage<Logo, SamplerComponent>;

pub type TexelDataComponent = Usage<Texels, RwLock<Vec<u8>>>;

pub type BunniesComponent = RwLock<Vec<Locals>>;

