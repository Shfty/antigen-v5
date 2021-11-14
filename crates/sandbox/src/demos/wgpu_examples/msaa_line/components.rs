use antigen_core::Usage;
use antigen_wgpu::BufferComponent;

// Hello triangle renderer tag
pub struct MsaaLine;

// Vertex buffer usage flag
pub enum VertexBuffer {}

pub type VertexBufferComponent = Usage<VertexBuffer, BufferComponent>;

