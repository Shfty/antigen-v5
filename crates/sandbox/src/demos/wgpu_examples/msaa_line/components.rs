use antigen_core::{RwLock, Usage};
use antigen_wgpu::{BufferComponent, MeshVertices};

use super::Vertex;

// Hello triangle renderer tag
pub struct MsaaLine;

// Vertex buffer usage flag
pub enum VertexBuffer {}

pub type VertexBufferComponent = Usage<VertexBuffer, BufferComponent>;

pub type MeshVerticesComponent = Usage<MeshVertices, RwLock<Vec<Vertex>>>;
