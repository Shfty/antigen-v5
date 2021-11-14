mod components;
mod systems;

use std::sync::{atomic::AtomicUsize, Arc};

use antigen_winit::RedrawUnconditionally;
pub use components::*;
pub use systems::*;

use antigen_core::{
    parallel, serial, single, AddIndirectComponent, ImmutableSchedule, RwLock, Serial, Single,
    Usage,
};

use antigen_wgpu::{
    assemble_buffer_data,
    wgpu::{
        BufferAddress, BufferDescriptor, BufferUsages, Device, ShaderModuleDescriptor, ShaderSource,
    },
    BindGroupComponent, BufferComponent, CommandBuffersComponent, ComputePipelineComponent,
    RenderAttachmentTextureView, RenderPipelineComponent, ShaderModuleComponent, SurfaceConfigurationComponent,
};

use rand::{distributions::Distribution, SeedableRng};

pub enum Compute {}
pub enum Draw {}
pub enum Vertex {}
pub enum Uniform {}
pub enum FrontBuffer {}
pub enum BackBuffer {}

pub type VertexBufferComponent = Usage<Vertex, BufferComponent>;
pub type UniformBufferComponent = Usage<Uniform, BufferComponent>;

pub type FrontBufferComponent = Usage<FrontBuffer, BufferComponent>;
pub type BackBufferComponent = Usage<BackBuffer, BufferComponent>;

pub type FrontBufferBindGroupComponent = Usage<FrontBuffer, BindGroupComponent>;
pub type BackBufferBindGroupComponent = Usage<BackBuffer, BindGroupComponent>;

pub type ComputeShaderModuleComponent = Usage<Compute, ShaderModuleComponent>;
pub type DrawShaderModuleComponent = Usage<Draw, ShaderModuleComponent>;

const NUM_PARTICLES: usize = 1500;
const PARTICLES_PER_GROUP: usize = 64;

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    antigen_winit::assemble_window(cmd, &(window_entity,));
    antigen_wgpu::assemble_window_surface(cmd, &(window_entity,));

    // Add title to window
    antigen_winit::assemble_window_title(cmd, &(window_entity,), &"Boids");

    // Redraw the window unconditionally
    cmd.add_component(window_entity, RedrawUnconditionally);

    // Renderer
    cmd.add_component(renderer_entity, Boids);
    cmd.add_component(renderer_entity, RenderPipelineComponent::pending());
    cmd.add_component(renderer_entity, ComputePipelineComponent::pending());
    cmd.add_component(
        renderer_entity,
        Usage::<FrontBuffer, _>::new(BindGroupComponent::pending()),
    );
    cmd.add_component(
        renderer_entity,
        Usage::<BackBuffer, _>::new(BindGroupComponent::pending()),
    );
    cmd.add_component(renderer_entity, CommandBuffersComponent::new());
    cmd.add_indirect_component::<SurfaceConfigurationComponent>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Shaders
    antigen_wgpu::assemble_shader_usage::<Compute>(
        cmd,
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("compute.wgsl"))),
        },
    );

    antigen_wgpu::assemble_shader_usage::<Draw>(
        cmd,
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("draw.wgsl"))),
        },
    );

    // Buffer data
    // Vertices
    let vertex_buffer_data = [-0.01f32, -0.02, 0.01, -0.02, 0.00, 0.02];
    assemble_buffer_data::<Vertex, _>(cmd, renderer_entity, RwLock::new(vertex_buffer_data), 0);

    //  Particles
    let mut initial_particle_data = vec![0.0f32; (4 * NUM_PARTICLES) as usize];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let unif = rand::distributions::Uniform::new_inclusive(-1.0, 1.0);
    for particle_instance_chunk in initial_particle_data.chunks_mut(4) {
        particle_instance_chunk[0] = unif.sample(&mut rng); // posx
        particle_instance_chunk[1] = unif.sample(&mut rng); // posy
        particle_instance_chunk[2] = unif.sample(&mut rng) * 0.1; // velx
        particle_instance_chunk[3] = unif.sample(&mut rng) * 0.1; // vely
    }

    // Wrap initial data in an arc so both buffers share the same underlying source
    let initial_particle_data = Arc::new(RwLock::new(initial_particle_data));

    assemble_buffer_data::<FrontBuffer, _>(cmd, renderer_entity, initial_particle_data.clone(), 0);
    assemble_buffer_data::<BackBuffer, _>(cmd, renderer_entity, initial_particle_data, 0);

    // Uniforms
    let sim_param_data = [
        0.04f32, // deltaT
        0.1,     // rule1Distance
        0.025,   // rule2Distance
        0.025,   // rule3Distance
        0.02,    // rule1Scale
        0.05,    // rule2Scale
        0.005,   // rule3Scale
    ];

    assemble_buffer_data::<Uniform, _>(cmd, renderer_entity, RwLock::new(sim_param_data), 0);

    // Buffers
    antigen_wgpu::assemble_buffer::<Vertex>(
        cmd,
        renderer_entity,
        BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (6 * std::mem::size_of::<f32>()) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    antigen_wgpu::assemble_buffer::<Uniform>(
        cmd,
        renderer_entity,
        BufferDescriptor {
            label: Some("Simulation Parameter Buffer"),
            size: 7 * std::mem::size_of::<f32>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    antigen_wgpu::assemble_buffer::<FrontBuffer>(
        cmd,
        renderer_entity,
        BufferDescriptor {
            label: Some("Front Particle Buffer"),
            size: (4 * NUM_PARTICLES * std::mem::size_of::<f32>()) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    antigen_wgpu::assemble_buffer::<BackBuffer>(
        cmd,
        renderer_entity,
        BufferDescriptor {
            label: Some("Back Particle Buffer"),
            size: (4 * NUM_PARTICLES * std::mem::size_of::<f32>()) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );
}

pub fn prepare_schedule() -> ImmutableSchedule<Serial> {
    serial![
        parallel![
            antigen_wgpu::create_shader_modules_usage_system::<Compute>(),
            antigen_wgpu::create_shader_modules_usage_system::<Draw>(),
            antigen_wgpu::create_buffers_system::<Vertex>(),
            antigen_wgpu::create_buffers_system::<Uniform>(),
            antigen_wgpu::create_buffers_system::<FrontBuffer>(),
            antigen_wgpu::create_buffers_system::<BackBuffer>(),
        ],
        parallel![
            antigen_wgpu::buffer_write_system::<Vertex, RwLock<[f32; 6]>, [f32; 6]>(),
            antigen_wgpu::buffer_write_system::<Uniform, RwLock<[f32; 7]>, [f32; 7]>(),
            antigen_wgpu::buffer_write_system::<FrontBuffer, Arc<RwLock<Vec<f32>>>, Vec<f32>>(),
            antigen_wgpu::buffer_write_system::<BackBuffer, Arc<RwLock<Vec<f32>>>, Vec<f32>>(),
        ],
        boids_prepare_system(),
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    single![boids_render_system(
        AtomicUsize::new(0),
        ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32,
    )]
}
