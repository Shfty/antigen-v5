mod components;
mod systems;

use std::sync::{atomic::AtomicUsize, Arc};

use antigen_winit::{AssembleWinit, RedrawUnconditionally};
pub use components::*;
pub use systems::*;

use antigen_core::{AddIndirectComponent, Changed, ImmutableSchedule, RwLock, Serial, Single, parallel, serial, single};

use antigen_wgpu::{
    wgpu::{
        util::BufferInitDescriptor, BufferAddress, BufferDescriptor, BufferUsages, Device,
        ShaderModuleDescriptor, ShaderSource,
    },
    AssembleWgpu, RenderAttachmentTextureView, SurfaceConfigurationComponent,
};

use rand::{distributions::Distribution, SeedableRng};

const NUM_PARTICLES: usize = 1500;
const PARTICLES_PER_GROUP: usize = 64;

#[legion::system]
#[read_component(Device)]
pub fn assemble(cmd: &mut legion::systems::CommandBuffer) {
    let window_entity = cmd.push(());
    let renderer_entity = cmd.push(());

    // Assemble window
    cmd.assemble_winit_window(window_entity);
    cmd.assemble_wgpu_window_surface(window_entity);

    // Add title to window
    cmd.assemble_winit_window_title(window_entity, "Boids");

    // Redraw the window unconditionally
    cmd.add_component(window_entity, RedrawUnconditionally);

    // Renderer
    cmd.add_component(renderer_entity, Boids);

    cmd.assemble_wgpu_render_pipeline(renderer_entity);
    cmd.assemble_wgpu_compute_pipeline(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<FrontBuffer>(renderer_entity);
    cmd.assemble_wgpu_bind_group_with_usage::<BackBuffer>(renderer_entity);
    cmd.assemble_wgpu_command_buffers(renderer_entity);

    cmd.add_indirect_component::<Changed<SurfaceConfigurationComponent>>(renderer_entity, window_entity);
    cmd.add_indirect_component::<RenderAttachmentTextureView>(renderer_entity, window_entity);

    // Shaders
    cmd.assemble_wgpu_shader_with_usage::<Compute>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("compute.wgsl"))),
        },
    );

    cmd.assemble_wgpu_shader_with_usage::<Draw>(
        renderer_entity,
        ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("draw.wgsl"))),
        },
    );

    // Buffer data
    // Vertices
    const VERTEX_BUFFER_DATA: [f32; 6] = [-0.01f32, -0.02, 0.01, -0.02, 0.00, 0.02];

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

    cmd.assemble_wgpu_buffer_data_with_usage::<FrontBuffer, _>(
        renderer_entity,
        initial_particle_data.clone(),
        0,
    );
    cmd.assemble_wgpu_buffer_data_with_usage::<BackBuffer, _>(
        renderer_entity,
        initial_particle_data,
        0,
    );

    // Uniforms
    const SIM_PARAM_DATA: [f32; 7] = [
        0.04f32, // deltaT
        0.1,     // rule1Distance
        0.025,   // rule2Distance
        0.025,   // rule3Distance
        0.02,    // rule1Scale
        0.05,    // rule2Scale
        0.005,   // rule3Scale
    ];

    // Buffers
    cmd.assemble_wgpu_buffer_init_with_usage::<Vertex>(
        renderer_entity,
        BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&VERTEX_BUFFER_DATA),
        },
    );

    cmd.assemble_wgpu_buffer_init_with_usage::<Uniform>(
        renderer_entity,
        BufferInitDescriptor {
            label: Some("Simulation Parameter Buffer"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(&SIM_PARAM_DATA),
        },
    );

    cmd.assemble_wgpu_buffer_with_usage::<FrontBuffer>(
        renderer_entity,
        BufferDescriptor {
            label: Some("Front Particle Buffer"),
            size: (4 * NUM_PARTICLES * std::mem::size_of::<f32>()) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );

    cmd.assemble_wgpu_buffer_with_usage::<BackBuffer>(
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
            antigen_wgpu::create_buffers_init_system::<Vertex>(),
            antigen_wgpu::create_buffers_init_system::<Uniform>(),
            antigen_wgpu::create_buffers_system::<FrontBuffer>(),
            antigen_wgpu::create_buffers_system::<BackBuffer>(),
        ],
        parallel![
            antigen_wgpu::buffer_write_system::<FrontBuffer, Arc<RwLock<Vec<f32>>>, Vec<f32>>(),
            antigen_wgpu::buffer_write_system::<BackBuffer, Arc<RwLock<Vec<f32>>>, Vec<f32>>(),
        ],
        boids_prepare_system(),
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Single> {
    println!("Allocating render schedule");
    single![boids_render_system(
        AtomicUsize::new(0),
        ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32,
    )]
}
