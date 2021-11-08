// TODO: Implement remaining WGPU demos using ECS pattern
//       [✓] Boids
//       [ ] Bunnymark
//       [ ] Capture(?)
//       [ ] Conservative Raster
//       [✓] Cube
//       [ ] Hello Compute
//       [✓] Hello Triangle
//       [ ] Hello Windows(?)
//       [ ] Hello(?)
//       [ ] Mipmap
//       [ ] MSAA Line
//       [ ] Shadow
//       [ ] Texture Arrays
//       [ ] Water
//
// TODO: Reimplement map renderer

mod demos;

pub use demos::*;

use antigen_core::*;
use antigen_wgpu::wgpu::{self, Backends, DeviceDescriptor, Features, Instance, Limits};
use antigen_winit::winit::event::{Event, WindowEvent};

const GAME_TICK_DURATION: std::time::Duration = std::time::Duration::from_secs(1);

fn main() -> ! {
    // Create world
    let world = ImmutableWorld::default();

    // Init WGPU backend
    let backend_bits = wgpu::util::backend_bits_from_env().unwrap_or(Backends::PRIMARY);

    let instance = Instance::new(backend_bits);
    println!("Created WGPU instance: {:#?}\n", instance);

    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance,
        backend_bits,
        None,
    ))
    .expect("Failed to acquire WGPU adapter");

    let adapter_info = adapter.get_info();
    println!("Acquired WGPU adapter: {:#?}\n", adapter_info);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &DeviceDescriptor {
            label: None,
            features: Features::default() | Features::POLYGON_MODE_LINE,
            limits: Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap();

    println!("Acquired WGPU device: {:#?}\n", device);
    println!("Acquired WGPU queue: {:#?}\n", queue);

    // Assemble winit / wgpu backend entities
    antigen_winit::assemble_winit_entity(&mut world.write());
    antigen_wgpu::assemble_wgpu_entity(&mut world.write(), instance, adapter, device, queue);

    // Assemble modules
    single![demos::transform_integration::assemble_system()].execute_and_flush(&world);
    demos::wgpu_examples::assemble_schedule().execute_and_flush(&world);

    // Spawn threads
    std::thread::spawn(game_thread(world.clone()));
    winit_thread(world);
}

pub fn game_thread(world: ImmutableWorld) -> impl Fn() {
    move || {
        // Crate schedule
        let mut schedule = serial![
            crate::demos::transform_integration::integrate_schedule(),
            crate::demos::transform_integration::print_schedule()
        ];

        // Run schedule in loop
        antigen_util::spin_loop(GAME_TICK_DURATION, || schedule.execute(&world))
    }
}

pub fn winit_thread(world: ImmutableWorld) -> ! {
    // Reacts to changes in surface size
    // Runs on main events cleared and window resize
    let surface_resize_schedule = || {
        serial![
            antigen_wgpu::surface_size_system()
            antigen_wgpu::surface_texture_size_system()
            demos::wgpu_examples::cube::cube_resize_system()
        ]
    };

    // Resets dirty flags that should only remain active for one frame
    let reset_dirty_flags_schedule = parallel![
        antigen_winit::reset_resize_window_dirty_flags_system(),
        antigen_wgpu::reset_surface_size_dirty_flag_system(),
        antigen_wgpu::reset_texture_size_dirty_flag_system(),
    ];

    // Create winit event schedules
    let mut main_events_cleared_schedule = serial![
        antigen_winit::window_title_system(),
        antigen_winit::window_request_redraw_schedule(),
        parallel![
            antigen_wgpu::create_window_surfaces_schedule(),
            surface_resize_schedule(),
        ],
        parallel![
            crate::demos::wgpu_examples::prepare_schedule(),
            reset_dirty_flags_schedule,
        ],
    ];

    let mut redraw_requested_schedule = single![antigen_wgpu::surface_textures_views_system()];
    let mut window_resized_schedule = serial![
        antigen_winit::resize_window_system(),
        surface_resize_schedule(),
    ];
    let mut window_close_requested_schedule = single![antigen_winit::close_window_system()];

    let mut redraw_events_cleared_schedule = serial![
        crate::demos::wgpu_examples::render_schedule(),
        antigen_wgpu::submit_and_present_schedule(),
    ];

    // Enter winit event loop
    antigen_winit::winit::event_loop::EventLoop::new().run(antigen_winit::event_loop_wrapper(
        world,
        move |world, event, _, _control_flow| match &event {
            Event::MainEventsCleared => {
                main_events_cleared_schedule.execute(world);
            }
            Event::RedrawRequested(_) => {
                redraw_requested_schedule.execute(world);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(_) => {
                    window_resized_schedule.execute(world);
                }
                WindowEvent::CloseRequested => {
                    window_close_requested_schedule.execute(world);
                }
                _ => (),
            },
            Event::RedrawEventsCleared => {
                redraw_events_cleared_schedule.execute(world);
            }
            _ => (),
        },
    ))
}
