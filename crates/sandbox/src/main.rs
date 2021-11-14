// TODO: Factor out Texels, MeshIndices, etc
//       Implement with usage flags instead
//
// TODO: Support for creating buffers via init
//       Could use a two-variant enum in the component to hold either
//       Alternately, use descriptors as components
//          This is probably better, since it separates it and allows for a changed flag
//
// TODO: Implement remaining WGPU demos using ECS pattern
//       [✓] Boids
//       [✓] Bunnymark
//       [ ] Capture(?)
//       [ ] Conservative Raster
//       [✓] Cube
//       [ ] Hello Compute
//       [✓] Hello Triangle
//       [ ] Hello Windows(?)
//       [ ] Hello(?)
//       [ ] Mipmap
//       [✓] MSAA Line
//           [✓] MSAA rendering
//           [✓] Recreate framebuffer on resize
//           [✓] Render bundle recreation
//       [ ] Shadow
//       [ ] Texture Arrays
//       [ ] Water
//
// TODO: Refactor Cube renderer to use original vertex layout
//
// TODO: Figure out a better way to assemble Usage<U, ChangedFlag<T>>
//
// TODO: Improve WindowEventComponent
//
// TODO: Reimplement map renderer
//
// TODO: Investigate frame drops on window events
//       Ex. Obvious framerate dip when moving mouse over bunnymark window in release mode
//       [ ] Test rendering inside of RedrawRequested instead of RedrawEventsCleared
//       [ ] Test rendering at the end of MainEventsCleared

mod demos;

pub use demos::*;

use antigen_core::*;
use antigen_wgpu::wgpu::{DeviceDescriptor, Features, Limits};
use antigen_winit::winit::event::{Event, WindowEvent};

const GAME_TICK_DURATION: std::time::Duration = std::time::Duration::from_secs(1);

fn main() -> ! {
    // Create world
    let world = ImmutableWorld::default();

    // Assemble winit backend
    antigen_winit::assemble_winit_entity(&mut world.write());

    // Assemble WGPU backend
    antigen_wgpu::assemble_wgpu_entity_from_env(
        &mut world.write(),
        &DeviceDescriptor {
            label: None,
            features: Features::default() | Features::POLYGON_MODE_LINE,
            limits: Limits::downlevel_defaults(),
        },
        None,
        None,
    );

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
            demos::wgpu_examples::cube::cube_resize_system()
            demos::wgpu_examples::msaa_line::msaa_line_resize_system()
        ]
    };

    // Resets dirty flags that should only remain active for one frame
    let reset_dirty_flags_schedule =
        parallel![
            antigen_winit::reset_resize_window_dirty_flags_system(),
            antigen_wgpu::reset_surface_config_changed_system(),
        ];

    // Create winit event schedules
    let mut main_events_cleared_schedule = serial![
        antigen_winit::window_title_system(),
        antigen_winit::window_request_redraw_schedule(),
        serial![
            antigen_wgpu::window_surfaces_schedule(),
            surface_resize_schedule(),
        ],
        parallel![
            crate::demos::wgpu_examples::prepare_schedule(),
            reset_dirty_flags_schedule,
        ],
    ];

    let mut redraw_requested_schedule = single![antigen_wgpu::surface_textures_views_system()];

    let mut redraw_events_cleared_schedule = serial![
        crate::demos::wgpu_examples::render_schedule(),
        antigen_wgpu::submit_and_present_schedule(),
    ];

    let mut window_resized_schedule = serial![
        antigen_winit::resize_window_system(),
        surface_resize_schedule(),
    ];
    let mut window_keyboard_event_schedule = parallel![
        crate::demos::wgpu_examples::bunnymark::keyboard_event_schedule(),
        crate::demos::wgpu_examples::msaa_line::keyboard_event_schedule(),
    ];
    let mut window_close_requested_schedule = single![antigen_winit::close_window_system()];

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
                WindowEvent::KeyboardInput { .. } => window_keyboard_event_schedule.execute(world),
                _ => (),
            },
            Event::RedrawEventsCleared => {
                redraw_events_cleared_schedule.execute(world);
            }
            _ => (),
        },
    ))
}
