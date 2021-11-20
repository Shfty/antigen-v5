// TODO: Implement remaining WGPU demos using ECS pattern
//       [✓] Boids
//       [✓] Bunnymark
//       [ ] Capture(?)
//       [✓] Conservative Raster
//       [✓] Cube
//       [ ] Hello Compute
//           Need to account for non-window GPU work
//       [✓] Hello Triangle
//       [ ] Hello Windows
//           Not interesting from a GPU standpoint,
//           but useful to demonstrate ECS approach
//       [✓] Mipmap
//       [✓] MSAA Line
//           [✓] MSAA rendering
//           [✓] Recreate framebuffer on resize
//           [✓] Render bundle recreation
//       [>] Skybox
//          [✓] First working implementation
//          [✓] Window resize support
//          [✓] Fix thread hang when dragging-sizing to zero
//          [ ] Upload buffer data using staging belt
//          [ ] Input handling
//              Will need to refactor camera into a component
//       [ ] Shadow
//       [✓] Texture Arrays
//          [✓] Base implementation
//          [✓] Fix red texture not rendering
//       [ ] Water
//
// TODO: Factor mipmap generation out of Mipmap renderer and into a generalized system
//       Generator is effectively a renderer in and of itself
//       Should be able to give TextureComponent a sibling GenerateMipmaps component,
//       have everything be automatic from there
//
// TODO: Boilerplate reduction for reading and unwrapping RwLock<LazyComponent::Ready>
//
// TODO: Figure out a better way to assemble Usage<U, ChangedFlag<T>>
//       add_component_with_usage_and_changed_flag?
//
// TODO: Improve WindowEventComponent
//       Split into discrete components?
//
// TODO: Investigate Encoder::copy_buffer_to_texture
//       What are its characteristics versus Queue::write_texture?
//       Is Queue::write_texture just a wrapper for it?
//       Is it worth writing an alternate texture writing system that uses it?
//
// TODO: StagingBelt integration
//       Requires a command encoder, but doesn't have to be coupled to drawing
//       Treat as its own 'data upload' step that occurs before draw
//       StagingBelt itself isn't Send + Sync, will need to be thread-local
//       Shouldn't be recreating it on every upload - point is efficient buffer reuse
//       [✓] Initial manager + component + system implementation
//       [ ] Assemly implementation
//       [ ] First renderer integration
//       [ ] Figure out how best to handle reclaiming / device polling
//           Currently uploading everything at once and polling in wait mode to avoid futures
//           Ideally should use poll mode, use futures to block associated render system
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
use antigen_wgpu::{
    wgpu::{DeviceDescriptor, Features, Limits},
    StagingBeltManager,
};
use antigen_winit::winit::event::{Event, WindowEvent};

const GAME_TICK_DURATION: std::time::Duration = std::time::Duration::from_secs(1);

fn main() -> ! {
    // Create world
    let world = ImmutableWorld::default();

    // Assemble winit backend
    antigen_winit::assemble_winit_backend(&mut world.write());

    // Assemble WGPU backend
    antigen_wgpu::assemble_wgpu_entity_from_env(
        &mut world.write(),
        &DeviceDescriptor {
            label: None,
            features: Features::default()
                | Features::POLYGON_MODE_LINE
                | Features::CONSERVATIVE_RASTERIZATION
                | Features::TIMESTAMP_QUERY
                | Features::PIPELINE_STATISTICS_QUERY
                | Features::SPIRV_SHADER_PASSTHROUGH
                | Features::TEXTURE_BINDING_ARRAY
                | (
                    // Features for texture arrays
                    Features::PUSH_CONSTANTS
                        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                        | Features::UNSIZED_BINDING_ARRAY
                )
                | (
                    // Features for skybox texture compression
                    Features::TEXTURE_COMPRESSION_BC
                    //    | Features::TEXTURE_COMPRESSION_ETC2
                    //    | Features::TEXTURE_COMPRESSION_ASTC_LDR
                ),
            limits: Limits {
                max_push_constant_size: 4,
                max_texture_dimension_2d: 4096,
                ..Limits::downlevel_defaults()
            },
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
    let mut staging_belt_manager = StagingBeltManager::new();

    // Resets dirty flags that should only remain active for one frame
    let reset_dirty_flags_schedule = parallel![
        antigen_winit::reset_resize_window_dirty_flags_system(),
        antigen_wgpu::reset_surface_config_changed_system(),
    ];

    // Create winit event schedules
    let mut main_events_cleared_schedule = serial![
        antigen_winit::window_title_system(),
        antigen_winit::window_request_redraw_schedule(),
        serial![
            antigen_wgpu::window_surfaces_schedule(),
            demos::wgpu_examples::surface_resize_schedule(),
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

    let mut post_redraw_events_cleared_schedule = serial![antigen_wgpu::device_poll_system(
        antigen_wgpu::wgpu::Maintain::Wait
    ),];

    let mut window_resized_schedule = serial![
        antigen_winit::resize_window_system(),
        demos::wgpu_examples::surface_resize_schedule(),
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
                antigen_wgpu::staging_belt_write_thread_local::<(), RwLock<Vec<u8>>, Vec<u8>>(
                    &world.read(),
                    &mut staging_belt_manager,
                );
                antigen_wgpu::staging_belt_finish_thread_local::<()>(
                    &world.read(),
                    &mut staging_belt_manager,
                );
                redraw_events_cleared_schedule.execute(world);
                antigen_wgpu::staging_belt_recall_thread_local::<()>(
                    &world.read(),
                    &mut staging_belt_manager,
                );
                post_redraw_events_cleared_schedule.execute(world);
            }
            _ => (),
        },
    ))
}
