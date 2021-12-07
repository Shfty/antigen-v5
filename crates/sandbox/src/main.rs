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
//       [✓] Skybox
//          [✓] First working implementation
//          [✓] Window resize support
//          [✓] Fix thread hang when dragging-sizing to zero
//          [✓] Upload buffer data using staging belt
//          [✓] Input handling
//              Will need to refactor camera into a component
//       [✓] Shadow
//          [✓] Implement objects as legion entities
//              [✓] Create in module
//              [✓] Write to uniform buffer via component
//          [✓] Fix runtime errors
//          [✓] Fix lifetime issues with RenderPass
//              * Render passes need to be singular, not per-object
//              * Vertex / index buffer read guards need to exist before render pass
//              * Fetch cube / plane vertex / index buffers at top of render system
//              * Use tags or enums to determine which objects are interested in which buffer
//          [✓] Debug low-res shadow output
//          [✓] Implement resize handling
//          [ ] Structural polish (ECS uniform upload, on-change light updates)
//          [ ] Convert to wgpu-native coordinate system
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
// TODO: [>] Boilerplate reduction for reading and unwrapping RwLock<LazyComponent::Ready>
//
// TODO: [✓] Refactor ChangedFlag as a wrapper instead of a separate component
//       Can implement its methods in a trait and use Deref to tag them onto an existing type,
//          with or without a usage flag
//       Less components means less boilerplate in user code,
//       but introduces more pressure to typedef complex components
//
// TODO: [ ] Factor out AddComponentWithUsageFlag in favor of AsUsage
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
//       [✓] Assemly implementation
//       [✓] First renderer integration
//       [✓] Compose thread-local functions similar to schedules
//       [✓] Avoid using thread-local functions in StagingBelt consumers
//       [ ] Figure out how best to handle reclaiming / device polling
//           Currently uploading everything at once and polling in wait mode to avoid futures
//           Ideally should use poll mode, use futures to block associated render systems?
//
// TODO: Reimplement map demo scene
//       [ ] Renderer
//       [ ] Physics integration
//       [ ] Map texture loading
//       [ ] Basic first-person character control
//       [ ] Render WGPU demos to texture, tie to named map texture for display in-world
//
// TODO: Investigate frame drops on window events
//       Ex. Obvious framerate dip when moving mouse over bunnymark window in release mode
//       Seems to have changed since implementing cursor move handling - needs more investigation
//       [ ] Test rendering inside of RedrawRequested instead of RedrawEventsCleared
//       [ ] Test rendering at the end of MainEventsCleared

mod demos;

pub use demos::*;

use antigen_core::*;
use antigen_wgpu::wgpu::{DeviceDescriptor, Features, Limits};

const GAME_TICK_DURATION: std::time::Duration = std::time::Duration::from_secs(1);

fn main() -> ! {
    //tracing_subscriber::fmt::fmt().pretty().init();

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
    // Enter winit event loop
    antigen_winit::winit::event_loop::EventLoop::new().run(antigen_winit::wrap_event_loop(
        world,
        antigen_winit::winit_event_handler(antigen_wgpu::winit_event_handler(
            demos::wgpu_examples::winit_event_handler(antigen_winit::winit_event_terminator()),
        )),
    ))
}
