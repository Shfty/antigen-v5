use antigen_core::{ImmutableWorld, ReadWriteLock};
use antigen_wgpu::StagingBeltManager;
use antigen_winit::{
    winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoopWindowTarget},
    },
    EventLoopHandler,
};
use legion::World;

use crate::{parallel, ImmutableSchedule, Parallel};

pub mod boids;
pub mod bunnymark;
pub mod conservative_raster;
pub mod cube;
pub mod hello_triangle;
pub mod mipmap;
pub mod msaa_line;
pub mod shadow;
pub mod skybox;
pub mod texture_arrays;

pub fn assemble_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        hello_triangle::assemble_system(),
        cube::assemble_system(),
        boids::assemble_system(),
        bunnymark::assemble_system(),
        msaa_line::assemble_system(),
        conservative_raster::assemble_system(),
        mipmap::assemble_system(),
        texture_arrays::assemble_system(),
        skybox::assemble_system(),
        //shadow::assemble_system(),
    ]
}

fn prepare_thread_local(world: &World, staging_belt_manager: &mut StagingBeltManager) {
    skybox::prepare_thread_local(world, staging_belt_manager);
}

pub fn winit_event_handler<T>(mut f: impl EventLoopHandler<T>) -> impl EventLoopHandler<T> {
    let mut staging_belt_manager = StagingBeltManager::new();

    let mut prepare_schedule = parallel![
        hello_triangle::prepare_schedule(),
        cube::prepare_schedule(),
        boids::prepare_schedule(),
        bunnymark::prepare_schedule(),
        msaa_line::prepare_schedule(),
        conservative_raster::prepare_schedule(),
        mipmap::prepare_schedule(),
        texture_arrays::prepare_schedule(),
        skybox::prepare_schedule(),
        //shadow::prepare_schedule(),
    ];

    let mut render_schedule = parallel![
        hello_triangle::render_schedule(),
        cube::render_schedule(),
        boids::render_schedule(),
        bunnymark::render_schedule(),
        msaa_line::render_schedule(),
        conservative_raster::render_schedule(),
        mipmap::render_schedule(),
        texture_arrays::render_schedule(),
        skybox::render_schedule(),
        //shadow::render_schedule(),
    ];

    let mut surface_resize_schedule = parallel![
        cube::cube_resize_system()
        msaa_line::msaa_line_resize_system()
        conservative_raster::conservative_raster_resize_system()
        mipmap::mipmap_resize_system(),
        skybox::skybox_resize_system(),
    ];

    let mut keyboard_event_schedule = parallel![
        bunnymark::keyboard_event_schedule(),
        msaa_line::keyboard_event_schedule(),
    ];

    let mut window_cursor_moved_schedule = parallel![skybox::cursor_moved_schedule(),];

    move |world: &ImmutableWorld,
          event: Event<'static, T>,
          event_loop_window_target: &EventLoopWindowTarget<T>,
          control_flow: &mut ControlFlow| {
        match &event {
            Event::MainEventsCleared => {
                antigen_wgpu::create_staging_belt_thread_local(
                    &world.read(),
                    &mut staging_belt_manager,
                );
                surface_resize_schedule.execute(world);
                prepare_schedule.execute(world);
                prepare_thread_local(&world.read(), &mut staging_belt_manager);
                antigen_wgpu::staging_belt_flush_thread_local(&world.read(), &mut staging_belt_manager);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(_) => {
                    surface_resize_schedule.execute(world);
                }
                WindowEvent::KeyboardInput { .. } => keyboard_event_schedule.execute(world),
                WindowEvent::CursorMoved { .. } => window_cursor_moved_schedule.execute(world),
                _ => (),
            },
            Event::RedrawEventsCleared => {
                antigen_wgpu::staging_belt_finish_thread_local(
                    &world.read(),
                    &mut staging_belt_manager,
                );

                render_schedule.execute(world);

                antigen_wgpu::submit_and_present_schedule().execute(world);
                antigen_wgpu::staging_belt_recall_thread_local(
                    &world.read(),
                    &mut staging_belt_manager,
                );
            }
            _ => (),
        }

        f(world, event, event_loop_window_target, control_flow);
    }
}
