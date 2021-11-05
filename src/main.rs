mod antigen_core;
mod antigen_wgpu;
mod antigen_winit;
mod demos;
mod threads;

pub use antigen_core::*;
pub use antigen_wgpu::*;
pub use antigen_winit::*;
pub use demos::*;
pub use threads::*;

pub use parking_lot;

fn main() -> ! {
    let world = ImmutableWorld::default();

    let hello_triangle_window = world.write().push(());
    let hello_triangle_renderer = world.write().push(());

    let cube_window = world.write().push(());
    let cube_renderer = world.write().push(());

    parallel![
        // WGPU
        antigen_wgpu::assemble_wgpu_system(),
        // Hello Triangle window
        antigen_winit::assemble_window_system((hello_triangle_window,)),
        antigen_wgpu::assemble_window_surface_system((hello_triangle_window,)),
        // Cube window
        antigen_winit::assemble_window_system((cube_window,)),
        antigen_wgpu::assemble_window_surface_system((cube_window,)),
        // Transform integration test
        demos::transform_integration::assemble_system(),
    ]
    .execute_and_flush(&world);

    // Renderers
    parallel![
        demos::wgpu_examples::hello_triangle::assemble_system((
            hello_triangle_renderer,
            hello_triangle_window,
        )),
        demos::wgpu_examples::cube::assemble_system((cube_renderer, cube_window))
    ]
    .execute_and_flush(&world);

    // Winit thread schedules
    let main_events_cleared_schedule = serial![
        create_window_surfaces_system(),
        parallel![
            crate::demos::wgpu_examples::hello_triangle::hello_triangle_prepare_system(),
            crate::demos::wgpu_examples::cube::cube_prepare_system(),
            redraw_windows_on_main_events_cleared_system()
        ]
    ];

    let redraw_events_cleared_schedule = serial![
        parallel![
            crate::demos::wgpu_examples::hello_triangle::hello_triangle_render_system(),
            crate::demos::wgpu_examples::cube::cube_render_system(),
        ],
        submit_command_buffers_system(),
        surface_texture_present_system()
        surface_texture_view_drop_system()
    ];

    // Spawn threads and enter main loop
    std::thread::spawn(game_thread(world.clone()));
    winit_thread(
        world,
        main_events_cleared_schedule,
        redraw_events_cleared_schedule,
    )
}
