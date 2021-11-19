use crate::{parallel, ImmutableSchedule, Parallel};

pub mod boids;
pub mod cube;
pub mod hello_triangle;
pub mod bunnymark;
pub mod msaa_line;
pub mod conservative_raster;
pub mod mipmap;
pub mod texture_arrays;

pub fn assemble_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        /*
        hello_triangle::assemble_system(),
        cube::assemble_system(),
        boids::assemble_system(),
        bunnymark::assemble_system(),
        msaa_line::assemble_system(),
        conservative_raster::assemble_system(),
        mipmap::assemble_system(),
        */
        texture_arrays::assemble_system(),
    ]
}

pub fn prepare_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        /*
        hello_triangle::prepare_schedule(),
        cube::prepare_schedule(),
        boids::prepare_schedule(),
        bunnymark::prepare_schedule(),
        msaa_line::prepare_schedule(),
        conservative_raster::prepare_schedule(),
        mipmap::prepare_schedule(),
        */
        texture_arrays::prepare_schedule(),
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        /*
        hello_triangle::render_schedule(),
        cube::render_schedule(),
        boids::render_schedule(),
        bunnymark::render_schedule(),
        msaa_line::render_schedule(),
        conservative_raster::render_schedule(),
        mipmap::render_schedule(),
        */
        texture_arrays::render_schedule(),
    ]
}

pub fn surface_resize_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        /*
        cube::cube_resize_system()
        msaa_line::msaa_line_resize_system()
        conservative_raster::conservative_raster_resize_system()
        mipmap::mipmap_resize_system(),
        */
    ]
}

