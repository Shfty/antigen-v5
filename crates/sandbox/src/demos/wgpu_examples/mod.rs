use crate::{parallel, ImmutableSchedule, Parallel};

pub mod boids;
pub mod cube;
pub mod hello_triangle;
pub mod bunnymark;
pub mod msaa_line;
pub mod conservative_raster;

pub fn assemble_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        hello_triangle::assemble_system(),
        cube::assemble_system(),
        boids::assemble_system(),
        bunnymark::assemble_system(),
        msaa_line::assemble_system(),
        conservative_raster::assemble_system(),
    ]
}

pub fn prepare_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        hello_triangle::prepare_schedule(),
        cube::prepare_schedule(),
        boids::prepare_schedule(),
        bunnymark::prepare_schedule(),
        msaa_line::prepare_schedule(),
        conservative_raster::prepare_schedule(),
    ]
}

pub fn render_schedule() -> ImmutableSchedule<Parallel> {
    parallel![
        hello_triangle::render_schedule(),
        cube::render_schedule(),
        boids::render_schedule(),
        bunnymark::render_schedule(),
        msaa_line::render_schedule(),
        conservative_raster::render_schedule(),
    ]
}

