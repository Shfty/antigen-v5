use std::time::Duration;

use crate::{parallel, serial, ImmutableWorld};

pub fn game_thread(world: ImmutableWorld) -> impl Fn() {
    move || {
        #[rustfmt::skip]
        let mut schedule = serial![
            parallel![
                crate::demos::transform_integration::integrate_position_system(),
                crate::demos::transform_integration::integrate_rotation_system(),
            ],
            crate::demos::transform_integration::print_position_system()
        ];

        loop {
            schedule.execute(&world);
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}
