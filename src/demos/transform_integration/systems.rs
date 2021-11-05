use crate::ReadWriteLock;
use super::{Position, Rotation, LinearVelocity, AngularVelocity};

// Integrate position by linear velocity
#[legion::system(par_for_each)]
pub fn integrate_position(linear_velocity: &LinearVelocity, position: &Position) {
    let linear_velocity = linear_velocity.read();
    let mut position = position.write();

    position.0 += linear_velocity.0;
    position.1 += linear_velocity.1;
    position.2 += linear_velocity.2;
}

// Integrate rotation by angular velocity
#[legion::system(par_for_each)]
pub fn integrate_rotation(angular_velocity: &AngularVelocity, rotation: &Rotation) {
    let angular_velocity = angular_velocity.read();
    let mut rotation = rotation.write();

    *rotation += *angular_velocity;
}

// Print position components
#[legion::system(par_for_each)]
pub fn print_position(position: &Position) {
    println!("Position: {:#?}", position.read());
}
