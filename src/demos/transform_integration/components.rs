use parking_lot::RwLock;

use crate::impl_read_write_lock;

// Position
#[derive(Debug, Default)]
pub struct Position(RwLock<(f32, f32, f32)>);

impl Position {
    pub fn new(position: (f32, f32, f32)) -> Self {
        Position(RwLock::new(position))
    }
}

impl_read_write_lock!(Position, 0, (f32, f32, f32));

// Rotation
#[derive(Debug, Default)]
pub struct Rotation(RwLock<f32>);

impl Rotation {
    pub fn new(rotation: f32) -> Self {
        Rotation(RwLock::new(rotation))
    }
}

impl_read_write_lock!(Rotation, 0, f32);

// Linear velocity
#[derive(Debug, Default)]
pub struct LinearVelocity(RwLock<(f32, f32, f32)>);

impl_read_write_lock!(LinearVelocity, 0, (f32, f32, f32));

impl LinearVelocity {
    pub fn new(velocity: (f32, f32, f32)) -> Self {
        LinearVelocity(RwLock::new(velocity))
    }
}

// Angular velocity
#[derive(Debug, Default)]
pub struct AngularVelocity(RwLock<f32>);

impl_read_write_lock!(AngularVelocity, 0, f32);

impl AngularVelocity {
    pub fn new(angular_velocity: f32) -> Self {
        AngularVelocity(RwLock::new(angular_velocity))
    }
}

