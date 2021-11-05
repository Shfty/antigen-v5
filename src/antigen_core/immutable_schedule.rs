use std::marker::PhantomData;

use legion::World;
use legion::{
    storage::ComponentTypeId,
    systems::{ParallelRunnable, Runnable, SystemId, UnsafeResources},
    world::ArchetypeAccess,
};

use crate::{ImmutableWorld, ReadWriteLock};
use rayon::prelude::*;

pub enum Serial {}
pub enum Parallel {}

pub trait RunSchedule {
    unsafe fn run_unsafe(
        runnables: &mut Vec<Box<dyn ParallelRunnable>>,
        world: &World,
        resources: &UnsafeResources,
    );
}

impl RunSchedule for Serial {
    unsafe fn run_unsafe(
        runnables: &mut Vec<Box<dyn ParallelRunnable>>,
        world: &World,
        resources: &UnsafeResources,
    ) {
        for runnable in runnables {
            runnable.run_unsafe(world, resources);
        }
    }
}

impl RunSchedule for Parallel {
    unsafe fn run_unsafe(
        runnables: &mut Vec<Box<dyn ParallelRunnable>>,
        world: &World,
        resources: &UnsafeResources,
    ) {
        runnables
            .par_iter_mut()
            .map(|system| system.run_unsafe(world, resources))
            .for_each(drop);
    }
}

/// A schedule designed for use with interior-mutable worlds.
///
/// ImmutableSchedule<Serial> runs its systems in sequence.
/// ImmutableSchedule<Parallel> runs its systems in parallel.
///
/// Both of these implement [`Runnable`] and [`ParallelRunnable`], so can be nested arbitrarily.
///
/// Limitations:
/// * [`Resources`] cannot be used
///     * [`Resources`] is !Send + !Sync, and requires exclusive mutable access.
///     * [`ImmutableSchedule`] relies on [`Runnable::run_unsafe`], which requires [`UnsafeResources`].
///     * [`UnsafeResources`] has no methods for accessing or modifying its underlying data.
///     * Instead, use singleton entities.
///     * If thread safety is an issue, it's recommended to
///
/// * [`ImmutableSchedule`] does not expose a [`CommandBuffer`] via the [`Runnable`] trait.
///     * [`Runnable`] can only expose a single [`CommandBuffer`].
///     * [`CommandBuffer`] exposes no methods for accessing its underlying data.
///     * Thus, multiple nested [`CommandBuffer`]s can't be merged.
///     * To work around this buffers are flushed manually via [`ImmutableSchedule::flush`],
///       which can only access the [`Runnable::command_buffer_mut`]s of its direct children.
///     * In short, this means only top-level systems may add or remove components and entities
///       from the world.

pub struct ImmutableSchedule<T> {
    system_id: SystemId,
    runnables: Vec<Box<dyn ParallelRunnable>>,
    reads_ids: Vec<ComponentTypeId>,
    archetypes: ArchetypeAccess,
    _phantom: PhantomData<T>,
}

impl<T> Default for ImmutableSchedule<T> {
    fn default() -> Self {
        ImmutableSchedule {
            system_id: SystemId::from("ImmutableSchedule"),
            runnables: Default::default(),
            reads_ids: Default::default(),
            archetypes: ArchetypeAccess::Some(Default::default()),
            _phantom: Default::default(),
        }
    }
}

impl ImmutableSchedule<()> {
    pub fn serial() -> ImmutableSchedule<Serial> {
        ImmutableSchedule {
            system_id: SystemId::from("ImmutableSchedule<Serial>"),
            ..Default::default()
        }
    }

    pub fn parallel() -> ImmutableSchedule<Parallel> {
        ImmutableSchedule {
            system_id: SystemId::from("ImmutableSchedule<Parallel>"),
            ..Default::default()
        }
    }
}

impl<T> ImmutableSchedule<T> {
    pub fn add_system<S: ParallelRunnable + 'static>(mut self, system: S) -> Self {
        let (writes_resources, writes_components) = system.writes();

        if writes_resources.len() > 0 {
            panic!("System requires mutable access to resources")
        }

        if writes_components.len() > 0 {
            panic!("System requires mutable access to components")
        }

        match system.accesses_archetypes() {
            ArchetypeAccess::All => self.archetypes = ArchetypeAccess::All,
            ArchetypeAccess::Some(in_archetypes) => match &mut self.archetypes {
                ArchetypeAccess::All => (),
                ArchetypeAccess::Some(archetypes) => archetypes.union_with(in_archetypes),
            },
        }

        self.reads_ids.extend(system.reads().1);
        self.runnables.push(Box::new(system));
        self
    }

    /// Flush system command buffers
    fn flush(&mut self, world: &ImmutableWorld) {
        let mut resources = legion::systems::Resources::default();
        for system in &mut self.runnables {
            let world_id = world.read().id();
            if let Some(command_buffer) = system.command_buffer_mut(world_id) {
                if !command_buffer.is_empty() {
                    command_buffer.flush(&mut world.write(), &mut resources);
                }
            }
        }
    }
}

impl<T: RunSchedule> ImmutableSchedule<T> {
    pub fn execute(&mut self, world: &ImmutableWorld) {
        self.prepare(&world.read());
        unsafe { self.run_unsafe(&world.read(), &mut Default::default()) };
    }

    pub fn execute_and_flush(&mut self, world: &ImmutableWorld) {
        self.execute(world);
        self.flush(world);
    }
}

impl<T: RunSchedule> Runnable for ImmutableSchedule<T> {
    fn name(&self) -> Option<&legion::systems::SystemId> {
        Some(&self.system_id)
    }

    fn reads(
        &self,
    ) -> (
        &[legion::systems::ResourceTypeId],
        &[legion::storage::ComponentTypeId],
    ) {
        (&[], &self.reads_ids)
    }

    fn writes(
        &self,
    ) -> (
        &[legion::systems::ResourceTypeId],
        &[legion::storage::ComponentTypeId],
    ) {
        (&[], &[])
    }

    fn prepare(&mut self, world: &World) {
        self.runnables
            .par_iter_mut()
            .map(|system| system.prepare(&world))
            .for_each(drop);
    }

    fn accesses_archetypes(&self) -> &legion::world::ArchetypeAccess {
        &self.archetypes
    }

    unsafe fn run_unsafe(&mut self, world: &World, resources: &UnsafeResources) {
        T::run_unsafe(&mut self.runnables, world, resources)
    }

    fn command_buffer_mut(
        &mut self,
        _world: legion::world::WorldId,
    ) -> Option<&mut legion::systems::CommandBuffer> {
        None
    }
}

/// Construct an [`ImmutableSchedule<Serial>`] from a set of [`ParallelRunnable`] systems.
#[macro_export]
macro_rules ! serial {
    ($($system:expr $(,)?)*) => {
        $crate::ImmutableSchedule::serial()
        $(
            .add_system($system)
        )*;
    };
}

/// Construct an [`ImmutableSchedule<Parallel>`] from a set of [`ParallelRunnable`] systems.
#[macro_export]
macro_rules ! parallel {
    ($($system:expr $(,)?)*) => {
        $crate::ImmutableSchedule::serial()
        $(
            .add_system($system)
        )*;
    };
}
