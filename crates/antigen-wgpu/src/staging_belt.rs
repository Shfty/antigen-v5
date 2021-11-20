use antigen_core::{
    ChangedFlag, GetIndirect, IndirectComponent, LazyComponent, ReadWriteLock, RwLock,
    RwLockReadGuard, RwLockWriteGuard, Usage,
};
use legion::{IntoQuery, World};
use wgpu::{
    util::StagingBelt, Buffer, BufferAddress, BufferSize, CommandEncoder, CommandEncoderDescriptor,
    Device,
};

use std::{
    collections::BTreeMap,
    future::Future,
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{BufferComponent, CommandBuffersComponent, ToBytes};

// Staging belt
static STAGING_BELT_ID_HEAD: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StagingBeltId(usize);

pub struct StagingBeltManager(BTreeMap<StagingBeltId, StagingBelt>);

impl StagingBeltManager {
    pub fn new() -> Self {
        StagingBeltManager(Default::default())
    }

    pub fn create_staging_belt(&mut self, chunk_size: BufferAddress) -> StagingBeltId {
        let staging_belt = StagingBelt::new(chunk_size);
        let id = STAGING_BELT_ID_HEAD.fetch_add(1, Ordering::Relaxed);
        let id = StagingBeltId(id);
        self.0.insert(id, staging_belt);
        id
    }

    pub fn write_buffer(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        offset: BufferAddress,
        size: BufferSize,
        belt_id: &StagingBeltId,
        data: &[u8],
    ) {
        self.0
            .get_mut(belt_id)
            .unwrap()
            .write_buffer(encoder, target, offset, size, device)
            .copy_from_slice(data);
    }

    pub fn finish(&mut self, belt_id: &StagingBeltId) {
        self.0.get_mut(belt_id).unwrap().finish()
    }

    pub fn recall(&mut self, belt_id: &StagingBeltId) -> impl Future + Send {
        self.0.get_mut(belt_id).unwrap().recall()
    }
}

// Staging belt handle
pub enum StagingBeltTag {}
pub type StagingBeltComponent = Usage<StagingBeltTag, RwLock<StagingBeltId>>;

// Staging belt buffer write operation
pub struct StagingBeltWriteComponent<T> {
    offset: RwLock<BufferAddress>,
    size: RwLock<BufferSize>,
    _phantom: PhantomData<T>,
}

impl<T> ReadWriteLock<BufferAddress> for StagingBeltWriteComponent<T> {
    fn read(&self) -> RwLockReadGuard<BufferAddress> {
        self.offset.read()
    }

    fn write(&self) -> RwLockWriteGuard<BufferAddress> {
        self.offset.write()
    }
}

impl<T> ReadWriteLock<BufferSize> for StagingBeltWriteComponent<T> {
    fn read(&self) -> RwLockReadGuard<BufferSize> {
        self.size.read()
    }

    fn write(&self) -> RwLockWriteGuard<BufferSize> {
        self.size.write()
    }
}

impl<T> StagingBeltWriteComponent<T> {
    pub fn new(offset: BufferAddress, size: BufferSize) -> Self {
        StagingBeltWriteComponent {
            offset: RwLock::new(offset),
            size: RwLock::new(size),
            _phantom: Default::default(),
        }
    }
}

// Write data to buffer via staging belt
pub fn staging_belt_write_thread_local<
    T: Send + Sync + 'static,
    L: ReadWriteLock<V> + Send + Sync + 'static,
    V: ToBytes,
>(
    world: &World,
    staging_belt_manager: &mut StagingBeltManager,
) {
    let device = if let Some(device) = <&Device>::query().iter(world).next() {
        device
    } else {
        return;
    };

    <(
        &Usage<T, StagingBeltWriteComponent<L>>,
        &L,
        &ChangedFlag<L>,
        &IndirectComponent<Usage<T, StagingBeltComponent>>,
        &IndirectComponent<Usage<T, BufferComponent>>,
        &IndirectComponent<Usage<T, CommandBuffersComponent>>,
    )>::query()
    .for_each(
        world,
        |(staging_belt_write, value, dirty_flag, staging_belt, buffer, command_buffers)| {
            let staging_belt = world.get_indirect(staging_belt).unwrap();
            let buffer = world.get_indirect(buffer).unwrap();
            let command_buffers = world.get_indirect(command_buffers).unwrap();

            if dirty_flag.get() {
                let staging_belt = staging_belt.read();

                let buffer = buffer.read();
                let buffer = if let LazyComponent::Ready(buffer) = &*buffer {
                    buffer
                } else {
                    return;
                };

                let offset = *ReadWriteLock::<BufferAddress>::read(staging_belt_write);
                let size = *ReadWriteLock::<BufferSize>::read(staging_belt_write);

                let value = value.read();
                let bytes = value.to_bytes();

                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor { label: None });

                staging_belt_manager.write_buffer(
                    device,
                    &mut encoder,
                    buffer,
                    offset,
                    size,
                    &*staging_belt,
                    bytes,
                );

                command_buffers.write().push(encoder.finish());

                dirty_flag.set(false);
            }
        },
    );
}

pub fn staging_belt_finish_thread_local<T: Send + Sync + 'static>(
    world: &World,
    staging_belt_manager: &mut StagingBeltManager,
) {
    <&Usage<T, StagingBeltComponent>>::query().for_each(world, |staging_belt| {
        staging_belt_manager.finish(&*staging_belt.read());
    });
}

pub fn staging_belt_recall_thread_local<T: Send + Sync + 'static>(
    world: &World,
    staging_belt_manager: &mut StagingBeltManager,
) {
    <&Usage<T, StagingBeltComponent>>::query().for_each(world, |staging_belt| {
        // Ignore resulting future - this assumes the wgpu device is being polled in wait mode
        let _ = staging_belt_manager.recall(&*staging_belt.read());
    });
}
