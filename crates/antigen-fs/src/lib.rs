use antigen_core::{Construct, LazyComponent, ReadWriteLock, RwLock, Usage};
use legion::Entity;
use std::{fs::File, path::PathBuf};

pub enum FileBytes {}
pub enum FileString {}

pub type PathComponent = RwLock<PathBuf>;
pub type FileComponent = RwLock<LazyComponent<File>>;
pub type FileBytesComponent = Usage<FileBytes, RwLock<LazyComponent<Vec<u8>>>>;
pub type FileStringComponent = Usage<FileString, RwLock<LazyComponent<String>>>;

pub fn assemble_file<U: Send + Sync + 'static>(
    cmd: &mut legion::systems::CommandBuffer,
    entity: Entity,
    path: PathBuf,
) {
    cmd.add_component(entity, Usage::<U, PathComponent>::construct(path));
    cmd.add_component(
        entity,
        Usage::<U, FileComponent>::construct(LazyComponent::Pending),
    );
}

pub fn assemble_file_bytes<U: Send + Sync + 'static>(
    cmd: &mut legion::systems::CommandBuffer,
    entity: Entity,
    path: PathBuf,
) {
    cmd.add_component(
        entity,
        Usage::<U, FileBytesComponent>::construct(LazyComponent::Pending),
    );
    assemble_file::<U>(cmd, entity, path)
}

pub fn assemble_file_string<U: Send + Sync + 'static>(
    cmd: &mut legion::systems::CommandBuffer,
    entity: Entity,
    path: PathBuf,
) {
    cmd.add_component(
        entity,
        Usage::<U, FileStringComponent>::construct(LazyComponent::Pending),
    );
    assemble_file::<U>(cmd, entity, path)
}

#[legion::system(par_for_each)]
pub fn load_files<U: Send + Sync + 'static>(
    path: &Usage<U, PathComponent>,
    file: &Usage<U, FileComponent>,
) {
    let f = if let LazyComponent::Pending = &*file.read() {
        std::fs::File::open(&*path.read()).unwrap()
    } else {
        return;
    };

    file.write().set_ready(f);
}

#[legion::system(par_for_each)]
pub fn read_file_bytes<U: Send + Sync + 'static>(
    file: &Usage<U, FileComponent>,
    bytes: &Usage<U, FileBytesComponent>,
) {
    let mut drop = false;
    if let LazyComponent::Ready(f) = &mut *file.write() {
        let mut buf = Vec::<u8>::default();
        std::io::Read::read(f, &mut buf).unwrap();
        bytes.write().set_ready(buf);
        drop = true;
    }

    if drop {
        file.write().set_dropped();
    }
}

#[legion::system(par_for_each)]
pub fn read_file_string<U: Send + Sync + 'static>(
    file: &Usage<U, FileComponent>,
    string: &Usage<U, FileStringComponent>,
) {
    let mut drop = false;
    if let LazyComponent::Ready(f) = &mut *file.write() {
        let mut buf = String::default();
        std::io::Read::read_to_string(f, &mut buf).unwrap();
        string.write().set_ready(buf);
        drop = true;
    }

    if drop {
        file.write().set_dropped();
    }
}
