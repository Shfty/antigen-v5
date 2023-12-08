use std::path::PathBuf;

use antigen_core::{Construct, LazyComponent, ReadWriteLock, RwLock, Usage};
use antigen_fs::{assemble_file_string, FileStringComponent};
use legion::{systems::CommandBuffer, Entity};
use shambler::GeoMap;

pub type MapFileComponent = RwLock<LazyComponent<GeoMap>>;

pub fn assemble_map_file<U: Send + Sync + 'static>(
    cmd: &mut CommandBuffer,
    entity: Entity,
    path: PathBuf,
) {
    cmd.add_component(
        entity,
        Usage::<U, MapFileComponent>::construct(LazyComponent::Pending),
    );
    assemble_file_string::<U>(cmd, entity, path);
}

#[legion::system(par_for_each)]
pub fn parse_map_file<U: Send + Sync + 'static>(
    file_string: &Usage<U, FileStringComponent>,
    map_file: &Usage<U, MapFileComponent>,
) {
    let file_string = file_string.read();
    if map_file.read().is_pending() {
        if let LazyComponent::Ready(string) = &*file_string {
            let map = string.parse::<shambler::shalrath::repr::Map>().unwrap();
            let map = GeoMap::from(map);
            map_file.write().set_ready(map)
        }
    }
}
