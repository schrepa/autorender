use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::{Res, ResMut};
use engine::prelude::Resource;
use engine_core::collections::HashMap;
use engine_core::{Background, Files, Landed};

pub(crate) mod sets;

pub(crate) struct SourceFile {
    pub(crate) path: &'static str,
    pub(crate) digest: u128,
    pub(crate) size: [u32; 2],
}

include!(concat!(env!("OUT_DIR"), "/source_files.rs"));

pub(crate) fn file(path: &str) -> Option<&'static SourceFile> {
    SOURCE_FILES.iter().find(|file| file.path == path)
}

pub(crate) fn set_digest(paths: &[String]) -> Option<u64> {
    let joined = paths.join("|");
    SET_DIGESTS
        .iter()
        .find(|(set, _)| *set == joined)
        .map(|&(_, digest)| digest)
}

enum Held {
    Wanted,
    Reading(Background<Arc<[u8]>, String>),
    Read(Arc<[u8]>),
    Failed(String),
}

#[derive(Default)]
struct Shelf {
    root: String,
    held: HashMap<String, Held>,
}

#[derive(Resource, Clone, Default)]
pub(crate) struct Sources(Arc<Mutex<Shelf>>);

impl Sources {
    pub(crate) fn new(root: &str) -> Self {
        Self(Arc::new(Mutex::new(Shelf {
            root: root.to_owned(),
            held: HashMap::default(),
        })))
    }

    pub(crate) fn bytes(&self, path: &str) -> Result<Option<Arc<[u8]>>, String> {
        let mut shelf = lock(&self.0);
        match shelf.held.get(path) {
            Some(Held::Read(bytes)) => Ok(Some(Arc::clone(bytes))),
            Some(Held::Failed(error)) => Err(error.clone()),
            Some(Held::Wanted | Held::Reading(_)) => Ok(None),
            None => {
                shelf.held.insert(path.to_owned(), Held::Wanted);
                Ok(None)
            }
        }
    }
}

fn lock(held: &Mutex<Shelf>) -> MutexGuard<'_, Shelf> {
    held.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn read(sources: Option<Res<Sources>>, mut files: ResMut<Files>) {
    let Some(sources) = sources else {
        return;
    };
    let mut shelf = lock(&sources.0);
    let Shelf { root, held } = &mut *shelf;
    for (path, state) in held.iter_mut() {
        if matches!(state, Held::Wanted) {
            *state = Held::Reading(files.read_from(root, path));
        }
        let Held::Reading(reading) = state else {
            continue;
        };
        let landed = match reading.poll(&*files) {
            Landed::Waiting => None,
            Landed::Ready(bytes) => Some(Held::Read(Arc::clone(bytes))),
            Landed::Failed(error) => Some(Held::Failed(error.to_string())),
        };
        if let Some(landed) = landed {
            *state = landed;
        }
    }
}
