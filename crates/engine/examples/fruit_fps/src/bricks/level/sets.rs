use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::*;
use engine::{Error, ErrorCode};
use engine_core::collections::{Cursor, Since};
use engine_core::{Asset, Assets, MeshFile};
use engine_visible_sets::{Cell, VisibleSets};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub(super) fn parsed(bytes: Arc<[u8]>, uri: &str) -> Result<Arc<VisibleSets>, Error> {
    VisibleSets::read(&bytes)
        .map(Arc::new)
        .map_err(|problem| Error::new(ErrorCode::InvalidRecord, uri).got(problem))
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LevelSets {
    pub file: Asset<Arc<VisibleSets>>,
}

#[derive(Resource, Clone, Debug, Default)]
pub(super) struct SetIndex {
    file: Option<Arc<VisibleSets>>,
    index: BTreeMap<Asset<MeshFile>, u32>,
    pub unknown: u32,
}

pub(super) fn adopt(
    found: Query<&LevelSets>,
    touched: Query<(), Changed<LevelSets>>,
    assets: Res<Assets>,
    mut seen: Local<Cursor>,
    mut sets: ResMut<SetIndex>,
) {
    let removed = found.is_empty() && sets.file.is_some();
    if matches!(assets.changed(&mut seen), Since::Same) && touched.is_empty() && !removed {
        return;
    }
    let file = found
        .iter()
        .find_map(|sets| assets.get(sets.file).ok())
        .cloned();
    let mut next = SetIndex {
        file: file.clone(),
        ..SetIndex::default()
    };
    if let Some(file) = file {
        for (at, id) in file.ids().iter().enumerate() {
            match assets.named::<MeshFile>(id) {
                Some(mesh) => {
                    next.index.insert(mesh, at as u32);
                }
                None => next.unknown += 1,
            }
        }
    }
    *sets = next;
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SetView<'a> {
    pub at: Cell<'a>,
    index: &'a BTreeMap<Asset<MeshFile>, u32>,
}

impl SetView<'_> {
    pub fn hides(&self, mesh: Option<Asset<MeshFile>>) -> Option<bool> {
        let at = *self.index.get(&mesh?)?;
        Some(!self.at.kept(at))
    }
}

impl SetIndex {
    pub fn loaded(&self) -> bool {
        self.file.is_some()
    }

    pub fn at(&self, eye: Vec3) -> Option<SetView<'_>> {
        Some(SetView {
            at: self.file.as_deref()?.at(eye.to_array())?,
            index: &self.index,
        })
    }
}
