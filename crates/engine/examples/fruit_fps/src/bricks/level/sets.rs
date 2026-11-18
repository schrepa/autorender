use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::*;
use engine::{Error, ErrorCode};
use engine_core::{GeometryHandle, SceneContext, SceneCx};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"LVS1";

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SetsFile {
    origin: [f32; 2],
    cell_m: f32,
    cells: [u32; 2],
    top_m: f32,
    ids: Vec<String>,
    floors: Vec<f32>,
    stride: usize,
    kept: Vec<u8>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8], String> {
        let end = self
            .at
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len());
        let Some(end) = end else {
            return Err(format!(
                "ends at byte {} while {count} more were needed",
                self.bytes.len()
            ));
        };
        let taken = &self.bytes[self.at..end];
        self.at = end;
        Ok(taken)
    }

    fn word(&mut self) -> Result<[u8; 4], String> {
        let taken = self.take(4)?;
        Ok([taken[0], taken[1], taken[2], taken[3]])
    }

    fn count(&mut self) -> Result<u32, String> {
        self.word().map(u32::from_le_bytes)
    }

    fn real(&mut self) -> Result<f32, String> {
        self.word().map(f32::from_le_bytes)
    }
}

fn parsed(bytes: &[u8]) -> Result<SetsFile, String> {
    let mut read = Reader { bytes, at: 0 };
    if read.take(4)? != MAGIC {
        return Err("does not start with LVS1".into());
    }
    let cells = [read.count()?, read.count()?];
    let origin = [read.real()?, read.real()?];
    let cell_m = read.real()?;
    let top_m = read.real()?;
    if !(cell_m.is_finite() && cell_m > 0.0 && top_m.is_finite() && top_m > 0.0) {
        return Err(format!("cell_m {cell_m} and top_m {top_m} must be above 0"));
    }
    let chunks = read.count()? as usize;
    let mut ids = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        let length = read.take(2)?;
        let length = u16::from_le_bytes([length[0], length[1]]) as usize;
        let id = std::str::from_utf8(read.take(length)?)
            .map_err(|_| format!("chunk id {} is not UTF-8", ids.len()))?;
        ids.push(id.to_owned());
    }
    let padding = (4 - read.at % 4) % 4;
    read.take(padding)?;
    let count = cells[0] as usize * cells[1] as usize;
    let mut floors = Vec::with_capacity(count);
    for _ in 0..count {
        floors.push(read.real()?);
    }
    let stride = chunks.div_ceil(8);
    let kept = read.take(count * stride)?.to_vec();
    let left = bytes.len() - read.at;
    if left != 0 && left != count * stride {
        return Err(format!("{left} bytes left after the kept sets"));
    }
    Ok(SetsFile {
        origin,
        cell_m,
        cells,
        top_m,
        ids,
        floors,
        stride,
        kept,
    })
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LevelSets {
    pub file: String,
    #[serde(skip)]
    parsed: Option<Arc<SetsFile>>,
}

impl LevelSets {
    pub(super) fn files(&self) -> Vec<String> {
        vec![self.file.clone()]
    }

    pub(super) fn prepared(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let at = format!("{path}.file");
        let Some(bytes) = cx.buffer(&self.file, &at)? else {
            return Ok(self);
        };
        let file = parsed(&bytes).map_err(|problem| {
            Error::new(ErrorCode::InvalidRecord, at.clone())
                .got(format!("{}: {problem}", self.file))
        })?;
        self.parsed = Some(Arc::new(file));
        Ok(self)
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub(super) struct VisibleSets {
    file: Option<Arc<SetsFile>>,
    index: BTreeMap<GeometryHandle, u32>,
    pub unknown: u32,
}

impl PartialEq for VisibleSets {
    fn eq(&self, other: &Self) -> bool {
        let same = match (&self.file, &other.file) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            (Some(_), None) | (None, Some(_)) => false,
        };
        same && self.index == other.index
    }
}

pub(super) fn adopt(
    found: Query<&LevelSets>,
    scene: Option<Res<SceneContext>>,
    mut sets: ResMut<VisibleSets>,
) {
    let file = found.iter().find_map(|sets| sets.parsed.clone());
    let mut next = VisibleSets {
        file: file.clone(),
        ..VisibleSets::default()
    };
    if let (Some(file), Some(scene)) = (file, scene.as_deref()) {
        let named: BTreeMap<&str, GeometryHandle> = scene
            .assets
            .meshes
            .iter()
            .map(|(name, handle)| (name.as_str(), *handle))
            .collect();
        for (at, id) in file.ids.iter().enumerate() {
            match named.get(id.as_str()) {
                Some(handle) => {
                    next.index.insert(*handle, at as u32);
                }
                None => next.unknown += 1,
            }
        }
    }
    if *sets != next {
        *sets = next;
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SetView<'a> {
    pub cell: [u32; 2],
    kept: &'a [u8],
    index: &'a BTreeMap<GeometryHandle, u32>,
}

impl SetView<'_> {
    fn bit(row: &[u8], at: u32) -> bool {
        row.get(at as usize / 8)
            .is_some_and(|byte| byte & (1 << (at % 8)) != 0)
    }

    pub fn hides(&self, mesh: Option<GeometryHandle>) -> Option<bool> {
        let at = *self.index.get(&mesh?)?;
        Some(!Self::bit(self.kept, at))
    }
}

impl VisibleSets {
    pub fn loaded(&self) -> bool {
        self.file.is_some()
    }

    pub fn at(&self, eye: Vec3) -> Option<SetView<'_>> {
        let file = self.file.as_deref()?;
        let x = ((eye.x - file.origin[0]) / file.cell_m).floor();
        let z = ((eye.z - file.origin[1]) / file.cell_m).floor();
        if !(x >= 0.0 && z >= 0.0 && x < file.cells[0] as f32 && z < file.cells[1] as f32) {
            return None;
        }
        let cell = [x as u32, z as u32];
        let row = cell[1] as usize * file.cells[0] as usize + cell[0] as usize;
        let floor = *file.floors.get(row)?;
        let height = eye.y - floor;
        if !(floor.is_finite() && (0.0..=file.top_m).contains(&height)) {
            return None;
        }
        let span = row * file.stride..(row + 1) * file.stride;
        Some(SetView {
            cell,
            kept: file.kept.get(span)?,
            index: &self.index,
        })
    }
}
