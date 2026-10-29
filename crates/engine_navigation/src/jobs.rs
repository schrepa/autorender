use std::sync::Arc;

use engine_core::ErrorCode;
use engine_math::collections::HashMap;
use engine_terrain::codec::{self, Error, Reader};
use engine_terrain::jobs::{TerrainWorker, stamped};
use engine_terrain::{Layers, Pyramid, Stamp, TileKey, record};

use crate::{Gait, TilePortals, tile_portals};

#[derive(Clone, Debug, PartialEq)]
pub struct PortalsJob {
    pub stack: String,
    pub pyramid: Pyramid,
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
    pub gait: Gait,
}

record!(PortalsJob {
    stack,
    pyramid,
    key,
    stamps,
    gait,
});

impl PortalsJob {
    pub const KIND: &'static str = "terrain_portals";

    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn portals_output(bytes: &[u8]) -> Result<TilePortals, Error> {
    codec::decode(bytes)
}

struct Portals {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    key: TileKey,
    gait: Gait,
}

#[derive(Default)]
pub struct NavigationWorker {
    running: HashMap<u64, Portals>,
}

impl NavigationWorker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn runs(kind: &str) -> bool {
        kind == PortalsJob::KIND
    }

    pub fn holds(&self, id: u64) -> bool {
        self.running.contains_key(&id)
    }

    pub fn run(
        &mut self,
        kind: &str,
        input: &[u8],
        terrain: &TerrainWorker,
    ) -> Result<Vec<u8>, String> {
        const WHOLE: u64 = u64::MAX;
        self.start(WHOLE, kind, input, terrain)?;
        loop {
            if let Some(output) = self.step(WHOLE, u64::MAX)? {
                return Ok(output);
            }
        }
    }

    pub fn start(
        &mut self,
        id: u64,
        kind: &str,
        input: &[u8],
        terrain: &TerrainWorker,
    ) -> Result<(), String> {
        let portals = Self::begin(kind, input, terrain).map_err(|error| error.to_string())?;
        self.running.insert(id, portals);
        Ok(())
    }

    fn begin(kind: &str, input: &[u8], terrain: &TerrainWorker) -> Result<Portals, Error> {
        if !Self::runs(kind) {
            return Err(Error::new(ErrorCode::UnknownVariant, "$.kind")
                .got(kind)
                .variants([PortalsJob::KIND]));
        }
        let mut reader = Reader::new(input);
        let PortalsJob {
            stack,
            pyramid,
            key,
            stamps,
            gait,
        } = reader.get()?;
        reader.finish()?;
        gait.validate(&pyramid)?;
        if !pyramid.contains(key) || u32::from(key.level) != gait.level {
            return Err(Error::new(ErrorCode::OutOfRange, "$.key").got([
                u16::from(key.level),
                key.x,
                key.z,
            ]));
        }
        let layers = terrain
            .stack(&stack)
            .ok_or_else(|| Error::new(ErrorCode::UnknownReference, "$.stack").got(&stack))?;
        Ok(Portals {
            layers: stamped(layers, &stamps)?,
            pyramid,
            key,
            gait,
        })
    }

    pub fn step(&mut self, id: u64, _budget: u64) -> Result<Option<Vec<u8>>, String> {
        let Some(job) = self.running.remove(&id) else {
            return Err(Error::new(ErrorCode::UnknownReference, "$.id")
                .got(id)
                .to_string());
        };
        tile_portals(&job.layers, &job.pyramid, job.key, &job.gait)
            .map(|portals| Some(codec::encode(&portals)))
            .map_err(|error| error.to_string())
    }

    pub fn cancel(&mut self, id: u64) {
        self.running.remove(&id);
    }
}
