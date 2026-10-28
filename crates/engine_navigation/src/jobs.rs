use std::sync::Arc;

use engine_math::collections::HashMap;
use engine_terrain::codec::{self, CodecError, Reader};
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

pub fn portals_output(bytes: &[u8]) -> Result<TilePortals, CodecError> {
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
        if !Self::runs(kind) {
            return Err(format!("unknown job kind {kind}"));
        }
        let invalid = |error: CodecError| error.to_string();
        let mut reader = Reader::new(input);
        let PortalsJob {
            stack,
            pyramid,
            key,
            stamps,
            gait,
        } = reader.get().map_err(invalid)?;
        reader.finish().map_err(invalid)?;
        gait.validate(&pyramid)?;
        if !pyramid.contains(key) || u32::from(key.level) != gait.level {
            return Err("portals of a tile outside its gait's level".into());
        }
        let stack = terrain
            .stack(&stack)
            .ok_or_else(|| format!("unknown stack {stack}"))?;
        let layers = stamped(stack, &stamps)?;
        self.running.insert(
            id,
            Portals {
                layers,
                pyramid,
                key,
                gait,
            },
        );
        Ok(())
    }

    pub fn step(&mut self, id: u64, _budget: u64) -> Result<Option<Vec<u8>>, String> {
        let Some(job) = self.running.remove(&id) else {
            return Err("no such job".into());
        };
        tile_portals(&job.layers, &job.pyramid, job.key, &job.gait)
            .map(|portals| Some(codec::encode(&portals)))
    }

    pub fn cancel(&mut self, id: u64) {
        self.running.remove(&id);
    }
}
