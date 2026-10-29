use engine_core::{Error, ErrorCode, GroundGrid, GroundWorker, Pyramid, TileKey};
use engine_math::collections::HashMap;
use engine_terrain::codec::{self, Reader};
use engine_terrain::record;

use crate::{Gait, TilePortals, WalkGround};

#[derive(Clone, Debug, PartialEq)]
pub struct PortalsJob {
    pub source: Vec<u8>,
    pub pyramid: Pyramid,
    pub key: TileKey,
    pub gait: Gait,
}

record!(PortalsJob {
    source,
    pyramid,
    key,
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
    grid: GroundGrid,
    pyramid: Pyramid,
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
        ground: &dyn GroundWorker,
    ) -> Result<Vec<u8>, String> {
        const WHOLE: u64 = u64::MAX;
        self.start(WHOLE, kind, input, ground)?;
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
        ground: &dyn GroundWorker,
    ) -> Result<(), String> {
        let portals = Self::begin(kind, input, ground).map_err(|error| error.to_string())?;
        self.running.insert(id, portals);
        Ok(())
    }

    fn begin(kind: &str, input: &[u8], ground: &dyn GroundWorker) -> Result<Portals, Error> {
        if !Self::runs(kind) {
            return Err(Error::new(ErrorCode::UnknownVariant, "$.kind")
                .got(kind)
                .variants([PortalsJob::KIND]));
        }
        let mut reader = Reader::new(input);
        let PortalsJob {
            source,
            pyramid,
            key,
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
        Ok(Portals {
            grid: ground.grid(&source, key)?,
            pyramid,
            gait,
        })
    }

    pub fn step(&mut self, id: u64, _budget: u64) -> Result<Option<Vec<u8>>, String> {
        let Some(job) = self.running.remove(&id) else {
            return Err(Error::new(ErrorCode::UnknownReference, "$.id")
                .got(id)
                .to_string());
        };
        let ground = WalkGround::new(&job.grid, &job.pyramid);
        Ok(Some(codec::encode(&TilePortals::new(&ground, &job.pyramid, &job.gait))))
    }

    pub fn cancel(&mut self, id: u64) {
        self.running.remove(&id);
    }
}
