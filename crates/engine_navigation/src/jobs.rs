use engine_codec::{self as codec, record};
use engine_core::{Error, ErrorCode, Job, JobCx, JobKind, JobRef};
use engine_places::ground::{GroundGrid, Pyramid, TileKey};

use crate::{Gait, SOURCE, TilePortals, WalkGround};

pub(crate) const WALK_GROUND_NAME: &str = "navigation::walk_ground";
pub(crate) const PORTALS_NAME: &str = "navigation::portals";

pub static WALK_GROUND: JobKind = JobKind::new(WALK_GROUND_NAME, &SOURCE, start_ground);
pub static PORTALS: JobKind = JobKind::new(PORTALS_NAME, &SOURCE, start_portals);

#[derive(Clone, Debug, PartialEq)]
pub struct GroundJob {
    pub grid: JobRef,
    pub source: Vec<u8>,
    pub key: TileKey,
}

record!(GroundJob { grid, source, key });

impl GroundJob {
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PortalsJob {
    pub grid: JobRef,
    pub source: Vec<u8>,
    pub pyramid: Pyramid,
    pub key: TileKey,
    pub gait: Gait,
}

record!(PortalsJob {
    grid,
    source,
    pyramid,
    key,
    gait,
});

impl PortalsJob {
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn portals_output(bytes: &[u8]) -> Result<TilePortals, Error> {
    codec::decode(bytes)
}

fn grid_input(source: &[u8], key: TileKey) -> Vec<u8> {
    let mut input = source.to_vec();
    input.extend_from_slice(&codec::encode(&key));
    input
}

fn start_ground(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let GroundJob { grid, source, key } = codec::decode(input)?;
    cx.start(&grid, &grid_input(&source, key))
}

fn start_portals(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let PortalsJob {
        grid,
        source,
        pyramid,
        key,
        gait,
    } = codec::decode(input)?;
    gait.validate(&pyramid)?;
    if !pyramid.contains(key) || u32::from(key.level) != gait.level {
        return Err(Error::new(ErrorCode::OutOfRange, "$.key").got([
            u16::from(key.level),
            key.x,
            key.z,
        ]));
    }
    let grid = cx.start(&grid, &grid_input(&source, key))?;
    Ok(Box::new(Portals {
        grid,
        pyramid,
        gait,
    }))
}

struct Portals {
    grid: Box<dyn Job>,
    pyramid: Pyramid,
    gait: Gait,
}

impl Job for Portals {
    fn step(&mut self, budget: u64, cx: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let Some(bytes) = self.grid.step(budget, cx)? else {
            return Ok(None);
        };
        let mut grid: GroundGrid = codec::decode(&bytes)?;
        let water = grid.water.take();
        let ground = WalkGround::new(&grid, water, &self.pyramid);
        Ok(Some(codec::encode(&TilePortals::new(
            &ground,
            &self.pyramid,
            &self.gait,
        ))))
    }
}
