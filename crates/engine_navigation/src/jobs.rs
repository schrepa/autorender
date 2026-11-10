use engine_codec::{self as codec, record};
use engine_core::{Error, ErrorCode, Job, JobCx, JobKind};
use engine_tiles::Pyramid;

use crate::{Gait, SOURCE, TilePortals, WalkGround};

pub(crate) const PORTALS_NAME: &str = "navigation::portals";

pub static PORTALS: JobKind = JobKind::new(PORTALS_NAME, &SOURCE, start_portals);

#[derive(Clone, Debug, PartialEq)]
pub struct PortalsJob {
    pub ground: WalkGround,
    pub pyramid: Pyramid,
    pub gait: Gait,
}

record!(PortalsJob {
    ground,
    pyramid,
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

fn start_portals(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let job: PortalsJob = codec::decode(input)?;
    job.gait.validate(&job.pyramid)?;
    let key = job.ground.key();
    if !job.pyramid.contains(key) || u32::from(key.level) != job.gait.level {
        return Err(Error::new(ErrorCode::OutOfRange, "$.ground.key").got([
            u16::from(key.level),
            key.x,
            key.z,
        ]));
    }
    if !job.ground.fits() {
        return Err(codec::invalid("$.ground"));
    }
    Ok(Box::new(Portals(Some(Box::new(job)))))
}

struct Portals(Option<Box<PortalsJob>>);

impl Job for Portals {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let Some(job) = self.0.take() else {
            return Err(Error::new(ErrorCode::InvalidValue, "$.step"));
        };
        Ok(Some(codec::encode(&TilePortals::new(
            &job.ground,
            &job.pyramid,
            &job.gait,
        ))))
    }
}
