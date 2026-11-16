use bevy_ecs::prelude::*;
use engine_core::{Background, FrameBudget, FrameTime, JobKind, Jobs, RecordKey, Records, Taken};
use engine_heightfield::TileKey;
use engine_heightfield::tile::TileJob;

use crate::TileWater;
use crate::jobs::drawn::{DrawnWater, FloodHead, landed};
use crate::state::{HeldWater, WaterState, Work};

pub(crate) type DrawnWork = Background<Option<(TileWater, DrawnWater)>>;

pub(crate) fn start_drawing(
    jobs: &mut Jobs<'_, '_>,
    kind: &'static JobKind,
    input: Vec<u8>,
    priority: u32,
) -> DrawnWork {
    let record = RecordKey::of(kind, &input);
    jobs.start(kind, input, priority, move |bytes| {
        landed(&bytes, record).map_err(|error| kind.undecoded(error))
    })
}

pub(crate) fn draw_input(held: &HeldWater, record: &[u8]) -> Vec<u8> {
    let mut input = bytemuck::cast_slice::<f32, u8>(&held.face.heights).to_vec();
    input.extend_from_slice(record);
    input
}

pub(crate) fn flood_input(
    held: &HeldWater,
    tile: TileJob,
    (flow, running): ([u8; 32], &[u8]),
    regions: Vec<[f32; 4]>,
) -> Vec<u8> {
    let water = held.water.as_deref();
    let head = FloodHead {
        tile,
        flow,
        running: running.len() as u64,
        regions,
        spacing: held.face.spacing,
        wet: water.is_some(),
        held: water.map_or([0; 4], TileWater::holding),
        solids: water.map_or(&[][..], TileWater::solids).into(),
    };
    let mut input = engine_codec::encode(&head);
    input.extend_from_slice(running);
    input.extend_from_slice(bytemuck::cast_slice(&held.face.heights));
    if let Some(water) = water {
        input.extend_from_slice(bytemuck::cast_slice(water.surfaces()));
        input.extend_from_slice(bytemuck::cast_slice(water.info()));
    }
    input
}

pub(crate) fn draw(
    records: Res<Records>,
    (time, budget): (Res<FrameTime>, Res<FrameBudget>),
    mut state: ResMut<WaterState>,
) {
    let state = &mut *state;
    let frame = time.frame;
    let mut landed_any = false;
    for terrain in &mut state.terrains {
        let drawing: Vec<TileKey> = terrain
            .held
            .iter()
            .filter(|(_, held)| matches!(held.work, Some(Work::Draw(_) | Work::Redraw(..))))
            .map(|(&key, _)| key)
            .collect();
        for key in drawing {
            if landed_any && budget.expired() {
                return;
            }
            let Some(held) = terrain.held.get_mut(&key) else {
                continue;
            };
            let (work, flood) = match held.work.take() {
                Some(Work::Draw(work)) => (work, None),
                Some(Work::Redraw(work, to)) => (work, Some(to)),
                other @ (Some(Work::Base(_)) | None) => {
                    held.work = other;
                    continue;
                }
            };
            let made = match work.take(&*records) {
                Taken::Waiting(work) => {
                    held.work = Some(match flood {
                        None => Work::Draw(work),
                        Some(to) => Work::Redraw(work, to),
                    });
                    continue;
                }
                Taken::Ready(made) => Ok(made),
                Taken::Failed(error) => Err(error),
            };
            landed_any = true;
            match flood {
                None => match made {
                    Ok(made) => {
                        held.keep_made();
                        held.hold(made);
                        held.watered = None;
                    }
                    Err(error) => {
                        state.failures.record(frame, error.to_string());
                        held.made = Some(Err(error));
                    }
                },
                Some(to) => {
                    held.watered = to;
                    match made {
                        Ok(Some(made)) => {
                            state.rewatered_tiles += 1;
                            held.hold(Some(made));
                        }
                        Ok(None) => {}
                        Err(error) => state.failures.record(frame, error.to_string()),
                    }
                }
            }
            terrain.touch(key);
        }
    }
}
