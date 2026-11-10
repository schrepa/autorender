use bevy_ecs::prelude::*;
use engine_core::{Background, FrameBudget, FrameTime, JobKind, Jobs, RecordKey, Records, Taken};
use engine_heightfield::TileKey;
use engine_heightfield::tile::TileJob;

use crate::TileWater;
use crate::jobs::drawn::{DrawHead, DrawnWater, FloodHead, landed};
use crate::state::{Drawing, HeldWater, WaterState, Work};

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
    let beds = held.face.river_beds.as_deref().unwrap_or(&[]);
    let head = DrawHead {
        spacing: held.face.spacing,
        beds: beds.len() as u64,
    };
    let mut input = engine_codec::encode(&head);
    input.extend_from_slice(bytemuck::cast_slice(&held.face.heights));
    input.extend_from_slice(bytemuck::cast_slice(beds));
    input.extend_from_slice(record);
    input
}

pub(crate) fn flood_input(
    held: &HeldWater,
    tile: TileJob,
    (flow, running): ([u8; 32], &[u8]),
    regions: Vec<[f32; 4]>,
) -> Vec<u8> {
    let beds = held.face.river_beds.as_deref().unwrap_or(&[]);
    let water = held.water.as_deref();
    let head = FloodHead {
        tile,
        flow,
        running: running.len() as u64,
        regions,
        spacing: held.face.spacing,
        beds: beds.len() as u64,
        wet: water.is_some(),
        held: water.map_or([0; 4], TileWater::holding),
        solids: water.map_or(&[][..], TileWater::solids).into(),
    };
    let mut input = engine_codec::encode(&head);
    input.extend_from_slice(running);
    input.extend_from_slice(bytemuck::cast_slice(&held.face.heights));
    input.extend_from_slice(bytemuck::cast_slice(beds));
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
            let (work, drawing) = match held.work.take() {
                Some(Work::Draw(work)) => (work, None),
                Some(Work::Redraw(work, drawing)) => (work, Some(drawing)),
                other @ (Some(Work::Base(_)) | None) => {
                    held.work = other;
                    continue;
                }
            };
            let made = match work.take(&*records) {
                Taken::Waiting(work) => {
                    held.work = Some(match drawing {
                        None => Work::Draw(work),
                        Some(drawing) => Work::Redraw(work, drawing),
                    });
                    continue;
                }
                Taken::Ready(made) => Ok(made),
                Taken::Failed(error) => Err(error),
            };
            landed_any = true;
            match drawing {
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
                Some(drawing) => match (made.map_err(|error| error.to_string()), drawing) {
                    (Ok(made), Drawing::Given(record)) => {
                        held.hold(made);
                        held.watered = None;
                        held.given = Some(record);
                    }
                    (Ok(made), Drawing::Flood(to)) => {
                        if made.is_some() {
                            state.rewatered_tiles += 1;
                            held.hold(made);
                            held.given = None;
                        }
                        held.watered = to;
                    }
                    (Err(error), Drawing::Given(_)) => {
                        terrain.settled.water_failed(key, &error);
                        state.failures.record(frame, error);
                    }
                    (Err(error), Drawing::Flood(to)) => {
                        held.watered = to;
                        state.failures.record(frame, error);
                    }
                },
            }
            terrain.touch(key);
        }
    }
}
