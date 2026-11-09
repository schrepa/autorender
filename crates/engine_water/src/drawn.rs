use bevy_ecs::prelude::*;
use engine_core::{FrameBudget, FrameTime, Records};
use engine_heightfield::TileKey;
use engine_heightfield::tile::TileJob;

use crate::TileWater;
use crate::jobs::DRAWN;
use crate::jobs::drawn::{DrawHead, FloodHead, landed};
use crate::state::{Base, Drawing, HeldWater, WaterState};

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
            .filter(|(_, held)| matches!(held.base, Base::Drawing(_) | Base::Redrawing(..)))
            .map(|(&key, _)| key)
            .collect();
        for key in drawing {
            if landed_any && budget.expired() {
                return;
            }
            let Some(held) = terrain.held.get_mut(&key) else {
                continue;
            };
            let asked = match &held.base {
                Base::Drawing(asked) | Base::Redrawing(asked, _) => asked,
                Base::Waiting | Base::Asking(_) | Base::Failed | Base::Made | Base::Left => {
                    continue;
                }
            };
            let Some(output) = records.get(asked) else {
                continue;
            };
            landed_any = true;
            let record = *asked.key();
            let made = output.map_err(|error| error.to_string()).and_then(|bytes| {
                landed(&bytes, record).map_err(|error| {
                    format!("{}: its output does not decode: {error}", DRAWN.name())
                })
            });
            let base = std::mem::replace(&mut held.base, Base::Made);
            if let Base::Drawing(_) = base {
                match made {
                    Ok(made) => {
                        held.hold(made);
                        held.watered = None;
                    }
                    Err(error) => {
                        held.base = Base::Failed;
                        state.failures.record(frame, error);
                    }
                }
            } else if let Base::Redrawing(_, drawing) = base {
                match (made, drawing) {
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
                }
            }
            terrain.touch(key);
        }
    }
}
