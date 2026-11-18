use glam::Vec3;

use super::aim::{look_angles, radians, wrap};
use super::numbers::PlayNumbers;
use super::sense::World;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    pub yaw: f32,
    pub pitch: f32,
    pub landing: Vec3,
    pub miss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Progress {
    Working,
    Solved(Shot),
    Refused,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Search {
    slot: usize,
    target: Vec3,
    origin: Vec3,
    bearing: f32,
    tried: u32,
    best: Option<Shot>,
}

const COLUMNS: [f32; 3] = [0.0, 1.0, -1.0];
const REFINE: [f32; 4] = [0.5, -0.5, 0.25, -0.25];

fn pitch_step(play: &PlayNumbers) -> f32 {
    let span = play.throw_pitch_deg[1] - play.throw_pitch_deg[0];
    radians(span) / play.throw_steps.saturating_sub(1).max(1) as f32
}

impl Search {
    pub fn new(slot: usize, target: Vec3, origin: Vec3) -> Search {
        Search {
            slot,
            target,
            origin,
            bearing: look_angles(origin, target).0,
            tried: 0,
            best: None,
        }
    }

    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    fn candidate(&self, play: &PlayNumbers) -> Option<(f32, f32)> {
        let steps = play.throw_steps.max(1);
        let coarse = steps * COLUMNS.len() as u32;
        if self.tried < coarse {
            let column = COLUMNS[(self.tried / steps) as usize];
            let row = self.tried % steps;
            let yaw = wrap(self.bearing + column * radians(play.throw_yaw_deg));
            let pitch = radians(play.throw_pitch_deg[0]) + pitch_step(play) * row as f32;
            return Some((yaw, pitch));
        }
        let refine = REFINE.get((self.tried - coarse) as usize)?;
        let best = self.best?;
        Some((best.yaw, best.pitch + refine * pitch_step(play)))
    }

    pub fn step(&mut self, world: &dyn World, play: &PlayNumbers) -> Progress {
        for _ in 0..play.throw_work_per_tick.max(1) {
            let Some((yaw, pitch)) = self.candidate(play) else {
                return match self.best {
                    Some(best) if best.miss <= play.line_up_m => Progress::Solved(best),
                    Some(_) | None => Progress::Refused,
                };
            };
            self.tried += 1;
            let Some(landing) = world.throw(self.slot, self.origin, yaw, pitch) else {
                continue;
            };
            let miss = (landing - self.target).length();
            if self.best.is_none_or(|best| miss < best.miss) {
                self.best = Some(Shot {
                    yaw,
                    pitch,
                    landing,
                    miss,
                });
            }
        }
        Progress::Working
    }
}
