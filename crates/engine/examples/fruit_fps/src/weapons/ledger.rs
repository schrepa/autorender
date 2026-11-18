use glam::Vec3;

use super::hits::HitJoints;
use super::numbers::StandJoints;
use super::recoil::sin_cos;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Harm {
    victim: u32,
    attacker: u32,
    tick: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DamageLedger {
    harms: Vec<Harm>,
}

impl DamageLedger {
    pub fn record(&mut self, victim: u32, attacker: u32, tick: u64) {
        self.harms.push(Harm {
            victim,
            attacker,
            tick,
        });
    }

    pub fn assistants(&self, victim: u32, killer: Option<u32>, now: u64, window: u64) -> Vec<u32> {
        let mut found: Vec<u32> = self
            .harms
            .iter()
            .filter(|harm| {
                harm.victim == victim
                    && harm.attacker != victim
                    && Some(harm.attacker) != killer
                    && now.saturating_sub(harm.tick) <= window
            })
            .map(|harm| harm.attacker)
            .collect();
        found.sort_unstable();
        found.dedup();
        found
    }

    pub fn forget_victim(&mut self, victim: u32) {
        self.harms.retain(|harm| harm.victim != victim);
    }

    pub fn prune(&mut self, now: u64, window: u64) {
        self.harms
            .retain(|harm| now.saturating_sub(harm.tick) <= window);
    }

    pub fn len(&self) -> usize {
        self.harms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.harms.is_empty()
    }
}

pub fn joints_from_body(
    stand: &StandJoints,
    feet: Vec3,
    yaw_radians: f32,
    height_share: f32,
) -> (HitJoints, Vec3) {
    let (yaw_sin, yaw_cos) = sin_cos(yaw_radians);
    let right = Vec3::new(yaw_cos, 0.0, -yaw_sin);
    let facing = Vec3::new(-yaw_sin, 0.0, -yaw_cos);
    let up = |height: f32| feet + Vec3::Y * (height * height_share);
    let half = right * (stand.feet_apart * 0.5);
    let joints = HitJoints {
        head: up(stand.head),
        spine: up(stand.spine),
        hip: up(stand.hip),
        feet: [feet - half, feet + half],
    };
    (joints, facing)
}
