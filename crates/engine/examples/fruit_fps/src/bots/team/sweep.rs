use glam::Vec3;

use super::tactics::{self, Hold};
use super::{Plan, Squad, Step};
use crate::bots::aim::flat_distance;
use crate::bots::debug::Why;
use crate::bots::sense::{CallKind, GamePhase};
use crate::rules::Side;

impl Squad {
    pub(super) fn heard_from(&mut self, kind: CallKind, tick: u64) {
        let contact = matches!(
            kind,
            CallKind::Spotted | CallKind::Heard | CallKind::Device | CallKind::Down
        );
        if contact && tick > self.last_contact {
            self.last_contact = tick;
        }
    }

    pub(super) fn sweep_due(&self, plan: &Plan<'_, '_>) -> bool {
        let sweepers = plan.alive().len();
        let quiet = plan.numbers.ticks(plan.numbers.play.sweep_quiet_seconds);
        let enemies = plan.sense.enemies_alive as usize;
        let members = plan.sense.members;
        let bot_carries = members
            .iter()
            .any(|member| member.alive && member.bot && member.carrying_device);
        let hunting_side = plan.sense.game.side == Side::Defend || !bot_carries;
        hunting_side
            && plan.sense.game.phase == GamePhase::Live
            && enemies > 0
            && enemies < sweepers
            && plan.tick.saturating_sub(self.last_contact) >= quiet
    }

    fn sweep_places(&mut self, plan: &Plan<'_, '_>) {
        let from = match plan.sense.game.side {
            Side::Attack => plan.tactics.defend_home,
            Side::Defend => plan.tactics.attack_home,
        };
        let mut places: Vec<(f32, Vec3)> = plan
            .places()
            .zones
            .iter()
            .map(|zone| {
                let centre = (zone.min + zone.max) * 0.5;
                let at = Vec3::new(centre.x, zone.min.y, centre.z);
                let length = tactics::path_length(plan.world, from, at, &mut self.buffer);
                (length, at)
            })
            .collect();
        places.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.sweep = places.into_iter().map(|(_, at)| at).collect();
        self.sweep_at = (0..self.sweep_at.len()).collect();
    }

    pub(super) fn sweep_step(&mut self, plan: &Plan<'_, '_>) {
        let members = plan.sense.members;
        let sweepers: Vec<usize> = plan
            .alive()
            .into_iter()
            .filter(|&m| members[m].bot)
            .collect();
        if sweepers.is_empty() {
            return;
        }
        let starting = self.step != Step::Hunt;
        if self.sweep.is_empty() {
            self.sweep_places(plan);
        }
        let count = self.sweep.len();
        if count == 0 {
            return;
        }
        if self.sweep_at.len() < members.len() {
            self.sweep_at.resize(members.len(), 0);
        }
        let reach = plan.numbers.play.spacing_m * 2.0;
        let eye = plan.eye();
        for (k, &m) in sweepers.iter().enumerate() {
            let mut index = self.sweep_at[m] % count;
            if starting {
                index = k % count;
            } else if flat_distance(members[m].position, self.sweep[index]) <= reach {
                index = (index + sweepers.len()) % count;
            }
            self.sweep_at[m] = index;
            let at = self.sweep[index];
            let duty = self.slots[m].duty;
            self.simple(
                plan,
                m,
                duty,
                Step::Hunt,
                Hold { at, face: at + eye },
                false,
            );
        }
        self.set_step(Step::Hunt, Why::LostTarget, plan.tick);
    }
}
