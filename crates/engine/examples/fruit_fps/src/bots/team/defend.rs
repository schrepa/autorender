use glam::Vec3;

use super::tactics::{self, Hold};
use super::{Duty, Plan, Role, Slot, Squad, Step, spaced};
use crate::bots::aim::flat_distance;
use crate::bots::debug::Why;
use crate::bots::sense::DeviceSense;

const MID: usize = usize::MAX;

impl Squad {
    pub(super) fn start_defense(&mut self, plan: &Plan<'_, '_>) {
        let mut alive = plan.alive();
        let members = plan.sense.members;
        let sites = plan.tactics.sites.len();
        let mut wanted: Vec<(usize, usize)> = Vec::with_capacity(alive.len());
        let mut depth = 0usize;
        while wanted.len() < alive.len() && sites > 0 {
            for site in 0..sites {
                if wanted.len() < alive.len() && depth < plan.site(site).ways.len() {
                    wanted.push((site, depth));
                }
            }
            if depth == 0 && wanted.len() < alive.len() {
                wanted.push((MID, 0));
            }
            depth += 1;
            if depth > alive.len() {
                break;
            }
        }
        if let (Some(mid), Some(controller)) = (
            wanted.iter().position(|w| w.0 == MID),
            alive
                .iter()
                .position(|&m| members[m].role == Role::Controller),
        ) && mid < alive.len()
        {
            alive.swap(mid, controller);
        }
        for (k, &m) in alive.iter().enumerate() {
            let (site, rank) = wanted.get(k).copied().unwrap_or((MID, k));
            let (duty, hold) = if site == MID {
                (Duty::Mid, plan.tactics.mid_hold)
            } else {
                (Duty::Anchor, plan.site(site).ways[rank].hold)
            };
            self.slots[m] = Slot {
                duty,
                group: site,
                rank,
                hold,
            };
        }
        self.approach = super::Approach::Hold;
        self.set_step(Step::Setup, Why::RoundStart, plan.tick);
    }

    pub(super) fn defend(&mut self, plan: &Plan<'_, '_>) {
        let alive = plan.alive();
        let numbers = plan.numbers;
        let mut hottest: Option<(usize, u32)> = None;
        for site in 0..plan.tactics.sites.len() {
            let pressure = self.pressure(site, numbers);
            if pressure >= numbers.play.rotate_seen && hottest.is_none_or(|(_, p)| pressure > p) {
                hottest = Some((site, pressure));
            }
        }
        if let Some((site, _)) = hottest
            && self.rotated != Some(site)
        {
            self.rotated = Some(site);
            self.site = Some(site);
            self.set_step(Step::Rotate, Why::Called, plan.tick);
        }
        for (k, &m) in alive.iter().enumerate() {
            let slot = self.slots[m];
            match self.rotated {
                Some(site)
                    if slot.group != site && !(slot.duty == Duty::Anchor && slot.rank == 0) =>
                {
                    let st = plan.site(site);
                    if st.ways.is_empty() {
                        continue;
                    }
                    let way = st.defend_order[k % st.defend_order.len()];
                    let hold = st.ways[way].hold;
                    let at = spaced(
                        hold.at,
                        hold.face,
                        k / st.defend_order.len(),
                        plan.numbers.play.spacing_m,
                    );
                    self.simple(
                        plan,
                        m,
                        Duty::Retaker,
                        Step::Rotate,
                        Hold {
                            at,
                            face: hold.face,
                        },
                        true,
                    );
                }
                _ => self.simple(plan, m, slot.duty, Step::Setup, slot.hold, true),
            }
        }
    }

    pub(super) fn retake(&mut self, plan: &Plan<'_, '_>) {
        let DeviceSense::Planted {
            site,
            at,
            fuse_ticks_left,
            ..
        } = plan.sense.game.device
        else {
            return;
        };
        self.site = Some(site);
        let alive = plan.alive();
        let members = plan.sense.members;
        let numbers = plan.numbers;
        let Some(&nearest) = alive.iter().min_by(|&&a, &&b| {
            let da = flat_distance(members[a].position, at);
            let db = flat_distance(members[b].position, at);
            members[b]
                .bot
                .cmp(&members[a].bot)
                .then(da.total_cmp(&db))
                .then(a.cmp(&b))
        }) else {
            return;
        };
        let travel =
            tactics::path_length(plan.world, members[nearest].position, at, &mut self.buffer)
                / numbers.player.run_speed_mps;
        let fuse = numbers.seconds(u64::from(fuse_ticks_left));
        let defuse = numbers.round.defuse_seconds;
        let play = &numbers.play;
        if fuse < defuse + travel {
            for &m in &alive {
                if self.saves(plan, m) {
                    self.save_order(plan, m, plan.tactics.defend_home);
                }
            }
            self.set_step(Step::Save, Why::Saving, plan.tick);
            return;
        }
        let critical = fuse < defuse + travel + play.defuse_margin_seconds;
        if alive.len() == 1 {
            self.lone_retake(
                plan,
                nearest,
                at,
                critical
                    || fuse < defuse + travel + play.defuse_margin_seconds + play.clear_seconds,
            );
            return;
        }
        if self.go_tick == u64::MAX {
            self.retake_gather(plan, &alive, at, critical);
        }
        if self.go_tick != u64::MAX {
            self.retake_push(plan, &alive, at, critical);
        }
    }

    fn entry_for(&mut self, plan: &Plan<'_, '_>, member: usize) -> Option<usize> {
        if let Some(way) = self.entry_way[member] {
            return Some(way);
        }
        let st = plan.site(self.site.unwrap_or(0));
        let from = plan.sense.members[member].position;
        let mut best: Option<(usize, f32)> = None;
        for &way in st.defend_order.iter().take(2) {
            let length = tactics::path_length(
                plan.world,
                from,
                st.ways[way].defend_stage,
                &mut self.buffer,
            );
            if best.is_none_or(|(_, l)| length < l) {
                best = Some((way, length));
            }
        }
        self.entry_way[member] = best.map(|(way, _)| way);
        self.entry_way[member]
    }

    fn lone_retake(&mut self, plan: &Plan<'_, '_>, member: usize, at: Vec3, now: bool) {
        let eye = plan.eye();
        if now {
            self.simple(
                plan,
                member,
                Duty::Defuser,
                Step::Defuse,
                Hold { at, face: at + eye },
                false,
            );
            self.set_step(Step::Defuse, Why::TimeShort, plan.tick);
            return;
        }
        let Some(way) = self.entry_for(plan, member) else {
            return;
        };
        let near = plan.site(self.site.unwrap_or(0)).ways[way].defend_stage;
        let spot = plan
            .world
            .cover_from(at + eye, near, plan.numbers.play.cover_search_m)
            .unwrap_or(near);
        self.simple(
            plan,
            member,
            Duty::Retaker,
            Step::Retake,
            Hold {
                at: spot,
                face: at + eye,
            },
            true,
        );
    }

    fn retake_gather(
        &mut self,
        plan: &Plan<'_, '_>,
        alive: &[usize],
        device: Vec3,
        critical: bool,
    ) {
        let st = plan.site(self.site.unwrap_or(0));
        if st.defend_order.is_empty() {
            return;
        }
        let spacing = plan.numbers.play.spacing_m;
        let members = plan.sense.members;
        let eye = plan.eye();
        let mut arrived = 0usize;
        for (k, &m) in alive.iter().enumerate() {
            if flat_distance(members[m].position, device) <= plan.numbers.play.site_near_m {
                let keep = self.orders[m].goal;
                self.simple(
                    plan,
                    m,
                    Duty::Retaker,
                    Step::Retake,
                    Hold {
                        at: keep,
                        face: device + eye,
                    },
                    true,
                );
                arrived += 1;
                continue;
            }
            let Some(index) = self.entry_for(plan, m) else {
                continue;
            };
            let st = plan.site(self.site.unwrap_or(0));
            let way = &st.ways[index];
            let at = spaced(way.defend_stage, way.mid, k, spacing);
            self.simple(
                plan,
                m,
                Duty::Retaker,
                Step::Retake,
                Hold {
                    at,
                    face: way.hold.face,
                },
                true,
            );
            if flat_distance(members[m].position, at) <= spacing * 2.0 {
                arrived += 1;
            }
        }
        if arrived + 1 >= alive.len() || plan.tick >= self.deadline || critical {
            self.go_tick = plan.tick;
            self.set_step(
                Step::Retake,
                if critical {
                    Why::TimeShort
                } else {
                    Why::Arrived
                },
                plan.tick,
            );
        }
    }

    fn retake_push(&mut self, plan: &Plan<'_, '_>, alive: &[usize], at: Vec3, critical: bool) {
        let site = self.site.unwrap_or(0);
        let st = plan.site(site);
        if st.ways.is_empty() {
            return;
        }
        let numbers = plan.numbers;
        let clear_ticks = numbers.ticks(numbers.play.clear_seconds);
        let quiet = self
            .last_spotted(site)
            .is_none_or(|t| plan.tick.saturating_sub(t) > clear_ticks);
        let clear = quiet && plan.tick >= self.go_tick + clear_ticks;
        let members = plan.sense.members;
        if (clear || critical) && self.defuser.is_none_or(|d| !members[d].alive) {
            self.defuser = alive.iter().copied().min_by(|&a, &b| {
                let da = flat_distance(members[a].position, at);
                let db = flat_distance(members[b].position, at);
                members[b]
                    .bot
                    .cmp(&members[a].bot)
                    .then(da.total_cmp(&db))
                    .then(a.cmp(&b))
            });
            self.set_step(
                Step::Defuse,
                if critical {
                    Why::TimeShort
                } else {
                    Why::Defusing
                },
                plan.tick,
            );
        }
        let eye = plan.eye();
        for (k, &m) in alive.iter().enumerate() {
            if Some(m) == self.defuser {
                self.simple(
                    plan,
                    m,
                    Duty::Defuser,
                    Step::Defuse,
                    Hold { at, face: at + eye },
                    false,
                );
                continue;
            }
            let way =
                &st.ways[self.entry_way[m].unwrap_or(st.defend_order[k % st.defend_order.len()])];
            let (duty, step) = if self.defuser.is_some() {
                (Duty::Cover, Step::Defuse)
            } else {
                (Duty::Retaker, Step::Retake)
            };
            let face = if self.defuser.is_some() {
                way.post.face
            } else {
                at + eye
            };
            let goal = spaced(
                way.post.at,
                way.post.face,
                k / st.defend_order.len(),
                numbers.play.spacing_m,
            );
            let mut order = super::Order {
                duty,
                step,
                goal,
                zone: None,
                face,
                go_tick: if self.defuser.is_some() {
                    0
                } else {
                    self.go_tick
                },
                careful: false,
                utility: None,
            };
            if flat_distance(members[m].position, at) <= numbers.play.site_near_m
                && self.defuser.is_none()
            {
                order.goal = self.orders[m].goal;
            }
            self.put(plan, m, order);
        }
    }
}
