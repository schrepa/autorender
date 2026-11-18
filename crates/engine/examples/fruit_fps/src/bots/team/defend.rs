use glam::Vec3;

use super::tactics::{self, Hold, SiteTactics};
use super::{Duty, Line, LineKind, Order, Plan, Role, Slot, Squad, Step, Utility, spaced};
use crate::bots::aim::flat_distance;
use crate::bots::debug::Why;
use crate::bots::sense::{Block, DeviceSense};

const MID: usize = usize::MAX;

fn post_of(st: &SiteTactics, rank: usize) -> Hold {
    match rank {
        0 => st.ways[0].hold,
        1 => st.ways[0].cross,
        _ => st.ways[(rank - 1).min(st.ways.len() - 1)].hold,
    }
}

impl Squad {
    pub(super) fn start_defense(&mut self, plan: &Plan<'_, '_>) {
        let mut alive = plan.alive();
        let members = plan.sense.members;
        let sites = plan.tactics.sites.len();
        let mut wanted: Vec<(usize, usize)> = Vec::with_capacity(alive.len());
        let mut depth = 0usize;
        while wanted.len() < alive.len() && sites > 0 {
            for site in 0..sites {
                let ways = plan.site(site).ways.len();
                if wanted.len() < alive.len() && ways > 0 && depth <= ways {
                    wanted.push((site, depth));
                }
            }
            if depth == 1 && wanted.len() < alive.len() {
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
                (Duty::Anchor, post_of(plan.site(site), rank))
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

    fn routed(&mut self, plan: &Plan<'_, '_>, member: usize, site: Option<usize>, hold: Hold) -> Hold {
        let places = plan.places();
        let play = &plan.numbers.play;
        let from = plan.sense.members[member].position;
        let site_zone = site.and_then(|s| places.sites.get(s)).map(|s| s.zone);
        let here = places.zone_at(from);
        if (site_zone.is_some() && here == site_zone)
            || flat_distance(from, hold.at) <= play.arrive_m * 2.0
        {
            return hold;
        }
        let safe = &plan.tactics.safe;
        let allowed = [site_zone, places.zone_at(hold.at)];
        let step = play.gate_step_m;
        if tactics::safe_path(plan.world, safe, from, hold.at, allowed, step, &mut self.buffer) {
            return hold;
        }
        let gate = site
            .and_then(|s| plan.site(s).back.map(|back| plan.site(s).ways[back].inside));
        let home = plan.tactics.defend_home;
        let at_home = flat_distance(from, home) <= play.site_near_m;
        match gate {
            Some(gate)
                if at_home
                    || tactics::safe_path(
                        plan.world,
                        safe,
                        from,
                        gate,
                        [site_zone, None],
                        step,
                        &mut self.buffer,
                    ) =>
            {
                Hold {
                    at: gate,
                    face: hold.face,
                }
            }
            Some(_) | None if at_home => hold,
            Some(_) | None => Hold {
                at: home,
                face: hold.face,
            },
        }
    }

    pub(super) fn defend(&mut self, plan: &Plan<'_, '_>) {
        let alive = plan.alive();
        let numbers = plan.numbers;
        let mut hottest: Option<(usize, u32)> = None;
        for site in 0..plan.tactics.sites.len() {
            let contact = self.contact(site, numbers);
            if contact > 0 && hottest.is_none_or(|(_, c)| contact > c) {
                hottest = Some((site, contact));
            }
        }
        if let Some((site, _)) = hottest
            && self.rotated != Some(site)
        {
            self.rotated = Some(site);
            self.site = Some(site);
            self.set_step(Step::Rotate, Why::Called, plan.tick);
        }
        if let &[last] = alive.as_slice()
            && plan.sense.enemies_alive >= numbers.play.retake_group
        {
            self.play_time(plan, last);
            return;
        }
        let mut rotating = 0usize;
        let held = self.rotated.map_or(0, |site| {
            alive.iter().filter(|&&m| self.slots[m].group == site).count() as u32
        });
        let heavy = self
            .rotated
            .is_some_and(|site| self.contact(site, numbers) > held + numbers.play.rotate_seen);
        for &m in &alive {
            let slot = self.slots[m];
            match self.rotated {
                Some(site)
                    if heavy
                        || (slot.group != site
                            && !(slot.duty == Duty::Anchor && slot.rank == 0)) =>
                {
                    self.rotate_to(plan, m, site, rotating, heavy);
                    rotating += 1;
                }
                _ => {
                    let site = (slot.group != MID).then_some(slot.group);
                    let hold = self.routed(plan, m, site, slot.hold);
                    self.simple(plan, m, slot.duty, Step::Setup, hold, true);
                }
            }
        }
    }

    fn rotate_to(&mut self, plan: &Plan<'_, '_>, member: usize, site: usize, k: usize, heavy: bool) {
        let st = plan.site(site);
        if st.ways.is_empty() {
            return;
        }
        let taken = heavy || self.contact(site, plan.numbers) >= plan.numbers.play.rotate_seen;
        let spacing = plan.numbers.play.spacing_m;
        let hold = match st.back {
            Some(back) if taken => {
                let way = &st.ways[back];
                Hold {
                    at: spaced(way.defend_stage, way.mid, k, spacing),
                    face: way.inside + plan.eye(),
                }
            }
            Some(_) | None => post_of(st, 1 + k % (st.ways.len() + 1)),
        };
        let hold = self.routed(plan, member, Some(site), hold);
        self.simple(plan, member, Duty::Retaker, Step::Rotate, hold, true);
    }

    fn play_time(&mut self, plan: &Plan<'_, '_>, member: usize) {
        let from = plan.sense.members[member].position;
        let site = self.rotated.unwrap_or_else(|| {
            (0..plan.tactics.sites.len())
                .min_by(|&a, &b| {
                    flat_distance(from, plan.site(a).plant)
                        .total_cmp(&flat_distance(from, plan.site(b).plant))
                        .then(a.cmp(&b))
                })
                .unwrap_or(0)
        });
        let st = plan.site(site);
        if st.ways.is_empty() {
            return;
        }
        let hold = match st.back {
            Some(back) => Hold {
                at: st.ways[back].defend_stage,
                face: st.ways[back].inside + plan.eye(),
            },
            None => post_of(st, 0),
        };
        let hold = self.routed(plan, member, Some(site), hold);
        self.simple(plan, member, Duty::Anchor, Step::Setup, hold, true);
        self.set_step(Step::Setup, Why::Saving, plan.tick);
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
        if let Some(back) = st.back {
            self.entry_way[member] = Some(back);
            return Some(back);
        }
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
        let hold = Hold {
            at: spot,
            face: at + eye,
        };
        let hold = self.routed(plan, member, self.site, hold);
        self.simple(plan, member, Duty::Retaker, Step::Retake, hold, true);
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
            let hold = Hold {
                at,
                face: way.hold.face,
            };
            let hold = self.routed(plan, m, self.site, hold);
            self.simple(plan, m, Duty::Retaker, Step::Retake, hold, true);
            if flat_distance(members[m].position, at) <= spacing * 2.0 {
                arrived += 1;
            }
        }
        let group = (plan.numbers.play.retake_group as usize).min(alive.len());
        let ready = arrived >= group && (arrived + 1 >= alive.len() || plan.tick >= self.deadline);
        if ready || critical {
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

    fn retake_utility(&self, plan: &Plan<'_, '_>, entry: usize, rank: usize) -> Option<Utility> {
        let st = plan.site(self.site.unwrap_or(0));
        let way = &st.ways[entry];
        let after_tick = self.go_tick;
        let site = self.site;
        match rank {
            0 => Some(Utility {
                block: Block::Flash,
                target: way.flash_at,
                after_tick,
                line: Line {
                    kind: LineKind::RetakeFlash,
                    site,
                    choke: Some(way.choke),
                },
            }),
            1 => {
                let far = (0..st.ways.len()).find(|&w| w != entry)?;
                Some(Utility {
                    block: Block::Smoke,
                    target: st.ways[far].post.at,
                    after_tick,
                    line: Line {
                        kind: LineKind::RetakeSmoke,
                        site,
                        choke: Some(st.ways[far].choke),
                    },
                })
            }
            _ => Some(Utility {
                block: Block::Reveal,
                target: st.plant,
                after_tick,
                line: Line {
                    kind: LineKind::RevealSite,
                    site,
                    choke: None,
                },
            }),
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
            let st = plan.site(site);
            let entry = self.entry_way[m].unwrap_or(st.defend_order[k % st.defend_order.len()]);
            let way = &st.ways[entry];
            let defusing = self.defuser.is_some();
            let (duty, step) = if defusing {
                (Duty::Cover, Step::Defuse)
            } else {
                (Duty::Retaker, Step::Retake)
            };
            let face = if defusing { way.post.face } else { at + eye };
            let goal = spaced(
                way.post.at,
                way.post.face,
                k / st.defend_order.len(),
                numbers.play.spacing_m,
            );
            let mut order = Order {
                duty,
                step,
                goal,
                zone: None,
                face,
                go_tick: if defusing { 0 } else { self.go_tick },
                careful: false,
                utility: (!defusing)
                    .then(|| self.retake_utility(plan, entry, k))
                    .flatten(),
            };
            if flat_distance(members[m].position, at) <= numbers.play.site_near_m && !defusing {
                order.goal = self.orders[m].goal;
            }
            self.put(plan, m, order);
        }
    }
}
