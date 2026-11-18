use glam::Vec3;

use super::tactics::{self, Hold, SiteTactics};
use super::{Duty, Line, LineKind, Order, Plan, Role, Slot, Squad, Step, Utility, spaced};
use crate::bots::aim::{clamp, flat, flat_distance, max, min};
use crate::bots::debug::Why;
use crate::bots::sense::{Block, DeviceSense, GamePhase};

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

    fn routed(
        &mut self,
        plan: &Plan<'_, '_>,
        member: usize,
        site: Option<usize>,
        hold: Hold,
    ) -> Hold {
        if plan.sense.game.phase == GamePhase::Planted {
            return hold;
        }
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
        if tactics::safe_path(
            plan.world,
            safe,
            from,
            hold.at,
            allowed,
            step,
            &mut self.buffer,
        ) {
            return hold;
        }
        let gate = site.and_then(|s| plan.site(s).back.map(|back| plan.site(s).ways[back].inside));
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
            let pressure = self.pressure(site, numbers);
            let heat = contact.max(pressure);
            let called = contact > 0 || pressure >= numbers.play.rotate_seen;
            if called && hottest.is_none_or(|(_, c)| heat > c) {
                hottest = Some((site, heat));
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
            alive
                .iter()
                .filter(|&&m| self.slots[m].group == site)
                .count() as u32
        });
        let heavy = self
            .rotated
            .is_some_and(|site| self.contact(site, numbers) > held + numbers.play.rotate_seen);
        let mid = plan.tactics.mid_hold.at;
        let mid_seen = self.contact_near(mid, numbers.play.site_near_m);
        let fallback = self
            .rotated
            .unwrap_or_else(|| self.fallback_site(plan, mid));
        for &m in &alive {
            let slot = self.slots[m];
            if slot.duty == Duty::Mid && mid_seen && self.rotated.is_none() {
                self.rotate_to(plan, m, fallback, rotating, false);
                rotating += 1;
                continue;
            }
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
                    self.orders[m].wait_contact = slot.duty == Duty::Anchor
                        && slot.rank == 1
                        && site.is_some_and(|s| self.contact(s, numbers) == 0);
                }
            }
        }
    }

    fn fallback_site(&self, plan: &Plan<'_, '_>, from: Vec3) -> usize {
        let numbers = plan.numbers;
        (0..plan.tactics.sites.len())
            .min_by(|&a, &b| {
                self.pressure(b, numbers)
                    .cmp(&self.pressure(a, numbers))
                    .then(
                        flat_distance(from, plan.site(a).plant)
                            .total_cmp(&flat_distance(from, plan.site(b).plant)),
                    )
                    .then(a.cmp(&b))
            })
            .unwrap_or(0)
    }

    fn rotate_to(
        &mut self,
        plan: &Plan<'_, '_>,
        member: usize,
        site: usize,
        k: usize,
        heavy: bool,
    ) {
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
        let critical =
            fuse < defuse + travel + play.defuse_margin_seconds || plan.sense.enemies_alive == 0;
        let short = fuse < defuse + travel + play.defuse_margin_seconds + play.clear_seconds;
        let down = plan.sense.enemies_alive >= alive.len() as u32 + play.retake_down;
        let saving = down || (alive.len() == 1 && short);
        if saving && alive.iter().any(|&m| self.saves(plan, m)) {
            for &m in &alive {
                if self.saves(plan, m) {
                    self.save_order(plan, m, plan.tactics.defend_home);
                } else {
                    self.lone_retake(plan, m, at, critical || short);
                }
            }
            self.set_step(Step::Save, Why::Saving, plan.tick);
            return;
        }
        if alive.len() == 1 {
            self.lone_retake(plan, nearest, at, critical || short);
            return;
        }
        if self.go_tick == u64::MAX {
            self.retake_gather(plan, &alive, at, fuse, critical);
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
        for &way in st.retake_order.iter().take(2) {
            let length = tactics::path_length(
                plan.world,
                from,
                st.ways[way].retake_stage,
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
        let near = plan.site(self.site.unwrap_or(0)).ways[way].retake_stage;
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

    fn stage_cost(&mut self, plan: &Plan<'_, '_>, from: Vec3, way: usize) -> (bool, f32) {
        let site = self.site.unwrap_or(0);
        let stage = plan.site(site).ways[way].retake_stage;
        let length = tactics::path_length(plan.world, from, stage, &mut self.buffer);
        let zone = plan.places().sites.get(site).map(|s| s.zone);
        let crosses = tactics::crosses(
            plan.world,
            from,
            &self.buffer,
            zone,
            plan.numbers.play.gate_step_m,
        );
        (crosses, length)
    }

    fn retake_entries(&mut self, plan: &Plan<'_, '_>, alive: &[usize], device: Vec3) {
        let members = plan.sense.members;
        let near = plan.numbers.play.site_near_m;
        let away: Vec<usize> = alive
            .iter()
            .copied()
            .filter(|&m| flat_distance(members[m].position, device) > near)
            .collect();
        let open: Vec<usize> = away
            .iter()
            .copied()
            .filter(|&m| self.entry_way[m].is_none())
            .collect();
        if open.is_empty() {
            return;
        }
        let ways: Vec<usize> = plan
            .site(self.site.unwrap_or(0))
            .retake_order
            .iter()
            .copied()
            .take(2)
            .collect();
        if ways.len() < 2 {
            for m in open {
                self.entry_for(plan, m);
            }
            return;
        }
        if away.len() < 3 {
            let mut best: Option<(bool, f32, usize)> = None;
            for &way in &ways {
                let mut crossing = false;
                let mut longest = 0.0f32;
                for &m in &away {
                    let (crosses, length) = self.stage_cost(plan, members[m].position, way);
                    crossing |= crosses;
                    longest = max(longest, length);
                }
                let better = best.is_none_or(|(c, l, _)| (crossing, longest) < (c, l));
                if better {
                    best = Some((crossing, longest, way));
                }
            }
            if let Some((_, _, way)) = best {
                for &m in &away {
                    self.entry_way[m] = Some(way);
                }
            }
            return;
        }
        let mut keys: Vec<(i32, f32, usize)> = Vec::with_capacity(open.len());
        for &m in &open {
            let from = members[m].position;
            let (cross_a, a) = self.stage_cost(plan, from, ways[0]);
            let (cross_b, b) = self.stage_cost(plan, from, ways[1]);
            keys.push((i32::from(cross_a) - i32::from(cross_b), a - b, m));
        }
        keys.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)).then(x.2.cmp(&y.2)));
        let first = keys.len() - keys.len() / 2;
        for (k, (_, _, m)) in keys.into_iter().enumerate() {
            self.entry_way[m] = Some(if k < first { ways[0] } else { ways[1] });
        }
    }

    fn retake_gather(
        &mut self,
        plan: &Plan<'_, '_>,
        alive: &[usize],
        device: Vec3,
        fuse: f32,
        critical: bool,
    ) {
        let site = self.site.unwrap_or(0);
        if plan.site(site).retake_order.is_empty() {
            return;
        }
        self.retake_entries(plan, alive, device);
        let numbers = plan.numbers;
        let play = &numbers.play;
        let spacing = play.spacing_m;
        let speed = numbers.player.run_speed_mps;
        let members = plan.sense.members;
        let eye = plan.eye();
        let ways = plan.site(site).ways.len();
        let mut ranks = vec![0usize; ways];
        let mut into: Vec<Option<f32>> = vec![None; ways];
        let mut arrived = 0usize;
        let mut slowest = 0.0f32;
        for &m in alive {
            if flat_distance(members[m].position, device) <= play.site_near_m {
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
            let Some(index) = self.entry_way[m] else {
                continue;
            };
            let rank = ranks[index];
            ranks[index] += 1;
            let way = &plan.site(site).ways[index];
            let stage = spaced(way.retake_stage, way.mid, rank, spacing);
            let at = plan
                .world
                .cover_from(way.inside + eye, stage, play.cover_search_m)
                .unwrap_or(stage);
            let face = way.hold.face;
            if into[index].is_none() {
                let from = way.retake_stage;
                into[index] =
                    Some(tactics::path_length(plan.world, from, device, &mut self.buffer) / speed);
            }
            self.simple(
                plan,
                m,
                Duty::Retaker,
                Step::Retake,
                Hold { at, face },
                false,
            );
            self.orders[m].wait_contact = true;
            let from = members[m].position;
            if flat_distance(from, at) <= spacing * 2.0 {
                arrived += 1;
            } else {
                let left = tactics::path_length(plan.world, from, at, &mut self.buffer) / speed;
                slowest = max(slowest, left);
            }
        }
        let entry = into.iter().flatten().fold(0.0f32, |a, &b| max(a, b));
        let spare = fuse
            - (numbers.round.defuse_seconds
                + play.defuse_margin_seconds
                + play.clear_seconds
                + play.utility_lead_seconds
                + entry);
        let group = (play.retake_group as usize).min(alive.len());
        let all = arrived >= alive.len();
        let late = slowest >= spare && arrived >= group;
        if all || late || spare <= 0.0 || critical {
            let rushed = critical || spare <= 0.0;
            self.start_retake(plan, alive, device, rushed);
            let why = if all { Why::Arrived } else { Why::TimeShort };
            self.set_step(Step::Retake, why, plan.tick);
        }
    }

    fn start_retake(&mut self, plan: &Plan<'_, '_>, alive: &[usize], device: Vec3, rushed: bool) {
        let numbers = plan.numbers;
        let site = self.site.unwrap_or(0);
        let speed = numbers.player.run_speed_mps;
        let lead = if rushed {
            0
        } else {
            numbers.ticks(numbers.play.utility_lead_seconds)
        };
        let ways = plan.site(site).ways.len();
        let mut used: Vec<usize> = alive.iter().filter_map(|&m| self.entry_way[m]).collect();
        used.sort_unstable();
        used.dedup();
        let mut entries: Vec<(usize, f32)> = Vec::with_capacity(used.len());
        for &w in &used {
            let way = &plan.site(site).ways[w];
            let length =
                tactics::path_length(plan.world, way.retake_stage, way.inside, &mut self.buffer);
            entries.push((w, length / speed));
        }
        let longest = entries.iter().fold(0.0f32, |a, e| max(a, e.1));
        self.go_tick = plan.tick;
        self.way_go = vec![plan.tick + lead; ways];
        for (w, time) in entries {
            self.way_go[w] = plan.tick + lead + numbers.ticks(longest - time);
        }
        self.retake_arrive = plan.tick + lead + numbers.ticks(longest);
        self.retake_tools = self.retake_tools_for(plan, alive, device);
    }

    fn far_angle(&self, plan: &Plan<'_, '_>, device: Vec3, from: Vec3) -> Vec3 {
        let play = &plan.numbers.play;
        let away = flat(device - from).normalize_or_zero();
        let point = device + away * play.retake_smoke_m;
        let places = plan.places();
        let zone = places
            .sites
            .get(self.site.unwrap_or(0))
            .and_then(|site| places.zones.get(site.zone as usize));
        let Some(zone) = zone else {
            return point;
        };
        let inset = play.spacing_m;
        let fit = |value: f32, a: f32, b: f32| {
            let low = min(a, b) + inset;
            let high = max(a, b) - inset;
            if low > high {
                (a + b) * 0.5
            } else {
                clamp(value, low, high)
            }
        };
        Vec3::new(
            fit(point.x, zone.min.x, zone.max.x),
            point.y,
            fit(point.z, zone.min.z, zone.max.z),
        )
    }

    fn retake_tools_for(
        &self,
        plan: &Plan<'_, '_>,
        alive: &[usize],
        device: Vec3,
    ) -> Vec<Option<Utility>> {
        let site = self.site.unwrap_or(0);
        let st = plan.site(site);
        let members = plan.sense.members;
        let mut tools: Vec<Option<Utility>> = vec![None; members.len()];
        let mut counts = vec![0usize; st.ways.len()];
        for &m in alive {
            if let Some(w) = self.entry_way[m] {
                counts[w] += 1;
            }
        }
        let used: Vec<usize> = (0..st.ways.len()).filter(|&w| counts[w] > 0).collect();
        let main = used
            .iter()
            .copied()
            .max_by(|&a, &b| counts[a].cmp(&counts[b]).then(b.cmp(&a)));
        let now = plan.tick;
        let line = |kind: LineKind, choke: Option<usize>| Line {
            kind,
            site: Some(site),
            choke,
        };
        let mut needs: Vec<(Block, Vec3, Option<usize>, Line, u64)> = Vec::new();
        for &w in &used {
            let go = self.way_go.get(w).copied().unwrap_or(now);
            let choke = Some(st.ways[w].choke);
            needs.push((
                Block::Flash,
                st.ways[w].flash_at,
                Some(w),
                line(LineKind::RetakeFlash, choke),
                go,
            ));
        }
        if let Some(main) = main {
            let far = self.far_angle(plan, device, st.ways[main].inside);
            needs.push((
                Block::Smoke,
                far,
                None,
                line(LineKind::RetakeSmoke, None),
                now,
            ));
        }
        for &w in &used {
            let choke = Some(st.ways[w].choke);
            needs.push((
                Block::DamageZone,
                st.ways[w].post.at,
                Some(w),
                line(LineKind::RetakeZone, choke),
                now,
            ));
        }
        needs.push((
            Block::Reveal,
            device,
            None,
            line(LineKind::RevealSite, None),
            now,
        ));
        for (block, target, way, line, after_tick) in needs {
            let pick = alive.iter().copied().find(|&m| {
                tools[m].is_none()
                    && members[m].bot
                    && members[m].has(block)
                    && way.is_none_or(|w| self.entry_way[m] == Some(w))
            });
            if let Some(m) = pick {
                tools[m] = Some(Utility {
                    block,
                    target,
                    after_tick,
                    line,
                });
            }
        }
        tools
    }

    fn retake_push(&mut self, plan: &Plan<'_, '_>, alive: &[usize], at: Vec3, critical: bool) {
        let site = self.site.unwrap_or(0);
        if plan.site(site).ways.is_empty() {
            return;
        }
        let numbers = plan.numbers;
        let clear_ticks = numbers.ticks(numbers.play.clear_seconds);
        let quiet = self
            .last_spotted(site)
            .is_none_or(|t| plan.tick.saturating_sub(t) > clear_ticks);
        let clear = quiet && plan.tick >= self.retake_arrive.saturating_add(clear_ticks);
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
        let spacing = numbers.play.spacing_m;
        let defusing = self.defuser.is_some();
        let mut ranks = vec![0usize; plan.site(site).ways.len()];
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
            let entry = self.entry_way[m].unwrap_or(st.retake_order[k % st.retake_order.len()]);
            let rank = ranks[entry];
            ranks[entry] += 1;
            let way = &st.ways[entry];
            let goal = match rank {
                0 => way.hold.at,
                1 => way.cross.at,
                _ => spaced(way.post.at, way.post.face, rank - 2, spacing),
            };
            let watch = st.ways[(entry + 1 + rank) % st.ways.len()].post.at + eye;
            let go = self.way_go.get(entry).copied().unwrap_or(self.go_tick);
            let mut order = Order {
                duty: if defusing { Duty::Cover } else { Duty::Retaker },
                step: if defusing { Step::Defuse } else { Step::Retake },
                goal,
                zone: None,
                face: if defusing { watch } else { at + eye },
                go_tick: if defusing { 0 } else { go },
                careful: false,
                utility: if defusing {
                    None
                } else {
                    self.retake_tools.get(m).copied().flatten()
                },
                hand_over: false,
                wait_contact: !defusing && plan.tick < go,
            };
            if flat_distance(members[m].position, at) <= numbers.play.site_near_m && !defusing {
                order.goal = self.orders[m].goal;
                order.go_tick = 0;
                order.wait_contact = false;
            }
            self.put(plan, m, order);
        }
    }
}
