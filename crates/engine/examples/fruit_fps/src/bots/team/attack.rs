use glam::Vec3;

use super::tactics::{self, Hold};
use super::{
    Approach, Duty, Line, LineKind, Order, Plan, Role, Slot, Squad, Step, Utility, spaced,
};
use crate::bots::aim::flat_distance;
use crate::bots::debug::Why;
use crate::bots::sense::{Block, DeviceSense};

fn pick(plan: &Plan<'_, '_>, alive: &[usize], taken: &[bool], role: Role) -> Option<usize> {
    alive
        .iter()
        .copied()
        .find(|&m| !taken[m] && plan.sense.members[m].role == role)
}

impl Squad {
    pub(super) fn start_attack(&mut self, plan: &Plan<'_, '_>) {
        let play = &plan.numbers.play;
        let alive = plan.alive();
        let members = plan.sense.members;
        let guns: u32 = alive.iter().map(|&m| members[m].gun_cost).sum();
        let mean_gun = guns.checked_div(alive.len() as u32).unwrap_or(0);
        self.approach = if mean_gun < play.eco_gun_cost {
            Approach::Eco
        } else {
            match self.rng.pick_weighted(&play.approach_weights) {
                0 => Approach::Fast,
                1 => Approach::Split,
                _ => Approach::Default,
            }
        };
        self.site = Some(self.rng.below(plan.tactics.sites.len()));
        self.assign_attack(plan, &alive);
        let quick = matches!(self.approach, Approach::Fast | Approach::Eco);
        let gather = if quick {
            play.gather_seconds[0]
        } else {
            play.gather_seconds[1]
        };
        self.deadline = plan.tick + plan.numbers.ticks(gather);
        let share = self.rng.range(play.default_pick_share);
        self.pick_tick = plan.tick + (plan.sense.game.ticks_left as f32 * share) as u64;
        self.go_tick = u64::MAX;
        let step = if self.approach == Approach::Default {
            Step::Wait
        } else {
            Step::Gather
        };
        self.set_step(step, Why::RoundStart, plan.tick);
    }

    fn assign_attack(&mut self, plan: &Plan<'_, '_>, alive: &[usize]) {
        let members = plan.sense.members;
        let mut taken = vec![false; members.len()];
        let mut duties = vec![Duty::Follow; members.len()];
        if let Some(carrier) = alive
            .iter()
            .copied()
            .find(|&m| members[m].carrying_device && members[m].bot)
        {
            let handoff = plan.numbers.play.handoff_m;
            let mate = (members[carrier].role == Role::Entry)
                .then(|| {
                    alive
                        .iter()
                        .copied()
                        .filter(|&m| {
                            m != carrier && members[m].bot && members[m].role != Role::Entry
                        })
                        .map(|m| {
                            (
                                m,
                                flat_distance(members[m].position, members[carrier].position),
                            )
                        })
                        .filter(|(_, d)| *d <= handoff)
                        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                        .map(|(m, _)| m)
                })
                .flatten();
            let holder = mate.unwrap_or(carrier);
            self.hand_over = mate.map(|mate| (carrier, mate));
            duties[holder] = Duty::Carrier;
            taken[holder] = true;
        }
        let entry = pick(plan, alive, &taken, Role::Entry)
            .or_else(|| alive.iter().copied().find(|&m| !taken[m]));
        if let Some(entry) = entry {
            duties[entry] = Duty::Entry;
            taken[entry] = true;
        }
        if let Some(smoke) = pick(plan, alive, &taken, Role::Controller) {
            duties[smoke] = Duty::Smoke;
            taken[smoke] = true;
        }
        let lurks =
            matches!(self.approach, Approach::Split | Approach::Default) && alive.len() >= 4;
        if lurks {
            let lurker = pick(plan, alive, &taken, Role::Denial)
                .or_else(|| pick(plan, alive, &taken, Role::Initiator));
            if let Some(lurker) = lurker {
                duties[lurker] = Duty::Lurk;
                taken[lurker] = true;
            }
        }
        let split = matches!(self.approach, Approach::Split | Approach::Default);
        let mut ranks = [0usize; 2];
        let mut first_follow = true;
        for &m in alive {
            let duty = duties[m];
            let group = match duty {
                Duty::Entry | Duty::Carrier => 0,
                Duty::Follow if first_follow => {
                    first_follow = false;
                    0
                }
                Duty::Follow
                | Duty::Smoke
                | Duty::Lurk
                | Duty::Anchor
                | Duty::Mid
                | Duty::Retaker
                | Duty::Defuser
                | Duty::Cover
                | Duty::Saver => usize::from(split),
            };
            let rank = ranks[group];
            ranks[group] += 1;
            self.slots[m] = Slot {
                duty,
                group,
                rank,
                hold: plan.tactics.mid_spot,
            };
        }
    }

    pub(super) fn attack(&mut self, plan: &Plan<'_, '_>) {
        let alive = plan.alive();
        if alive.is_empty() || self.site.is_none() {
            return;
        }
        self.follow_human(plan, &alive);
        self.check_time(plan, &alive);
        self.check_collapse(plan, &alive);
        match self.step {
            Step::Wait => self.attack_wait(plan, &alive),
            Step::Gather => self.attack_gather(plan, &alive),
            Step::Buy
            | Step::Setup
            | Step::Execute
            | Step::Plant
            | Step::PostPlant
            | Step::Rotate
            | Step::Retake
            | Step::Defuse
            | Step::Pickup
            | Step::Save
            | Step::Hunt => self.attack_execute(plan, &alive),
        }
        self.hand_over_orders(plan);
        if let DeviceSense::Dropped { at } = plan.sense.game.device {
            let members = plan.sense.members;
            let handed = self
                .hand_over
                .filter(|&(_, to)| members[to].alive)
                .map(|(_, to)| to);
            let dropper = self.hand_over.map(|(from, _)| from);
            let nearest = handed.or_else(|| {
                alive
                    .iter()
                    .copied()
                    .filter(|&m| Some(m) != dropper || alive.len() == 1)
                    .min_by(|&a, &b| {
                        let da = flat_distance(members[a].position, at);
                        let db = flat_distance(members[b].position, at);
                        members[b]
                            .bot
                            .cmp(&members[a].bot)
                            .then(da.total_cmp(&db))
                            .then(a.cmp(&b))
                    })
            });
            if let Some(m) = nearest {
                let face = at + plan.eye();
                self.simple(
                    plan,
                    m,
                    Duty::Carrier,
                    Step::Pickup,
                    Hold { at, face },
                    false,
                );
            }
        }
        if self.lost_round(plan) && !self.can_still_plant(plan, &alive) {
            for &m in &alive {
                if self.saves(plan, m) {
                    self.save_order(plan, m, plan.tactics.attack_home);
                }
            }
            self.why = Why::Saving;
        }
    }

    fn can_still_plant(&mut self, plan: &Plan<'_, '_>, alive: &[usize]) -> bool {
        let Some(site) = self.site else {
            return false;
        };
        let from = self.carrier_at(plan, alive);
        let numbers = plan.numbers;
        let length =
            tactics::path_length(plan.world, from, plan.site(site).plant, &mut self.buffer);
        let need = length / numbers.player.run_speed_mps
            + numbers.round.plant_seconds
            + numbers.play.late_round_seconds;
        plan.seconds_left() > need
    }

    fn stagger(&mut self, plan: &Plan<'_, '_>, member: usize, rank: usize) {
        let stagger = plan.numbers.ticks(plan.numbers.play.go_stagger_seconds);
        self.orders[member].go_tick = self.since + stagger * rank as u64;
    }

    fn carrier_at(&self, plan: &Plan<'_, '_>, alive: &[usize]) -> Vec3 {
        let members = plan.sense.members;
        if let Some(&m) = alive
            .iter()
            .find(|&&m| members[m].carrying_device && members[m].bot)
        {
            return members[m].position;
        }
        if let DeviceSense::Dropped { at } = plan.sense.game.device {
            return at;
        }
        let sum: Vec3 = alive.iter().map(|&m| members[m].position).sum();
        sum / alive.len().max(1) as f32
    }

    fn follow_human(&mut self, plan: &Plan<'_, '_>, alive: &[usize]) {
        if !matches!(self.step, Step::Wait | Step::Gather | Step::Execute) {
            return;
        }
        let members = plan.sense.members;
        let follow = plan.numbers.play.follow_human_m;
        let mut lead: Option<(usize, f32)> = None;
        for &m in alive {
            let member = &members[m];
            if member.bot {
                continue;
            }
            let Some(from) = self.human_from.get_mut(m) else {
                continue;
            };
            let from = *from.get_or_insert(member.position);
            let need = if member.carrying_device {
                follow * 0.5
            } else {
                follow
            };
            if let Some(site) = plan.places().site_at(member.position) {
                lead = Some((site, f32::INFINITY));
                break;
            }
            for site in 0..plan.tactics.sites.len() {
                let plant = plan.site(site).plant;
                let gain = flat_distance(from, plant) - flat_distance(member.position, plant);
                if gain >= need && lead.is_none_or(|(_, g)| gain > g) {
                    lead = Some((site, gain));
                }
            }
        }
        if let Some((site, _)) = lead
            && self.site != Some(site)
        {
            self.site = Some(site);
            self.why = Why::HumanLead;
            self.since = plan.tick;
        }
    }

    fn check_time(&mut self, plan: &Plan<'_, '_>, alive: &[usize]) {
        if !matches!(self.step, Step::Wait | Step::Gather) {
            return;
        }
        let from = self.carrier_at(plan, alive);
        let numbers = plan.numbers;
        let mut best: Option<(usize, f32)> = None;
        for (index, site) in plan.tactics.sites.iter().enumerate() {
            let length = tactics::path_length(plan.world, from, site.plant, &mut self.buffer);
            if best.is_none_or(|(_, l)| length < l) {
                best = Some((index, length));
            }
        }
        let Some((nearest, length)) = best else {
            return;
        };
        let current = self.site.unwrap_or(nearest);
        let to_current =
            tactics::path_length(plan.world, from, plan.site(current).plant, &mut self.buffer);
        let need = to_current / numbers.player.run_speed_mps
            + numbers.round.plant_seconds
            + numbers.play.late_round_seconds;
        if plan.seconds_left() < need {
            self.site = Some(if length < to_current {
                nearest
            } else {
                current
            });
            self.go_tick = plan.tick;
            self.alive_at_go = alive.len();
            self.set_step(Step::Execute, Why::TimeShort, plan.tick);
        }
    }

    fn check_collapse(&mut self, plan: &Plan<'_, '_>, alive: &[usize]) {
        if self.step != Step::Execute
            || alive.len() + 2 > self.alive_at_go
            || plan.tactics.sites.len() < 2
        {
            return;
        }
        let Some(site) = self.site else {
            return;
        };
        let from = self.carrier_at(plan, alive);
        if plan.places().site_at(from) == Some(site) {
            return;
        }
        let other = (site + 1) % plan.tactics.sites.len();
        let numbers = plan.numbers;
        let length =
            tactics::path_length(plan.world, from, plan.site(other).plant, &mut self.buffer);
        let need = length / numbers.player.run_speed_mps
            + numbers.round.plant_seconds
            + numbers.play.late_round_seconds
            + numbers.play.gather_seconds[0];
        if plan.seconds_left() > need {
            self.site = Some(other);
            self.deadline = plan.tick + numbers.ticks(numbers.play.gather_seconds[0]);
            self.go_tick = u64::MAX;
            self.set_step(Step::Gather, Why::PlanDied, plan.tick);
        }
    }

    fn lurk_hold(&self, plan: &Plan<'_, '_>, site: usize) -> Hold {
        let st = plan.site(site);
        match st.ways.last() {
            Some(way) if st.ways.len() > 2 => Hold {
                at: way.attack_stage,
                face: way.hold.face,
            },
            _ => plan.tactics.mid_spot,
        }
    }

    fn attack_wait(&mut self, plan: &Plan<'_, '_>, alive: &[usize]) {
        let sites = plan.tactics.sites.len();
        let spacing = plan.numbers.play.spacing_m;
        for (k, &m) in alive.iter().enumerate() {
            let slot = self.slots[m];
            let spot = k % (sites + 1);
            let hold = if slot.duty == Duty::Lurk || spot == sites {
                plan.tactics.mid_spot
            } else {
                let way = &plan.site(spot).ways[0];
                Hold {
                    at: way.attack_stage,
                    face: way.hold.face,
                }
            };
            let at = spaced(hold.at, hold.face, k / (sites + 1), spacing);
            self.simple(
                plan,
                m,
                slot.duty,
                Step::Wait,
                Hold {
                    at,
                    face: hold.face,
                },
                true,
            );
            self.stagger(plan, m, k);
        }
        let mut best: Option<(usize, u32)> = None;
        let mut seen = false;
        for site in 0..sites {
            let pressure = self.pressure(site, plan.numbers);
            seen |= pressure >= plan.numbers.play.rotate_seen;
            if best.is_none_or(|(_, p)| pressure < p) {
                best = Some((site, pressure));
            }
        }
        if plan.tick >= self.pick_tick || seen {
            if let Some((site, _)) = best {
                self.site = Some(site);
            }
            self.deadline = plan.tick + plan.numbers.ticks(plan.numbers.play.gather_seconds[1]);
            self.set_step(
                Step::Gather,
                if seen { Why::Called } else { Why::Order },
                plan.tick,
            );
        }
    }

    fn attack_gather(&mut self, plan: &Plan<'_, '_>, alive: &[usize]) {
        let Some(site) = self.site else {
            return;
        };
        let st = plan.site(site);
        let spacing = plan.numbers.play.spacing_m;
        let members = plan.sense.members;
        let mut need = 0usize;
        let mut arrived = 0usize;
        for &m in alive {
            let slot = self.slots[m];
            if slot.duty == Duty::Lurk {
                let hold = self.lurk_hold(plan, site);
                self.simple(plan, m, slot.duty, Step::Gather, hold, true);
                continue;
            }
            let way = &st.ways[slot.group.min(st.ways.len() - 1)];
            let at = spaced(way.attack_stage, way.mid, slot.rank, spacing);
            self.simple(
                plan,
                m,
                slot.duty,
                Step::Gather,
                Hold {
                    at,
                    face: way.hold.face,
                },
                true,
            );
            self.stagger(plan, m, slot.rank + slot.group);
            need += 1;
            if flat_distance(members[m].position, at) <= spacing * 2.0 {
                arrived += 1;
            }
        }
        if arrived + 1 >= need.max(1) || plan.tick >= self.deadline {
            self.go_tick = plan.tick + plan.numbers.ticks(plan.numbers.play.utility_lead_seconds);
            self.alive_at_go = alive.len();
            self.set_step(Step::Execute, Why::Arrived, plan.tick);
        }
    }

    fn attack_execute(&mut self, plan: &Plan<'_, '_>, alive: &[usize]) {
        let Some(site) = self.site else {
            return;
        };
        let st = plan.site(site);
        if st.ways.is_empty() {
            return;
        }
        let numbers = plan.numbers;
        let play = &numbers.play;
        let stagger = numbers.ticks(play.go_stagger_seconds);
        let lead = numbers.ticks(play.utility_lead_seconds);
        let members = plan.sense.members;
        let go = if self.go_tick == u64::MAX {
            plan.tick
        } else {
            self.go_tick
        };
        let deep = &st.ways[st.defend_order.first().copied().unwrap_or(0)];
        for &m in alive {
            let slot = self.slots[m];
            let way = &st.ways[slot.group.min(st.ways.len() - 1)];
            let duty = if members[m].carrying_device && members[m].bot {
                Duty::Carrier
            } else {
                slot.duty
            };
            let (goal, face, go_tick, utility, careful) = match duty {
                Duty::Entry => {
                    let flash = Utility {
                        block: Block::Flash,
                        target: way.flash_at,
                        after_tick: go.saturating_sub(lead / 2),
                        line: Line {
                            kind: LineKind::FlashCorner,
                            site: Some(site),
                            choke: Some(way.choke),
                        },
                    };
                    (deep.post.at, deep.post.face, go, Some(flash), false)
                }
                Duty::Smoke => {
                    let smoke = Utility {
                        block: Block::Smoke,
                        target: way.smoke_at,
                        after_tick: go.saturating_sub(lead),
                        line: Line {
                            kind: LineKind::SmokeHold,
                            site: Some(site),
                            choke: Some(way.choke),
                        },
                    };
                    (
                        way.post.at,
                        way.post.face,
                        go + 2 * stagger,
                        Some(smoke),
                        false,
                    )
                }
                Duty::Carrier => (st.plant, way.post.face, go + stagger, None, false),
                Duty::Lurk => {
                    let hold = self.lurk_hold(plan, site);
                    (hold.at, hold.face, go, None, true)
                }
                Duty::Follow
                | Duty::Anchor
                | Duty::Mid
                | Duty::Retaker
                | Duty::Defuser
                | Duty::Cover
                | Duty::Saver => {
                    let post = &st.ways[slot.rank % st.ways.len()].post;
                    let at = spaced(
                        post.at,
                        post.face,
                        slot.rank / st.ways.len(),
                        play.spacing_m,
                    );
                    let reveal = (members[m].role == Role::Initiator).then_some(Utility {
                        block: Block::Reveal,
                        target: st.plant,
                        after_tick: go.saturating_sub(lead),
                        line: Line {
                            kind: LineKind::RevealSite,
                            site: Some(site),
                            choke: None,
                        },
                    });
                    (
                        at,
                        post.face,
                        go + stagger * (2 + slot.rank as u64),
                        reveal,
                        false,
                    )
                }
            };
            let step = if duty == Duty::Carrier
                && plan.places().site_at(members[m].position) == Some(site)
            {
                self.set_step(Step::Plant, Why::Arrived, plan.tick);
                Step::Plant
            } else {
                Step::Execute
            };
            let order = Order {
                duty,
                step,
                goal,
                zone: None,
                face,
                go_tick,
                careful,
                utility,
                hand_over: false,
                wait_contact: false,
            };
            self.put(plan, m, order);
        }
    }

    fn hand_over_orders(&mut self, plan: &Plan<'_, '_>) {
        let Some((from, to)) = self.hand_over else {
            return;
        };
        let members = plan.sense.members;
        let early = matches!(self.step, Step::Wait | Step::Gather);
        if !early || !members[from].alive || !members[to].alive || members[to].carrying_device {
            self.hand_over = None;
            return;
        }
        let eye = plan.eye();
        if members[from].carrying_device {
            let at = members[from].position;
            let face = members[to].position + eye;
            self.put(
                plan,
                from,
                Order {
                    hand_over: true,
                    wait_contact: false,
                    ..Order {
                        goal: at,
                        face,
                        go_tick: 0,
                        ..self.orders[from]
                    }
                },
            );
        }
    }

    pub(super) fn post_plant(&mut self, plan: &Plan<'_, '_>) {
        let DeviceSense::Planted { site, .. } = plan.sense.game.device else {
            return;
        };
        self.site = Some(site);
        let st = plan.site(site);
        if st.ways.is_empty() {
            return;
        }
        let spacing = plan.numbers.play.spacing_m;
        for (k, m) in plan.alive().into_iter().enumerate() {
            let post = st.ways[k % st.ways.len()].post;
            let at = spaced(post.at, post.face, k / st.ways.len(), spacing);
            let duty = self.slots[m].duty;
            self.simple(
                plan,
                m,
                duty,
                Step::PostPlant,
                Hold {
                    at,
                    face: post.face,
                },
                true,
            );
        }
    }
}
