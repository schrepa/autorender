use glam::Vec3;

use super::aim::{PI, flat, flat_distance, sin_cos};
use super::numbers::BotNumbers;
use super::rng::Rng;
use super::sense::World;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steering {
    pub wish: Vec3,
    pub arrived: bool,
    pub stuck: bool,
    pub ahead: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mover {
    corners: Vec<Vec3>,
    next: usize,
    goal: Option<Vec3>,
    detour: Option<Vec3>,
    best: f32,
    progress_tick: u64,
    stuck_count: u32,
    lane: f32,
    lift: Vec3,
}

impl Mover {
    pub fn new() -> Mover {
        Mover {
            corners: Vec::with_capacity(32),
            next: 0,
            goal: None,
            detour: None,
            best: f32::INFINITY,
            progress_tick: 0,
            stuck_count: 0,
            lane: 0.0,
            lift: Vec3::ZERO,
        }
    }

    pub fn set_lane(&mut self, lane: f32, lift: Vec3) {
        self.lane = lane;
        self.lift = lift;
    }

    pub fn clear(&mut self) {
        self.corners.clear();
        self.next = 0;
        self.goal = None;
        self.detour = None;
        self.best = f32::INFINITY;
        self.stuck_count = 0;
    }

    fn plan(&mut self, from: Vec3, to: Vec3, world: &dyn World, tick: u64) {
        if world.path(from, to, &mut self.corners) {
            self.keep_lane(from, world);
        } else {
            self.corners.clear();
            self.corners.push(to);
        }
        self.next = 0;
        self.best = f32::INFINITY;
        self.progress_tick = tick;
    }

    fn keep_lane(&mut self, from: Vec3, world: &dyn World) {
        if self.lane <= 0.0 || self.corners.len() < 2 {
            return;
        }
        let lift = self.lift;
        for index in 0..self.corners.len() - 1 {
            let before = if index == 0 {
                from
            } else {
                self.corners[index - 1]
            };
            let corner = self.corners[index];
            let after = self.corners[index + 1];
            let turn = flat(after - corner).normalize_or_zero()
                - flat(corner - before).normalize_or_zero();
            let outward = -turn.normalize_or_zero();
            if outward == Vec3::ZERO {
                continue;
            }
            for share in [1.0, 0.5] {
                let moved = corner + outward * (self.lane * share);
                if world.sees(before + lift, moved + lift) && world.sees(moved + lift, after + lift)
                {
                    self.corners[index] = moved;
                    break;
                }
            }
        }
    }

    pub fn remaining(&self, from: Vec3) -> f32 {
        let Some(first) = self.corners.get(self.next) else {
            return 0.0;
        };
        let mut length = flat_distance(from, *first);
        for pair in self.corners[self.next..].windows(2) {
            length += flat_distance(pair[0], pair[1]);
        }
        length
    }

    fn unstick(
        &mut self,
        from: Vec3,
        numbers: &BotNumbers,
        rng: &mut Rng,
        world: &dyn World,
        tick: u64,
    ) {
        self.stuck_count += 1;
        if self.stuck_count % 2 == 1 {
            if let Some(goal) = self.goal {
                self.plan(from, self.detour.unwrap_or(goal), world, tick);
            }
            return;
        }
        let (sin, cos) = sin_cos(rng.unit() * 2.0 * PI);
        let point = from + Vec3::new(cos, 0.0, sin) * numbers.play.unstick_m;
        self.detour = Some(point);
        self.plan(from, point, world, tick);
    }

    pub fn steer(
        &mut self,
        from: Vec3,
        goal: Vec3,
        world: &dyn World,
        numbers: &BotNumbers,
        rng: &mut Rng,
        tick: u64,
    ) -> Steering {
        let play = &numbers.play;
        let moved_goal = self
            .goal
            .is_none_or(|g| flat_distance(g, goal) > play.repath_m);
        if moved_goal {
            self.goal = Some(goal);
            self.detour = None;
            self.stuck_count = 0;
            self.plan(from, goal, world, tick);
        }
        if flat_distance(from, goal) <= play.arrive_m {
            self.best = f32::INFINITY;
            self.progress_tick = tick;
            return Steering {
                wish: Vec3::ZERO,
                arrived: true,
                stuck: false,
                ahead: goal,
            };
        }
        while self.next < self.corners.len()
            && flat_distance(from, self.corners[self.next]) <= play.arrive_m
        {
            self.next += 1;
        }
        if self.next >= self.corners.len() {
            if self.detour.take().is_some() {
                self.plan(from, goal, world, tick);
            } else {
                self.corners.clear();
                self.corners.push(goal);
                self.next = 0;
            }
        }
        let remaining = self.remaining(from);
        let mut stuck = false;
        if remaining < self.best - play.stuck_progress_m {
            self.best = remaining;
            self.progress_tick = tick;
        } else if tick.saturating_sub(self.progress_tick) > numbers.ticks(play.stuck_seconds) {
            self.unstick(from, numbers, rng, world, tick);
            stuck = true;
        }
        let ahead = self.corners.get(self.next).copied().unwrap_or(goal);
        let wish = flat(ahead - from).normalize_or_zero();
        Steering {
            wish,
            arrived: false,
            stuck,
            ahead,
        }
    }

    pub fn reset_progress(&mut self, tick: u64) {
        self.best = f32::INFINITY;
        self.progress_tick = tick;
    }
}
