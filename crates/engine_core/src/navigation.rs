//! Navigation agents: entities that walk over terrains to a
//! destination. When an agent's destination changes, the shortest walk to
//! it from where it stands is searched over the generated terrain under it,
//! in steps: every agent's search shares the fixed tick's
//! budget of [`Navigation::samples_per_tick`], so long walks take several
//! ticks. Each fixed tick a walking agent moves along its walk at its
//! speed, on the ground (the finest resident tile's height, else the
//! walk's), turning toward where it walks; it arrives at the walk's end.
//! Agents walk unparented entities; give them `InterpolateTransform` to
//! present smoothly between ticks. Where the host runs workers,
//! they search the walks instead, the tick's budget untouched, and
//! agents start walking once a worker hands its walk back.

use bevy_ecs::prelude::*;
use engine_terrain::jobs::WALK_JOB;
use engine_terrain::{NavOptions, NavPath, WalkSearch};
use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_range};
use crate::math::{clamp, max};
use crate::rules::{ExtraRules, settings};
use crate::{FixedTime, JobId, Jobs, Parent, Terrains, Transform};

settings! {
    /// Navigation. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneNavigation")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Navigation {
        /// Whether agents walk.
        pub enabled: bool = true => any,
        /// The samples of search work each fixed tick does at most over every
        /// walk being searched.
        pub samples_per_tick: u32 = 32_768 => at_least(1024),
    }
}

impl ExtraRules for Navigation {}

impl Navigation {
    /// What a world without these settings reads as: no agents walking.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// Walks its entity over the terrain under it to `destination`, world x
/// and z: `speed` meters a second (0.01 to 100), turning at
/// most `turn` degrees a second (1 to 10000), its origin `offset` meters
/// above the ground; around obstacles within `radius` meters (0 to 100),
/// up steps no steeper than `max_slope` (rise per meter, 0.01 to 10),
/// through water no deeper than `max_water` meters (0 to 100), over the
/// pyramid level `level` (0 the finest, up to 15). Changing the
/// destination searches anew; none stands still. Requires version 9.
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[require(NavWalk)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct NavAgent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destination: Option<[f32; 2]>,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 100.0)))]
    pub speed: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 1.0, max = 10000.0)))]
    pub turn: f32,
    pub offset: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100.0)))]
    pub radius: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 10.0)))]
    pub max_slope: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100.0)))]
    pub max_water: f32,
    #[cfg_attr(feature = "schema", schemars(range(max = 15)))]
    pub level: u32,
}

impl Default for NavAgent {
    fn default() -> Self {
        Self {
            destination: None,
            speed: 1.4,
            turn: 360.0,
            offset: 0.0,
            radius: 0.4,
            max_slope: 0.8,
            max_water: 0.6,
            level: 0,
        }
    }
}

impl NavAgent {
    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        if let Some(destination) = self.destination
            && !destination.iter().all(|value| value.is_finite())
        {
            return Err(InvalidSetting::new("destination", "must be finite"));
        }
        check_range("speed", self.speed, 0.01..=100.0)?;
        check_range("turn", self.turn, 1.0..=10_000.0)?;
        if !self.offset.is_finite() {
            return Err(InvalidSetting::new("offset", "must be finite"));
        }
        check_range("radius", self.radius, 0.0..=100.0)?;
        check_range("max_slope", self.max_slope, 0.01..=10.0)?;
        check_range("max_water", self.max_water, 0.0..=100.0)?;
        if self.level > 15 {
            return Err(InvalidSetting::new("level", "must be at most 15"));
        }
        Ok(())
    }

    /// Every value in range: out-of-range values clamp, unusable ones fall
    /// back to the defaults.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let within = |value: f32, low: f32, high: f32, fallback: f32| {
            if value.is_finite() {
                clamp(value, low, high)
            } else {
                fallback
            }
        };
        Self {
            destination: self
                .destination
                .filter(|destination| destination.iter().all(|value| value.is_finite())),
            speed: within(self.speed, 0.01, 100.0, defaults.speed),
            turn: within(self.turn, 1.0, 10_000.0, defaults.turn),
            offset: within(self.offset, -1.0e4, 1.0e4, 0.0),
            radius: within(self.radius, 0.0, 100.0, defaults.radius),
            max_slope: within(self.max_slope, 0.01, 10.0, defaults.max_slope),
            max_water: within(self.max_water, 0.0, 100.0, defaults.max_water),
            level: self.level.min(15),
        }
    }

    /// How its walks are searched.
    fn options(&self) -> NavOptions {
        NavOptions {
            level: self.level,
            max_slope: self.max_slope,
            radius: self.radius,
            max_water: self.max_water,
            ..NavOptions::default()
        }
    }
}

/// Where an agent is in its walk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum NavStatus {
    /// Without a destination.
    #[default]
    Idle,
    Searching,
    Walking,
    Arrived,
    /// No walk reaches the destination, or no terrain lies under it.
    Unreachable,
}

/// An agent's walk: its status, the destination it was
/// searched for, its points (world, on the ground), the next it walks to,
/// its length (meters) and the samples its search took.
#[derive(Component, Clone, Debug, Default)]
pub struct NavWalk {
    pub status: NavStatus,
    pub destination: Option<[f32; 2]>,
    pub points: Vec<Vec3>,
    pub next: usize,
    pub length: f32,
    pub searched: usize,
}

/// What navigation did at the last fixed tick: agents, and
/// those searching, walking, arrived and unable to reach their
/// destination; and the samples of search work the tick did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct NavStats {
    pub agents: u32,
    pub searching: u32,
    pub walking: u32,
    pub arrived: u32,
    pub unreachable: u32,
    pub work: u64,
}

/// Agents' walks being searched, and what navigation did
/// at the last fixed tick. Inserted by default.
#[derive(Resource, Default)]
pub struct NavAgents {
    // Each search: its agent, the terrain's origin and the search; and each
    // search on a worker, with its job.
    searches: Vec<(Entity, Vec3, WalkSearch)>,
    walks: Vec<(Entity, Vec3, JobId)>,
    // Where the next tick's searches start, round robin.
    turn: usize,
    stats: NavStats,
}

impl NavAgents {
    /// What navigation did at the last fixed tick.
    pub fn stats(&self) -> NavStats {
        self.stats
    }
}

/// Navigation's systems.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NavSystems {
    /// Searches walks and moves agents along them, in `FixedSimulation`.
    Walk,
}

type Agents<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static NavAgent,
        &'static mut NavWalk,
        &'static mut Transform,
    ),
    Without<Parent>,
>;

/// Searches the walks of agents whose destination changed, then moves
/// walking agents along theirs.
pub(crate) fn walk_agents(
    time: Res<FixedTime>,
    settings: Option<Res<Navigation>>,
    terrains: Res<Terrains>,
    mut jobs: Option<ResMut<Jobs>>,
    mut navigation: ResMut<NavAgents>,
    mut agents: Agents,
) {
    let settings = settings.map_or(
        Navigation {
            enabled: false,
            ..Navigation::default()
        },
        |settings| settings.sanitized(),
    );
    navigation.stats = NavStats::default();
    let mut jobs = jobs.as_deref_mut();
    if !settings.enabled {
        navigation.searches.clear();
        for (_, _, id) in navigation.walks.drain(..) {
            if let Some(jobs) = jobs.as_deref_mut() {
                jobs.cancel(id);
            }
        }
        return;
    }
    // New destinations start searches from where agents stand: on workers
    // where they run.
    for (entity, agent, mut walk, transform) in &mut agents {
        let agent = agent.sanitized();
        if walk.destination == agent.destination {
            continue;
        }
        navigation.searches.retain(|(owner, ..)| *owner != entity);
        navigation.walks.retain(|&(owner, _, id)| {
            if owner == entity
                && let Some(jobs) = jobs.as_deref_mut()
            {
                jobs.cancel(id);
            }
            owner != entity
        });
        *walk = NavWalk {
            destination: agent.destination,
            ..NavWalk::default()
        };
        let Some(destination) = agent.destination else {
            continue;
        };
        let from = Vec2::new(transform.translation.x, transform.translation.z);
        let to = Vec2::from(destination);
        let options = agent.options();
        let handed = jobs
            .as_deref_mut()
            .filter(|jobs| jobs.workers() > 0)
            .and_then(|jobs| {
                let (_, origin, input) = terrains.walk_job(from, to, &options)?;
                let priority = crate::job_priority(crate::GAMEPLAY_JOBS, 0);
                // Hosts may keep walks: their input names
                // the recipe by its content.
                let identity = input.clone();
                Some((
                    origin,
                    jobs.submit_cacheable(WALK_JOB, input, priority, identity),
                ))
            });
        if let Some((origin, id)) = handed {
            navigation.walks.push((entity, origin, id));
            walk.status = NavStatus::Searching;
            continue;
        }
        match terrains.walk_search(from, to, &options) {
            Some((_, origin, search)) => {
                navigation.searches.push((entity, origin, search));
                walk.status = NavStatus::Searching;
            }
            None => walk.status = NavStatus::Unreachable,
        }
    }
    search(&settings, &mut navigation, &mut agents);
    if let Some(jobs) = jobs {
        take_walks(jobs, &terrains, &mut navigation, &mut agents);
    }
    // Walkers move along their walks, on the ground.
    let seconds = time.delta_seconds;
    for (_, agent, mut walk, mut transform) in &mut agents {
        navigation.stats.agents += 1;
        match walk.status {
            NavStatus::Searching => navigation.stats.searching += 1,
            NavStatus::Arrived => navigation.stats.arrived += 1,
            NavStatus::Unreachable => navigation.stats.unreachable += 1,
            _ => {}
        }
        if walk.status != NavStatus::Walking {
            continue;
        }
        navigation.stats.walking += 1;
        let agent = agent.sanitized();
        step(&agent, &mut walk, &mut transform, &terrains, seconds);
    }
}

/// Advances the searches in progress by the tick's budget, round robin,
/// and gives finished ones' walks to their agents.
fn search(settings: &Navigation, navigation: &mut NavAgents, agents: &mut Agents) {
    let mut budget = u64::from(settings.samples_per_tick);
    let count = navigation.searches.len();
    let mut finished = Vec::new();
    for offset in 0..count {
        if budget == 0 {
            break;
        }
        let index = (navigation.turn + offset) % count;
        let (entity, origin, search) = &mut navigation.searches[index];
        if let Some(found) = search.step(&mut budget) {
            finished.push((index, *entity, *origin, found, search.searched()));
        }
    }
    navigation.turn = navigation.turn.wrapping_add(1);
    navigation.stats.work = u64::from(settings.samples_per_tick) - budget;
    finished.sort_by_key(|&(index, ..)| std::cmp::Reverse(index));
    for (index, entity, origin, found, searched) in finished {
        navigation.searches.swap_remove(index);
        let Ok((_, _, mut walk, _)) = agents.get_mut(entity) else {
            continue;
        };
        give(&mut walk, origin, found, searched);
    }
}

/// Gives agents the walks workers found; a walk that failed
/// there is searched here instead.
fn take_walks(
    jobs: &mut Jobs,
    terrains: &Terrains,
    navigation: &mut NavAgents,
    agents: &mut Agents,
) {
    for (entity, origin, id) in std::mem::take(&mut navigation.walks) {
        let Some(output) = jobs.take(id) else {
            navigation.walks.push((entity, origin, id));
            continue;
        };
        let Ok((_, agent, mut walk, transform)) = agents.get_mut(entity) else {
            continue;
        };
        match output
            .ok()
            .and_then(|bytes| engine_terrain::jobs::walk_output(&bytes).ok())
        {
            Some(found) => {
                let searched = found.as_ref().map_or(0, |path| path.searched);
                give(&mut walk, origin, found, searched);
            }
            None => {
                let agent = agent.sanitized();
                let from = Vec2::new(transform.translation.x, transform.translation.z);
                let search = agent.destination.and_then(|destination| {
                    terrains.walk_search(from, Vec2::from(destination), &agent.options())
                });
                match search {
                    Some((_, origin, search)) => navigation.searches.push((entity, origin, search)),
                    None => walk.status = NavStatus::Unreachable,
                }
            }
        }
    }
}

/// Gives `walk` the walk a search `found` (terrain-local, of a terrain at
/// `origin`) after `searched` samples, or no walk where none reaches.
fn give(walk: &mut NavWalk, origin: Vec3, found: Option<NavPath>, searched: usize) {
    walk.searched = searched;
    match found {
        Some(path) => {
            walk.points = path
                .points
                .iter()
                .map(|point| Vec3::from_array(*point) + origin)
                .collect();
            walk.next = walk.points.len().min(1);
            walk.length = path.length;
            walk.status = NavStatus::Walking;
        }
        None => walk.status = NavStatus::Unreachable,
    }
}

/// Moves `agent` `seconds` along `walk`, on the ground, turning toward
/// where it walks.
fn step(
    agent: &NavAgent,
    walk: &mut NavWalk,
    transform: &mut Transform,
    terrains: &Terrains,
    seconds: f32,
) {
    let mut position = transform.translation;
    let mut travel = agent.speed * seconds;
    let mut heading = None;
    while travel > 0.0 && walk.next < walk.points.len() {
        let target = walk.points[walk.next];
        let to = Vec2::new(target.x - position.x, target.z - position.z);
        let distance = to.length();
        if distance > 1.0e-4 {
            heading = Some(to / distance);
        }
        if distance <= travel {
            position.x = target.x;
            position.z = target.z;
            travel -= distance;
            walk.next += 1;
        } else {
            let along = to / distance * travel;
            position.x += along.x;
            position.z += along.y;
            travel = 0.0;
        }
    }
    // On the finest resident ground, else the walk's own; and on the walk's
    // own where it runs above the ground, over a bridge's deck.
    let walked = walk_height(walk, position);
    let ground = terrains
        .height_at(position.x, position.z)
        .map_or(walked, |ground| {
            if walked > ground + DECK_ABOVE {
                walked
            } else {
                ground
            }
        });
    position.y = ground + agent.offset;
    transform.translation = position;
    if walk.next >= walk.points.len() {
        walk.status = NavStatus::Arrived;
    }
    if let Some(heading) = heading {
        // Its -z turns toward where it walks.
        let target = Quat::from_rotation_y(f32::atan2(-heading.x, -heading.y));
        let angle = transform.rotation.angle_between(target);
        let turn = agent.turn.to_radians() * seconds;
        let share = if angle > turn { turn / angle } else { 1.0 };
        transform.rotation = transform
            .rotation
            .slerp(target, max(share, 0.0))
            .normalize();
    }
}

/// Meters a walk runs above the ground beneath before agents stand on the
/// walk's own height, as on a bridge's deck: decks clear the
/// water below them by more.
const DECK_ABOVE: f32 = 1.0;

/// The walk's height at `position` between the points around its next.
fn walk_height(walk: &NavWalk, position: Vec3) -> f32 {
    let last = walk.points.len().saturating_sub(1);
    let (Some(a), Some(b)) = (
        walk.points.get(walk.next.saturating_sub(1).min(last)),
        walk.points.get(walk.next.min(last)),
    ) else {
        return position.y;
    };
    let span = Vec2::new(b.x - a.x, b.z - a.z);
    let along = Vec2::new(position.x - a.x, position.z - a.z);
    let t = if span.length_squared() > 1.0e-8 {
        clamp(along.dot(span) / span.length_squared(), 0.0, 1.0)
    } else {
        1.0
    };
    a.y + (b.y - a.y) * t
}
