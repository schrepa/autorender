engine::brick!(paths);

mod floor;
mod layout;
mod make;

use std::sync::Arc;

use engine::Error;
use engine::prelude::*;
use engine_core::{Asset, Assets};
use engine_nav::Costs;
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::actors::{Actor, Body, Human};
use crate::play::events::arms::ArmsEvents;
use crate::play::files::Triangles;
use crate::play::layout::Layout;
use crate::play::numbers::{GameNumbers, Numbers, PathsNumbers};
use crate::play::paths::{Built, HazardKind, Hazards, MeshSource, Paths, Tuning};
use crate::play::world::{Arena, floor_or_cooked};

use make::{Ground, Placed};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FloorForm {
    pub half_size_m: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CookedForm {
    pub mesh: Asset<Triangles>,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PathsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor: Option<FloorForm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooked: Option<CookedForm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<Asset<Layout>>,
}

impl PathsFile {
    fn checked(&self, path: &str) -> Result<(), Error> {
        floor_or_cooked(self.floor.as_ref(), self.cooked.as_ref(), path)
    }

    fn read<'a>(&self, assets: &'a Assets) -> Option<(Ground<'a>, Placed<'a>)> {
        let ground = match (&self.floor, &self.cooked) {
            (Some(floor), _) => Ground::Floor {
                half_size: floor.half_size_m,
            },
            (None, Some(cooked)) => Ground::Cooked(assets.get(cooked.mesh).ok()?),
            (None, None) => return None,
        };
        let placed = match self.layout {
            Some(layout) => Some((layout, assets.get(layout).ok()?)),
            None => None,
        };
        Some((ground, placed))
    }
}

#[derive(Resource, Default)]
struct State {
    file: Option<PathsFile>,
    numbers_revision: u64,
    arena_revision: u64,
    numbers: Option<PathsNumbers>,
    problem: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Mark {
    centre: Vec3,
    team: usize,
    tick: u64,
}

#[derive(Resource, Default)]
struct Heat {
    marks: Vec<Mark>,
    read_tick: Option<u64>,
    calm: Option<Hazards>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlaceStats {
    pub name: String,
    pub callout: String,
}

stats! {
    #[section("paths")]
    pub struct PathsStats {
        pub ready: bool,
        pub source: String,
        pub polygons: u32,
        pub portals: u32,
        pub islands: u32,
        pub empty_triangles: u32,
        pub queries: u32,
        pub expanded: u32,
        pub hazards: u32,
        pub heat: u32,
        pub human_place: Option<PlaceStats>,
        pub problem: Option<String>,
    }
}

pub struct PathsBrick;

impl Brick for PathsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<PathsFile>("paths")
                    .schema(engine_core::schema_of::<PathsFile>)
                    .check(PathsFile::checked),
            )
            .resource::<Paths>()
            .resource::<Hazards>()
            .resource::<State>()
            .resource::<Heat>()
            .systems(Schedule::BeforeDraw, adopt)
            .systems(Schedule::FixedSimulation, overlay)
            .dev(|dev| {
                dev.stats::<PathsStats>()
                    .systems(Schedule::Presentation, report);
            });
    }
}

type Rebuilt = Option<(Built, Tuning, Option<PathsNumbers>)>;

fn rebuild(
    file: Option<&PathsFile>,
    numbers: Option<&GameNumbers>,
    (arena, assets): (&Arena, &Assets),
) -> Result<Rebuilt, String> {
    let (Some(read), Some(numbers)) = (file.and_then(|file| file.read(assets)), numbers) else {
        return Ok(None);
    };
    if !arena.ready() {
        return Ok(None);
    }
    let built = make::build(read, numbers, arena)?;
    Ok(Some((built, make::tuning(numbers), numbers.paths.clone())))
}

fn adopt(
    (files, assets): (Query<Ref<PathsFile>>, Res<Assets>),
    numbers: Res<Numbers>,
    arena: Res<Arena>,
    mut paths: ResMut<Paths>,
    mut state: ResMut<State>,
) {
    let mut dirty = false;
    for file in &files {
        if file.is_changed() {
            state.file = Some(PathsFile::clone(&file));
            dirty = true;
        }
    }
    if numbers.revision() != state.numbers_revision || arena.revision() != state.arena_revision {
        state.numbers_revision = numbers.revision();
        state.arena_revision = arena.revision();
        dirty = true;
    }
    if !dirty {
        return;
    }
    match rebuild(state.file.as_ref(), numbers.get(), (&arena, &assets)) {
        Ok(Some((built, tuning, tables))) => {
            paths.adopt(Some(Arc::new(built)), Some(tuning));
            state.problem = tables
                .is_none()
                .then(|| "numbers.json has no `paths` table: no hazard or heat costs".to_owned());
            state.numbers = tables;
        }
        Ok(None) => {
            paths.adopt(None, None);
            state.numbers = None;
            state.problem = None;
        }
        Err(problem) => {
            paths.adopt(None, None);
            state.numbers = None;
            state.problem = Some(problem);
        }
    }
}

fn hazard_factor(numbers: &PathsNumbers, kind: HazardKind) -> f32 {
    let factors = &numbers.hazard_factors;
    match kind {
        HazardKind::Damage => factors.damage,
        HazardKind::Strike => factors.strike,
        HazardKind::Slow => factors.slow,
        HazardKind::NearSight => factors.near_sight,
        HazardKind::Reveal => factors.reveal,
    }
}

struct Overlay<'a> {
    built: &'a Built,
    hazards: &'a Hazards,
    heat: &'a Heat,
    numbers: &'a PathsNumbers,
    tick: u64,
    span: f32,
}

fn team_costs(overlay: &Overlay<'_>, team: usize) -> Costs {
    let mesh = &overlay.built.mesh;
    let numbers = overlay.numbers;
    let mut costs = overlay.built.plain.clone();
    for hazard in overlay.hazards.teams.get(team).into_iter().flatten() {
        costs.scale_disc(
            mesh,
            hazard.centre,
            hazard.radius,
            hazard_factor(numbers, hazard.kind),
        );
    }
    for smoke in &overlay.hazards.smokes {
        costs.scale_disc(
            mesh,
            smoke.centre,
            smoke.radius,
            numbers.hazard_factors.smoke,
        );
    }
    for mark in overlay.heat.marks.iter().filter(|mark| mark.team == team) {
        let fade = 1.0 - (overlay.tick - mark.tick) as f32 / overlay.span;
        let factor = 1.0 + (numbers.heat_factor - 1.0) * fade;
        costs.scale_disc(mesh, mark.centre, numbers.heat_radius_m, factor);
    }
    costs
}

fn remember_deaths(
    heat: &mut Heat,
    events: Option<&ArmsEvents>,
    actors: &Query<(&Actor, &Body)>,
    tick: u64,
) {
    let Some(events) = events else {
        return;
    };
    if heat.read_tick == Some(events.tick) {
        return;
    }
    heat.read_tick = Some(events.tick);
    for death in &events.deaths {
        if let Some((actor, body)) = actors.iter().find(|(actor, _)| actor.slot == death.victim) {
            heat.marks.push(Mark {
                centre: body.feet,
                team: actor.team.index(),
                tick,
            });
        }
    }
}

fn overlay(
    time: Res<FixedTime>,
    hazards: Res<Hazards>,
    events: Option<Res<ArmsEvents>>,
    actors: Query<(&Actor, &Body)>,
    state: Res<State>,
    mut heat: ResMut<Heat>,
    mut paths: ResMut<Paths>,
) {
    paths.reset_counts();
    let (Some(numbers), Some(built)) = (&state.numbers, paths.built()) else {
        return;
    };
    let tick = time.tick;
    remember_deaths(&mut heat, events.as_deref(), &actors, tick);
    let span = numbers.heat_seconds / time.delta_seconds;
    heat.marks.retain(|mark| ((tick - mark.tick) as f32) < span);
    let calm = heat.marks.is_empty();
    if calm && paths.costs(0).is_some() && heat.calm.as_ref() == Some(&*hazards) {
        return;
    }
    heat.calm = calm.then(|| hazards.clone());
    let overlay = Overlay {
        built,
        hazards: &hazards,
        heat: &heat,
        numbers,
        tick,
        span,
    };
    let teams = [team_costs(&overlay, 0), team_costs(&overlay, 1)];
    for (team, costs) in teams.into_iter().enumerate() {
        paths.set_costs(team, costs);
    }
}

fn report(
    paths: Res<Paths>,
    state: Res<State>,
    heat: Res<Heat>,
    hazards: Res<Hazards>,
    human: Query<&Body, With<Human>>,
    mut stats: Stats<PathsStats>,
) {
    let built = paths.built();
    let (queries, expanded) = paths.counts();
    let source = match built.map(|built| built.source) {
        Some(MeshSource::Floor) => "floor",
        Some(MeshSource::Cooked) => "cooked",
        None => "none",
    };
    stats.show(|| PathsStats {
        ready: paths.ready(),
        source: source.to_owned(),
        polygons: built.map_or(0, |b| b.mesh.polygon_count() as u32),
        portals: built.map_or(0, |b| b.mesh.report().portals as u32),
        islands: built.map_or(0, |b| b.mesh.report().islands as u32),
        empty_triangles: built.map_or(0, |b| b.mesh.report().empty_triangles as u32),
        queries,
        expanded,
        hazards: (hazards.teams.iter().map(Vec::len).sum::<usize>() + hazards.smokes.len()) as u32,
        heat: heat.marks.len() as u32,
        human_place: human.iter().next().and_then(|body| {
            paths.place_at(body.feet).map(|place| PlaceStats {
                name: place.name.clone(),
                callout: place.callout.clone(),
            })
        }),
        problem: state.problem.clone(),
    });
}
