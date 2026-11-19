engine::brick!(paths);

mod layout;
mod make;

use std::sync::Arc;

use engine::prelude::*;
use engine::{Error, ErrorCode};
use engine_core::SceneCx;
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::nav::Costs;
use crate::play::actors::{Actor, Body, Human};
use crate::play::events::arms::ArmsEvents;
use crate::play::numbers::{GameNumbers, Numbers};
use crate::play::paths::{Built, HazardKind, Hazards, MeshSource, Paths, Tuning};
use crate::play::world::Arena;

use layout::Layout;
use make::{Ground, PathsNumbers, Prepared};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FloorForm {
    pub half_size_m: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CookedForm {
    pub mesh: String,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PathsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor: Option<FloorForm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooked: Option<CookedForm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
    #[serde(skip)]
    prepared: Option<Arc<Prepared>>,
}

fn utf8(bytes: &[u8], at: &str, file: &str) -> Result<Arc<str>, Error> {
    std::str::from_utf8(bytes)
        .map(Arc::from)
        .map_err(|_| Error::new(ErrorCode::InvalidRecord, at.to_owned()).got(file))
}

impl PathsFile {
    fn checked(&self, path: &str) -> Result<(), Error> {
        match (&self.floor, &self.cooked) {
            (Some(_), None) | (None, Some(_)) => Ok(()),
            (Some(_), Some(_)) | (None, None) => {
                Err(Error::new(ErrorCode::InvalidValue, path.to_owned())
                    .variants(["floor", "cooked"]))
            }
        }
    }

    fn files(&self) -> Vec<String> {
        self.cooked
            .iter()
            .map(|cooked| cooked.mesh.clone())
            .chain(self.layout.iter().cloned())
            .collect()
    }

    fn ground(&self, cx: &SceneCx<'_>, path: &str) -> Result<Option<Ground>, Error> {
        match (&self.floor, &self.cooked) {
            (Some(floor), _) => Ok(Some(Ground::Floor {
                half_size: floor.half_size_m,
            })),
            (None, Some(cooked)) => {
                let at = format!("{path}.cooked.mesh");
                let Some(bytes) = cx.buffer(&cooked.mesh, &at)? else {
                    return Ok(None);
                };
                let mesh = engine_assets::MeshBuffer::parse(&bytes, &at)?;
                Ok(Some(Ground::Cooked {
                    positions: mesh
                        .vertices
                        .iter()
                        .map(|v| Vec3::from_array(v.position))
                        .collect(),
                    indices: mesh.indices.clone(),
                }))
            }
            (None, None) => Ok(None),
        }
    }

    fn prepared(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let Some(ground) = self.ground(cx, path)? else {
            return Ok(self);
        };
        let layout = match &self.layout {
            Some(file) => {
                let at = format!("{path}.layout");
                let Some(bytes) = cx.buffer(file, &at)? else {
                    return Ok(self);
                };
                let text = utf8(&bytes, &at, file)?;
                let parsed: Layout =
                    engine::bricks::parse_json(&text).map_err(|error| error.within(&at))?;
                Some((text, parsed))
            }
            None => None,
        };
        self.prepared = Some(Arc::new(Prepared { ground, layout }));
        Ok(self)
    }
}

#[derive(Resource, Default)]
struct State {
    file: Option<Arc<Prepared>>,
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
                    .check(PathsFile::checked)
                    .files(PathsFile::files)
                    .prepare(PathsFile::prepared),
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
    file: Option<&Prepared>,
    numbers: Option<&GameNumbers>,
    arena: &Arena,
) -> Result<Rebuilt, String> {
    let (Some(prepared), Some(numbers)) = (file, numbers) else {
        return Ok(None);
    };
    if !arena.ready() {
        return Ok(None);
    }
    let (nav, tables) = make::tables(numbers)?;
    let built = make::build(prepared, numbers, &nav, arena)?;
    Ok(Some((built, make::tuning(numbers, &nav), tables)))
}

fn adopt(
    files: Query<Ref<PathsFile>>,
    numbers: Res<Numbers>,
    arena: Res<Arena>,
    mut paths: ResMut<Paths>,
    mut state: ResMut<State>,
) {
    let mut dirty = false;
    for file in &files {
        if file.is_changed() {
            state.file = file.prepared.clone();
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
    match rebuild(state.file.as_deref(), numbers.get(), &arena) {
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
        islands: built.map_or(0, |b| b.mesh.report().islands.len() as u32),
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
