use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use engine::prelude::Resource;
use glam::Vec3;

use crate::nav::{Costs, CoverSpot, Eyes, NavMesh, NavNumbers, Path, Spot, Tactics};
use crate::play::world::Arena;
use crate::rules::Side;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HazardKind {
    Damage,
    Strike,
    Slow,
    NearSight,
    Reveal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hazard {
    pub kind: HazardKind,
    pub centre: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmokeCloud {
    pub centre: Vec3,
    pub radius: f32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Hazards {
    pub teams: [Vec<Hazard>; 2],
    pub smokes: Vec<SmokeCloud>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaceName {
    pub name: String,
    pub callout: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ZoneArea {
    pub place: PlaceName,
    pub min: Vec3,
    pub max: Vec3,
}

impl ZoneArea {
    pub fn covers(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    fn holds(&self, point: Vec3, below: f32) -> bool {
        self.covers(point) && point.y >= self.min.y - below && point.y <= self.max.y
    }

    pub fn area(&self) -> f32 {
        (self.max.x - self.min.x) * (self.max.z - self.min.z)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hold {
    pub site: String,
    pub choke: String,
    pub cover: String,
    pub at: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Watch {
    pub at: Vec3,
    pub zone: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnIsland {
    pub side: Side,
    pub at: Vec3,
    pub island: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshSource {
    Floor,
    Cooked,
}

#[derive(Clone, Debug)]
pub struct Built {
    pub source: MeshSource,
    pub mesh: NavMesh,
    pub plain: Costs,
    pub tactics: Tactics,
    pub zones: Vec<ZoneArea>,
    pub places: Vec<PlaceName>,
    pub holds: Vec<Hold>,
    pub watches: Vec<Watch>,
    pub spawn_islands: Vec<SpawnIsland>,
    pub layout: Option<Arc<str>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    pub nav: NavNumbers,
    pub radius: f32,
    pub run_speed: f32,
    pub eyes: Eyes,
}

#[derive(Resource, Default)]
pub struct Paths {
    built: Option<Arc<Built>>,
    tuning: Option<Tuning>,
    costs: [Option<Costs>; 2],
    revision: u64,
    queries: AtomicU32,
    expanded: AtomicU32,
}

impl Paths {
    pub fn adopt(&mut self, built: Option<Arc<Built>>, tuning: Option<Tuning>) {
        self.built = built;
        self.tuning = tuning;
        self.costs = [None, None];
        self.revision += 1;
    }

    pub fn set_costs(&mut self, team: usize, costs: Costs) {
        if let Some(slot) = self.costs.get_mut(team) {
            *slot = Some(costs);
        }
    }

    pub fn reset_counts(&mut self) {
        *self.queries.get_mut() = 0;
        *self.expanded.get_mut() = 0;
    }

    pub fn counts(&self) -> (u32, u32) {
        (
            self.queries.load(Ordering::Relaxed),
            self.expanded.load(Ordering::Relaxed),
        )
    }

    fn count(&self, expanded: u32) {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.expanded.fetch_add(expanded, Ordering::Relaxed);
    }

    pub fn ready(&self) -> bool {
        self.built.is_some() && self.tuning.is_some()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn built(&self) -> Option<&Built> {
        self.built.as_deref()
    }

    pub fn tuning(&self) -> Option<&Tuning> {
        self.tuning.as_ref()
    }

    pub fn mesh(&self) -> Option<&NavMesh> {
        self.built().map(|built| &built.mesh)
    }

    pub fn tactics(&self) -> Option<&Tactics> {
        self.built().map(|built| &built.tactics)
    }

    pub fn radius(&self) -> f32 {
        self.tuning.as_ref().map_or(0.0, |tuning| tuning.radius)
    }

    pub fn layout(&self) -> Option<&Arc<str>> {
        self.built()?.layout.as_ref()
    }

    pub fn places(&self) -> &[PlaceName] {
        self.built().map_or(&[], |built| &built.places)
    }

    pub fn holds(&self) -> &[Hold] {
        self.built().map_or(&[], |built| &built.holds)
    }

    pub fn watches(&self) -> &[Watch] {
        self.built().map_or(&[], |built| &built.watches)
    }

    pub fn spawn_islands(&self) -> &[SpawnIsland] {
        self.built().map_or(&[], |built| &built.spawn_islands)
    }

    pub fn costs(&self, team: usize) -> Option<&Costs> {
        self.costs.get(team)?.as_ref()
    }

    fn parts(&self) -> Option<(&Built, &Tuning)> {
        Some((self.built.as_deref()?, self.tuning.as_ref()?))
    }

    fn costs_for(&self, team: Option<usize>) -> Option<&Costs> {
        let built = self.built()?;
        Some(
            team.and_then(|team| self.costs(team))
                .unwrap_or(&built.plain),
        )
    }

    pub fn locate(&self, point: Vec3) -> Option<Spot> {
        let (built, tuning) = self.parts()?;
        built.mesh.locate(
            point,
            tuning.nav.locate_height_m,
            tuning.nav.locate_distance_m,
        )
    }

    pub fn nearest(&self, point: Vec3, reach_m: f32) -> Option<Vec3> {
        let (built, tuning) = self.parts()?;
        self.count(0);
        built
            .mesh
            .locate(point, tuning.nav.locate_height_m, reach_m)
            .map(|spot| spot.point)
    }

    pub fn path(&self, from: Vec3, to: Vec3, team: Option<usize>) -> Option<Path> {
        let (built, tuning) = self.parts()?;
        let (start, goal) = (self.locate(from)?, self.locate(to)?);
        let path = built
            .mesh
            .path(start, goal, tuning.radius, self.costs_for(team)?);
        self.count(path.as_ref().map_or(0, |path| path.expanded));
        path
    }

    pub fn path_into(
        &self,
        from: Vec3,
        to: Vec3,
        team: Option<usize>,
        corners: &mut Vec<Vec3>,
    ) -> bool {
        corners.clear();
        let Some(path) = self.path(from, to, team) else {
            return false;
        };
        corners.extend(path.corners.iter().skip(1));
        true
    }

    pub fn path_cost(&self, from: Vec3, to: Vec3, team: Option<usize>) -> Option<f32> {
        let (built, tuning) = self.parts()?;
        let (start, goal) = (self.locate(from)?, self.locate(to)?);
        self.count(0);
        built
            .mesh
            .path_cost(start, goal, tuning.radius, self.costs_for(team)?)
    }

    pub fn raycast(&self, from: Vec3, to: Vec3) -> Option<Vec3> {
        let (built, _) = self.parts()?;
        self.count(0);
        match self.locate(from) {
            Some(start) => built.mesh.raycast(start, to),
            None => Some(from),
        }
    }

    pub fn cover_from(
        &self,
        arena: &Arena,
        threat: Vec3,
        near: Vec3,
        max_distance: f32,
    ) -> Vec<CoverSpot> {
        let (Some((built, tuning)), Some(near)) = (self.parts(), self.locate(near)) else {
            return Vec::new();
        };
        self.count(0);
        built.tactics.positions_with_cover_from(
            &built.mesh,
            threat,
            near,
            tuning.eyes,
            tuning.radius,
            max_distance,
            |eye, target| arena.sees(eye, target),
        )
    }

    pub fn best_cover(
        &self,
        arena: &Arena,
        threat: Vec3,
        near: Vec3,
        max_distance: f32,
    ) -> Option<Vec3> {
        self.cover_from(arena, threat, near, max_distance)
            .first()
            .map(|cover| cover.spot.point)
    }

    pub fn random_point_in(&self, place: &str, seed: u64) -> Option<Vec3> {
        let built = self.built()?;
        let index = built.tactics.place(place)?;
        let polygons = &built.tactics.places()[index].polygons;
        self.count(0);
        built
            .mesh
            .random_point(seed, |id| polygons.binary_search(&id).is_ok())
            .map(|spot| spot.point)
    }

    pub fn travel_seconds(&self, from_place: &str, to_place: &str) -> Option<f32> {
        let (built, tuning) = self.parts()?;
        let from = built.tactics.place(from_place)?;
        let to = built.tactics.place(to_place)?;
        let length = built.tactics.travel(from, to)?;
        (tuning.run_speed > 0.0).then(|| length / tuning.run_speed)
    }

    pub fn place_at(&self, point: Vec3) -> Option<&PlaceName> {
        let (built, tuning) = self.parts()?;
        built
            .zones
            .iter()
            .filter(|zone| zone.holds(point, tuning.nav.locate_height_m))
            .min_by(|a, b| a.area().total_cmp(&b.area()))
            .map(|zone| &zone.place)
    }
}
