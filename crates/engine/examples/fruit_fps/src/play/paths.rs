use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use engine::prelude::Resource;
use engine_core::Asset;
use engine_nav::{Costs, NavMesh, Path, Scratch, Spot};
use glam::Vec3;

use crate::play::cover::Tactics;
use crate::play::layout::Layout;
use crate::play::numbers::NavNumbers;
use crate::play::world::Arena;

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
    pub layout: Option<Asset<Layout>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    pub nav: NavNumbers,
    pub radius: f32,
}

#[derive(Resource, Default)]
pub struct Paths {
    built: Option<Arc<Built>>,
    tuning: Option<Tuning>,
    costs: [Option<Costs>; 2],
    revision: u64,
    queries: AtomicU32,
    expanded: AtomicU32,
    scratch: Mutex<Scratch>,
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

    pub fn layout(&self) -> Option<Asset<Layout>> {
        self.built()?.layout
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
        let mut scratch = self.scratch.lock().unwrap_or_else(PoisonError::into_inner);
        let path = built.mesh.path(
            start,
            goal,
            tuning.radius,
            self.costs_for(team)?,
            &mut scratch,
        );
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

    pub fn best_cover(
        &self,
        arena: &Arena,
        threat: Vec3,
        near: Vec3,
        max_distance: f32,
    ) -> Option<Vec3> {
        let (built, tuning) = self.parts()?;
        let near = self.locate(near)?;
        self.count(0);
        built
            .tactics
            .best_cover_from(
                &built.mesh,
                threat,
                near,
                tuning.radius,
                max_distance,
                |eye, target| arena.sees(eye, target),
            )
            .map(|spot| spot.point)
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
