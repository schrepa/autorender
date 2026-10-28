use bevy_ecs::prelude::{Entity, World};
use glam::Mat4;

use crate::retained_render::RetainedRender;
use crate::{
    GlobalTransforms, PresentedTransforms, TransformIndex, WorldCells, lights, particles, probes,
};

pub(crate) struct Extractions {
    lights: lights::LightExtraction,
    particles: particles::ParticleExtraction,
    probes: probes::ProbeExtraction,
}

impl Extractions {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            lights: lights::LightExtraction::new(world),
            particles: particles::ParticleExtraction::new(world),
            probes: probes::ProbeExtraction::new(world),
        }
    }

    pub(crate) fn extract(
        &mut self,
        retained: &mut RetainedRender,
        world: &World,
        aspect: f32,
        gpu: bool,
    ) {
        retained.extract_deformation(world);
        retained.extract_lights_and_particles(world, &mut self.lights, &mut self.particles, gpu);
        retained.extract_probes(world, &mut self.probes);
        retained.extract_terrain(world, aspect);
        retained.set_streamed(
            world
                .get_resource::<WorldCells>()
                .map(WorldCells::stats)
                .unwrap_or_default(),
            world
                .get_resource::<crate::StaticTables>()
                .map(crate::StaticTables::render)
                .unwrap_or_default(),
        );
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Poses<'w> {
    Prepared(&'w PresentedTransforms),
    Authoritative(&'w GlobalTransforms),
}

impl<'w> Poses<'w> {
    pub(crate) fn of(world: &'w World, gpu: bool) -> Self {
        if gpu {
            Self::Authoritative(world.resource())
        } else {
            Self::Prepared(world.resource())
        }
    }

    pub(crate) fn model(self, entity: Entity) -> Option<Mat4> {
        match self {
            Self::Prepared(poses) => poses.get(entity)?.matrix(),
            Self::Authoritative(poses) => poses.get(entity)?.matrix(),
        }
    }

    pub(crate) fn model_indexed(self, entity: Entity, hint: &mut TransformIndex) -> Option<Mat4> {
        match self {
            Self::Prepared(poses) => poses.get_indexed(entity, hint)?.matrix(),
            Self::Authoritative(poses) => poses.get_indexed(entity, hint)?.matrix(),
        }
    }
}

#[derive(Default)]
pub(crate) struct Nearest {
    keys: Vec<u64>,
}

const POSITION: u64 = u32::MAX as u64;

impl Nearest {
    pub(crate) fn nearest_first(
        &mut self,
        distances: impl IntoIterator<Item = f32>,
        kept: usize,
    ) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.select(distances, kept);
        self.keys.sort_unstable();
        self.positions()
    }

    pub(crate) fn kept_in_order(
        &mut self,
        distances: impl ExactSizeIterator<Item = f32>,
        kept: usize,
    ) -> impl ExactSizeIterator<Item = usize> + '_ {
        if distances.len() <= kept {
            self.keys.clear();
            self.keys.extend(0..distances.len() as u64);
        } else {
            self.select(distances, kept);
            self.keys.sort_unstable_by_key(|key| key & POSITION);
        }
        self.positions()
    }

    fn select(&mut self, distances: impl IntoIterator<Item = f32>, kept: usize) {
        self.keys.clear();
        self.keys.extend(
            distances
                .into_iter()
                .enumerate()
                .map(|(position, distance)| {
                    u64::from(total_order(distance)) << 32 | position as u64
                }),
        );
        if kept < self.keys.len() {
            self.keys.select_nth_unstable(kept);
            self.keys.truncate(kept);
        }
    }

    fn positions(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.keys.iter().map(|key| (key & POSITION) as usize)
    }
}

fn total_order(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}
