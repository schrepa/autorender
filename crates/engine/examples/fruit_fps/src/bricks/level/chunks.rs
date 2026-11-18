use std::ops::Range;

use engine::prelude::*;
use engine_core::{
    AssetLibrary, DrawKind, DrawnTable, GeometryHandle, Layers, MaterialHandle, PassBit, PassMask,
    ViewFacts,
};

#[derive(Clone, Debug)]
pub(super) struct Chunk {
    pub mesh: Option<GeometryHandle>,
    pub material: Option<MaterialHandle>,
    pub instances: Range<u32>,
    pub passes: Option<PassMask>,
    pub layers: Layers,
    pub bounds: Option<[Vec3; 2]>,
    pub vertices: u64,
}

impl Chunk {
    pub fn drawn_in(&self, pass: Option<PassBit>) -> bool {
        self.passes
            .is_none_or(|mask| pass.is_some_and(|bit| mask.has(bit)))
    }

    fn nearest(&self, eye: Vec3) -> f32 {
        self.bounds.map_or(0.0, |[low, high]| {
            eye.max(low).min(high).distance_squared(eye)
        })
    }
}

pub(super) fn listed(
    table: &DrawnTable,
    library: Option<&AssetLibrary>,
    kind: DrawKind,
    into: &mut Vec<Chunk>,
) {
    into.clear();
    into.extend(table.batches(kind).map(|batch| {
        let mesh = batch
            .geometry
            .zip(library)
            .and_then(|(geometry, library)| library.mesh(geometry));
        let count = batch.instances.len() as u64;
        Chunk {
            mesh: batch.geometry,
            material: batch.material,
            instances: batch.instances.clone(),
            passes: batch.passes,
            layers: batch.layers,
            bounds: mesh.filter(|_| count == 1).map(|mesh| {
                let bounds = mesh.bounds();
                [bounds.min(), bounds.max()]
            }),
            vertices: mesh.map_or(0, |mesh| mesh.indices().len() as u64) * count,
        }
    }));
}

fn outside(view_projection: Mat4, [low, high]: [Vec3; 2]) -> bool {
    let mut beyond = [true; 5];
    for corner in 0..8_u32 {
        let point = Vec3::new(
            if corner & 1 == 0 { low.x } else { high.x },
            if corner & 2 == 0 { low.y } else { high.y },
            if corner & 4 == 0 { low.z } else { high.z },
        );
        let clip = view_projection * point.extend(1.0);
        beyond[0] &= clip.x < -clip.w;
        beyond[1] &= clip.x > clip.w;
        beyond[2] &= clip.y < -clip.w;
        beyond[3] &= clip.y > clip.w;
        beyond[4] &= clip.w <= 0.0;
    }
    beyond.contains(&true)
}

pub(super) fn ordered(chunks: &[Chunk], facts: &ViewFacts, into: &mut Vec<u32>) {
    into.clear();
    into.extend((0..chunks.len() as u32).filter(|&at| {
        let chunk = &chunks[at as usize];
        chunk.layers.overlaps(facts.layers)
            && chunk
                .bounds
                .is_none_or(|bounds| !outside(facts.view_projection, bounds))
    }));
    into.sort_by(|&a, &b| {
        let near = |at: u32| chunks[at as usize].nearest(facts.eye);
        near(a).total_cmp(&near(b))
    });
}
