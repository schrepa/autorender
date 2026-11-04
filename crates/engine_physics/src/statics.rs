use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::collections::HashMap;
use engine_places::physics::CollisionLayers;
use engine_places::solids::FieldTile;
use glam::Vec3;
use parry3d::shape::Shape;
use rapier3d::prelude as rp;

use crate::shapes::solid_shape;
use crate::solids::{Solid, Solids};
use crate::solver::Solver;

const SLACK: f32 = 8.0;
const GROUND_FRICTION: f32 = 0.8;
const SOLID_FRICTION: f32 = 0.7;
const UPPER_PHI: u64 = 0x9e37_79b9_0000_0001;

#[derive(Clone, Copy, Default)]
struct EntityHash;

impl BuildHasher for EntityHash {
    type Hasher = EntityHasher;

    fn build_hasher(&self) -> EntityHasher {
        EntityHasher::default()
    }
}

#[derive(Default)]
struct EntityHasher {
    hash: u64,
}

impl Hasher for EntityHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("entity_hasher_takes_u64");
    }

    fn write_u64(&mut self, bits: u64) {
        self.hash = bits.wrapping_mul(UPPER_PHI);
    }
}

struct Ground {
    field: Arc<FieldTile>,
    handle: rp::ColliderHandle,
}

struct Mirror {
    solid: Solid,
    layers: CollisionLayers,
    handle: rp::ColliderHandle,
    seen: u64,
}

type PlacedKey = (Entity, u16, u16, [u32; 3]);

pub(crate) fn groups(layers: CollisionLayers) -> rp::InteractionGroups {
    rp::InteractionGroups::new(
        rp::Group::from_bits_retain(layers.memberships),
        rp::Group::from_bits_retain(layers.filter),
        rp::InteractionTestMode::And,
    )
}

#[derive(Default)]
pub(crate) struct Statics {
    ground: Vec<Ground>,
    ground_revision: Option<u64>,
    placed: HashMap<PlacedKey, rp::ColliderHandle>,
    colliders: hashbrown::HashMap<Entity, Mirror, EntityHash>,
    reach: Option<(Vec3, Vec3)>,
    pass: u64,
}

fn fixed(shape: rp::SharedShape, pose: rp::Pose, friction: f32) -> rp::ColliderBuilder {
    rp::ColliderBuilder::new(shape)
        .position(pose)
        .friction(friction)
}

impl Statics {
    pub fn len(&self) -> usize {
        self.ground.len() + self.placed.len() + self.colliders.len()
    }

    pub fn grounded(&self, x: f32, z: f32) -> bool {
        self.ground.iter().any(|ground| {
            let (min, max) = (ground.field.min, ground.field.max);
            x >= min.x && x <= max.x && z >= min.z && z <= max.z
        })
    }

    pub fn sync(
        &mut self,
        solver: &mut Solver,
        solids: &Solids,
        bodies: Option<(Vec3, Vec3)>,
        is_body: impl Fn(Entity) -> bool,
    ) {
        let revision = Some(solids.ground_revision());
        let regrounded = revision != self.ground_revision;
        if regrounded {
            self.ground_revision = revision;
            self.sync_ground(solver, solids);
        }
        let Some((min, max)) = bodies else {
            self.forget_near(solver);
            return;
        };
        let inside = self
            .reach
            .is_some_and(|(low, high)| low.cmple(min).all() && high.cmpge(max).all());
        if !inside || regrounded {
            let reach = (min - Vec3::splat(SLACK), max + Vec3::splat(SLACK));
            self.reach = Some(reach);
            self.sync_placed(solver, solids, reach);
        }
        if let Some(reach) = self.reach {
            self.sync_colliders(solver, solids, reach, is_body);
        }
    }

    fn sync_ground(&mut self, solver: &mut Solver, solids: &Solids) {
        let mut kept = std::mem::take(&mut self.ground);
        for field in solids.ground() {
            if let Some(index) = kept
                .iter()
                .position(|ground| Arc::ptr_eq(&ground.field, field))
            {
                self.ground.push(kept.swap_remove(index));
                continue;
            }
            let shape: Arc<dyn Shape> = field.shape().clone();
            let handle = solver.add_solid(
                fixed(
                    rp::SharedShape(shape),
                    rp::Pose::from_translation(field.center),
                    GROUND_FRICTION,
                )
                .build(),
            );
            self.ground.push(Ground {
                field: Arc::clone(field),
                handle,
            });
        }
        for stale in kept {
            solver.remove_collider(stale.handle);
        }
    }

    fn sync_placed(&mut self, solver: &mut Solver, solids: &Solids, reach: (Vec3, Vec3)) {
        let mut wanted: HashMap<PlacedKey, Solid> = HashMap::default();
        for placed in solids.placed_in(reach.0, reach.1) {
            let center = placed.solid.center.to_array().map(f32::to_bits);
            wanted.insert(
                (placed.terrain, placed.part.layer, placed.part.asset, center),
                placed.solid,
            );
        }
        self.placed.retain(|key, handle| {
            let keep = wanted.remove(key).is_some();
            if !keep {
                solver.remove_collider(*handle);
            }
            keep
        });
        for (key, solid) in wanted {
            if let Some((shape, pose)) = solid_shape(&solid) {
                let handle = solver.add_solid(fixed(shape, pose, SOLID_FRICTION).build());
                self.placed.insert(key, handle);
            }
        }
    }

    fn sync_colliders(
        &mut self,
        solver: &mut Solver,
        solids: &Solids,
        reach: (Vec3, Vec3),
        is_body: impl Fn(Entity) -> bool,
    ) {
        self.pass += 1;
        let pass = self.pass;
        for found in solids.colliders_in(reach.0, reach.1) {
            if is_body(found.entity) {
                continue;
            }
            if let Some(mirror) = self.colliders.get_mut(&found.entity) {
                mirror.seen = pass;
                if mirror.solid == found.solid && mirror.layers == found.layers {
                    continue;
                }
                if let Some(collider) = solver.colliders.get_mut(mirror.handle) {
                    if mirror.solid.shape == found.solid.shape
                        && mirror.solid.axes == found.solid.axes
                    {
                        collider.set_translation(found.solid.center);
                    } else if let Some((shape, pose)) = solid_shape(&found.solid) {
                        collider.set_shape(shape);
                        collider.set_position(pose);
                    }
                    collider.set_collision_groups(groups(found.layers));
                }
                mirror.solid = found.solid;
                mirror.layers = found.layers;
                continue;
            }
            let Some((shape, pose)) = solid_shape(&found.solid) else {
                continue;
            };
            let handle = solver.add_solid(
                fixed(shape, pose, SOLID_FRICTION)
                    .collision_groups(groups(found.layers))
                    .build(),
            );
            self.colliders.insert(
                found.entity,
                Mirror {
                    solid: found.solid,
                    layers: found.layers,
                    handle,
                    seen: pass,
                },
            );
        }
        self.colliders.retain(|_, mirror| {
            let keep = mirror.seen == pass;
            if !keep {
                solver.remove_collider(mirror.handle);
            }
            keep
        });
    }

    fn forget_near(&mut self, solver: &mut Solver) {
        for (_, handle) in self.placed.drain() {
            solver.remove_collider(handle);
        }
        for (_, mirror) in self.colliders.drain() {
            solver.remove_collider(mirror.handle);
        }
        self.reach = None;
    }
}
