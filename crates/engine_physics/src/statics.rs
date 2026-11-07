use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::collections::HashMap;
use engine_places_simulation::physics::{Collide, ColliderSolid, CollisionLayers};
use engine_places_simulation::solids::{Bounds, FieldTile, SolidField};
use glam::Vec3;
use parry3d::shape::Shape;
use rapier3d::prelude as rp;

use crate::shapes::solid_shape;
use crate::solids::Solid;
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
    sources: Vec<Arc<SolidField>>,
    ground: Vec<Ground>,
    placed: HashMap<PlacedKey, rp::ColliderHandle>,
    colliders: hashbrown::HashMap<Entity, Mirror, EntityHash>,
    found: Vec<ColliderSolid>,
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

    pub fn sync(
        &mut self,
        solver: &mut Solver,
        collision: &dyn Collide,
        bodies: Option<(Vec3, Vec3)>,
        is_body: impl Fn(Entity) -> bool,
    ) {
        let grounds = collision.grounds();
        let regrounded = self.sources.len() != grounds.len()
            || self
                .sources
                .iter()
                .zip(grounds)
                .any(|(held, source)| !Arc::ptr_eq(held, source));
        if regrounded {
            self.sources = grounds.to_vec();
            self.sync_ground(solver);
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
            self.sync_placed(solver, reach);
        }
        if let Some(reach) = self.reach {
            self.sync_colliders(solver, collision, reach, is_body);
        }
    }

    fn sync_ground(&mut self, solver: &mut Solver) {
        let mut kept = std::mem::take(&mut self.ground);
        for field in self.sources.iter().flat_map(|source| &source.fields) {
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

    fn sync_placed(&mut self, solver: &mut Solver, reach: (Vec3, Vec3)) {
        let bounds = Bounds {
            min: reach.0,
            max: reach.1,
        };
        let mut wanted: HashMap<PlacedKey, Solid> = HashMap::default();
        for placed in self
            .sources
            .iter()
            .flat_map(|source| source.solids.query(move |node| node.overlaps(bounds)))
        {
            let Some(solid) = Solid::new(placed.shape, placed.model)
                .filter(|solid| solid.bounds().overlaps(bounds))
            else {
                continue;
            };
            let center = solid.center.to_array().map(f32::to_bits);
            wanted.insert((placed.owner, placed.layer, placed.asset, center), solid);
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
        collision: &dyn Collide,
        reach: (Vec3, Vec3),
        is_body: impl Fn(Entity) -> bool,
    ) {
        self.pass += 1;
        let pass = self.pass;
        let mut found = std::mem::take(&mut self.found);
        found.clear();
        collision.solids(reach.0, reach.1, &mut found);
        for held in &found {
            if is_body(held.entity) {
                continue;
            }
            let (solid, layers) = (Solid::of(held), held.layers);
            if let Some(mirror) = self.colliders.get_mut(&held.entity) {
                mirror.seen = pass;
                if mirror.solid == solid && mirror.layers == layers {
                    continue;
                }
                if let Some(collider) = solver.colliders.get_mut(mirror.handle) {
                    if mirror.solid.shape == solid.shape && mirror.solid.axes == solid.axes {
                        collider.set_translation(solid.center);
                    } else if let Some((shape, pose)) = solid_shape(&solid) {
                        collider.set_shape(shape);
                        collider.set_position(pose);
                    }
                    collider.set_collision_groups(groups(layers));
                }
                mirror.solid = solid;
                mirror.layers = layers;
                continue;
            }
            let Some((shape, pose)) = solid_shape(&solid) else {
                continue;
            };
            let handle = solver.add_solid(
                fixed(shape, pose, SOLID_FRICTION)
                    .collision_groups(groups(layers))
                    .build(),
            );
            self.colliders.insert(
                held.entity,
                Mirror {
                    solid,
                    layers,
                    handle,
                    seen: pass,
                },
            );
        }
        self.found = found;
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
