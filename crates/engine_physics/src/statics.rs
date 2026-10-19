//! The world's solids where bodies may meet them, mirrored into the solver as
//! fixed colliders following the collision world: terrains' heightfields
//! wherever collision holds them (shared, not copied), and the placed
//! objects, bridge decks and colliders (sensors aside) within reach of the
//! bodies.

use std::sync::Arc;

use bevy_ecs::entity::{Entity, EntityHashMap};
use engine_collision::{CollisionLayers, CollisionWorld, Solid};
use engine_core::TileKey;
use engine_core::collections::HashMap;
use glam::Vec3;
use parry3d::shape::Shape;
use rapier3d::prelude as rp;

use crate::shapes::solid_shape;
use crate::solver::Solver;

/// Meters beyond the bodies' reach the world's solids are mirrored within,
/// so moving bodies find them already there.
const SLACK: f32 = 8.0;
/// How the terrain's surface grips, and the other solids'.
const GROUND_FRICTION: f32 = 0.8;
const SOLID_FRICTION: f32 = 0.7;

/// A terrain tile's heightfield in the solver: its terrain and key, the
/// field collision built for it, its world footprint and its collider.
struct Ground {
    terrain: Entity,
    key: TileKey,
    field: Arc<parry3d::shape::HeightField>,
    min: Vec3,
    max: Vec3,
    handle: rp::ColliderHandle,
}

/// A collider in the solver: its solid and layers as mirrored, its
/// collider, and the pass that last found it.
struct Mirror {
    solid: Solid,
    layers: CollisionLayers,
    handle: rp::ColliderHandle,
    seen: u64,
}

/// A placed object: its terrain, placement layer and asset, and its
/// center's bits.
type PlacedKey = (Entity, u16, u16, [u32; 3]);

/// The solver's interaction groups of `layers`: both must admit each
/// other's memberships, as collision's queries do.
pub(crate) fn groups(layers: CollisionLayers) -> rp::InteractionGroups {
    rp::InteractionGroups::new(
        rp::Group::from_bits_retain(layers.memberships),
        rp::Group::from_bits_retain(layers.filter),
        rp::InteractionTestMode::And,
    )
}

/// The world's solids mirrored into the solver.
#[derive(Default)]
pub(crate) struct Statics {
    ground: Vec<Ground>,
    ground_revision: Option<u64>,
    placed: HashMap<PlacedKey, rp::ColliderHandle>,
    colliders: EntityHashMap<Mirror>,
    reach: Option<(Vec3, Vec3)>,
    pass: u64,
}

/// A fixed collider of `shape` at `pose`.
fn fixed(shape: rp::SharedShape, pose: rp::Pose, friction: f32) -> rp::ColliderBuilder {
    rp::ColliderBuilder::new(shape)
        .position(pose)
        .friction(friction)
}

impl Statics {
    /// Solids mirrored.
    pub fn len(&self) -> usize {
        self.ground.len() + self.placed.len() + self.colliders.len()
    }

    /// Whether a mirrored heightfield lies under world `(x, z)`.
    pub fn grounded(&self, x: f32, z: f32) -> bool {
        self.ground.iter().any(|ground| {
            x >= ground.min.x && x <= ground.max.x && z >= ground.min.z && z <= ground.max.z
        })
    }

    /// Follows `collision`: its heightfields, and its placed objects and
    /// colliders within reach of the box from `bodies`' first to second
    /// corner, leaving out the colliders `is_body` names.
    pub fn sync(
        &mut self,
        solver: &mut Solver,
        collision: &CollisionWorld,
        bodies: Option<(Vec3, Vec3)>,
        is_body: impl Fn(Entity) -> bool,
    ) {
        let revision = collision.ground_revision();
        let regrounded = revision != self.ground_revision;
        if regrounded {
            self.ground_revision = revision;
            self.sync_ground(solver, collision);
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
            self.sync_placed(solver, collision, reach);
        }
        if let Some(reach) = self.reach {
            self.sync_colliders(solver, collision, reach, is_body);
        }
    }

    /// The heightfields collision holds now, sharing their fields.
    fn sync_ground(&mut self, solver: &mut Solver, collision: &CollisionWorld) {
        let mut kept = std::mem::take(&mut self.ground);
        for solid in collision.ground_solids() {
            if let Some(index) = kept.iter().position(|ground| {
                ground.terrain == solid.terrain
                    && ground.key == solid.key
                    && Arc::ptr_eq(&ground.field, solid.field)
            }) {
                self.ground.push(kept.swap_remove(index));
                continue;
            }
            let field: Arc<dyn Shape> = solid.field.clone();
            let handle = solver.add_solid(
                fixed(
                    rp::SharedShape(field),
                    rp::Pose::from_translation(solid.center),
                    GROUND_FRICTION,
                )
                .build(),
            );
            self.ground.push(Ground {
                terrain: solid.terrain,
                key: solid.key,
                field: solid.field.clone(),
                min: solid.min,
                max: solid.max,
                handle,
            });
        }
        for stale in kept {
            solver.remove_collider(stale.handle);
        }
    }

    /// The placed objects within `reach`.
    fn sync_placed(
        &mut self,
        solver: &mut Solver,
        collision: &CollisionWorld,
        reach: (Vec3, Vec3),
    ) {
        let mut wanted: HashMap<PlacedKey, Solid> = HashMap::default();
        for placed in collision.placed_solids_in(reach.0, reach.1) {
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

    /// The colliders within `reach` but bodies' own, moved as theirs move.
    fn sync_colliders(
        &mut self,
        solver: &mut Solver,
        collision: &CollisionWorld,
        reach: (Vec3, Vec3),
        is_body: impl Fn(Entity) -> bool,
    ) {
        self.pass += 1;
        let pass = self.pass;
        for found in collision.collider_solids_in(reach.0, reach.1) {
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

    /// Lets go of every solid but the heightfields: no body is near any.
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
