use bevy_ecs::prelude::*;
use engine_collision::{Collider, CollisionWorld};
use engine_core::{
    FixedTime, Ground, InterpolateTransform, Parent, Physics, PhysicsStats, Transform,
};
use engine_terrain::Wind;
use glam::{Quat, Vec3};
use rapier3d::prelude as rp;

use crate::body::{BodyMotion, BodyProblem, BodyStatus, RigidBody};
use crate::coupling::{self, AIR_SPIN, Form, Motion};
use crate::shapes::{BodyShape, body_shape};
use crate::solver::Solver;
use crate::statics::{Statics, groups};

const MARGIN: f32 = 2.0;
const AHEAD: f32 = 0.5;

struct Body {
    entity: Entity,
    handle: rp::RigidBodyHandle,
    collider: rp::ColliderHandle,
    settings: RigidBody,
    shape: Collider,
    scale: Vec3,
    form: Form,
    written: (Vec3, Quat),
    status: BodyStatus,
}

pub(crate) struct Wanted {
    entity: Entity,
    settings: RigidBody,
    collider: Option<Collider>,
    pose: Option<Transform>,
    root: bool,
}

fn body_type(motion: BodyMotion) -> rp::RigidBodyType {
    match motion {
        BodyMotion::Dynamic => rp::RigidBodyType::Dynamic,
        BodyMotion::Kinematic => rp::RigidBodyType::KinematicPositionBased,
    }
}

fn solver_collider(settings: &RigidBody, collider: Collider, shape: &BodyShape) -> rp::Collider {
    rp::ColliderBuilder::new(shape.shape.clone())
        .density(settings.density)
        .friction(settings.friction)
        .restitution(settings.restitution)
        .collision_groups(groups(collider.layers()))
        .sensor(collider.is_sensor())
        .build()
}

impl Body {
    fn join(
        solver: &mut Solver,
        wanted: &Wanted,
        collider: Collider,
        pose: Transform,
        shape: &BodyShape,
    ) -> Self {
        let settings = wanted.settings;
        let body = rp::RigidBodyBuilder::new(body_type(settings.motion))
            .pose(rp::Pose::from_parts(pose.translation, pose.rotation))
            .linvel(Vec3::from_array(settings.velocity))
            .angvel(Vec3::from_array(settings.spin))
            .angular_damping(AIR_SPIN)
            .ccd_enabled(settings.fast)
            .build();
        let (handle, collider_handle) =
            solver.add_body(body, solver_collider(&settings, collider, shape));
        Self {
            entity: wanted.entity,
            handle,
            collider: collider_handle,
            settings,
            shape: collider,
            scale: pose.scale,
            form: Form::new(shape),
            written: (pose.translation, pose.rotation),
            status: BodyStatus::default(),
        }
    }

    fn follow(
        &mut self,
        solver: &mut Solver,
        wanted: &Wanted,
        collider: Collider,
        pose: Transform,
    ) -> Result<bool, BodyProblem> {
        let settings = wanted.settings;
        let reshaped = collider != self.shape
            || pose.scale != self.scale
            || settings.density != self.settings.density
            || settings.friction != self.settings.friction
            || settings.restitution != self.settings.restitution;
        if reshaped {
            let shape = body_shape(collider.shape(), pose.scale).ok_or(BodyProblem::BadScale)?;
            self.collider = solver.replace_collider(
                self.handle,
                self.collider,
                solver_collider(&settings, collider, &shape),
            );
            self.form = Form::new(&shape);
            self.shape = collider;
            self.scale = pose.scale;
        }
        let Some(body) = solver.bodies.get_mut(self.handle) else {
            return Ok(false);
        };
        if settings.motion != self.settings.motion {
            body.set_body_type(body_type(settings.motion), true);
        }
        if settings.velocity != self.settings.velocity {
            body.set_linvel(Vec3::from_array(settings.velocity), true);
        }
        if settings.spin != self.settings.spin {
            body.set_angvel(Vec3::from_array(settings.spin), true);
        }
        if settings.fast != self.settings.fast {
            body.enable_ccd(settings.fast);
        }
        self.settings = settings;
        let at = (pose.translation, pose.rotation);
        let target = rp::Pose::from_parts(pose.translation, pose.rotation);
        match settings.motion {
            BodyMotion::Kinematic => {
                body.set_next_kinematic_position(target);
                self.written = at;
                Ok(false)
            }
            BodyMotion::Dynamic if at != self.written => {
                body.set_position(target, true);
                self.written = at;
                Ok(true)
            }
            BodyMotion::Dynamic => Ok(false),
        }
    }
}

#[derive(Resource, Default)]
pub struct PhysicsWorld {
    solver: Option<Solver>,
    bodies: Vec<Body>,
    problems: Vec<(Entity, BodyProblem)>,
    statics: Statics,
    stats: PhysicsStats,
}

pub(crate) type BodyQuery = (
    Entity,
    &'static RigidBody,
    Option<&'static Collider>,
    Option<&'static Transform>,
    Has<Parent>,
);

impl PhysicsWorld {
    pub fn stats(&self) -> PhysicsStats {
        self.stats
    }

    pub fn status(&self, entity: Entity) -> Option<BodyStatus> {
        if let Ok(index) = self
            .bodies
            .binary_search_by_key(&entity, |body| body.entity)
        {
            return Some(self.bodies[index].status);
        }
        self.problems
            .iter()
            .find(|(held, _)| *held == entity)
            .map(|&(_, problem)| BodyStatus {
                problem: Some(problem),
                ..BodyStatus::default()
            })
    }

    pub(crate) fn holds(&self, entity: Entity) -> bool {
        self.bodies
            .binary_search_by_key(&entity, |body| body.entity)
            .is_ok()
    }

    pub(crate) fn solids(&self) -> impl Iterator<Item = engine_core::WaterObstacle> + '_ {
        let solver = self.solver.as_ref();
        self.bodies.iter().filter_map(move |body| {
            let rigid = solver?.bodies.get(body.handle)?;
            Some(engine_core::WaterObstacle {
                center: rigid.translation(),
                axes: glam::Mat3::from_quat(*rigid.rotation())
                    * glam::Mat3::from_diagonal(body.form.half),
                round: body.form.round,
                velocity: rigid.linvel(),
            })
        })
    }

    fn idle(&self) -> bool {
        self.solver.is_none() && self.problems.is_empty()
    }

    fn tick(&mut self, world: &mut World, mut wanted: Vec<Wanted>) {
        wanted.sort_unstable_by_key(|wanted| wanted.entity);
        let settings = world
            .get_resource::<Physics>()
            .copied()
            .unwrap_or_else(Physics::off)
            .sanitized();
        let Some(time) = world.get_resource::<FixedTime>().copied() else {
            return;
        };
        let ready =
            |wanted: &Wanted| wanted.root && wanted.collider.is_some() && wanted.pose.is_some();
        if self.solver.is_none() && !wanted.iter().any(ready) {
            self.problems = wanted
                .iter()
                .map(|wanted| {
                    let problem = if !wanted.root {
                        BodyProblem::NotRoot
                    } else if wanted.collider.is_none() {
                        BodyProblem::NoCollider
                    } else {
                        BodyProblem::NoTransform
                    };
                    (wanted.entity, problem)
                })
                .collect();
            self.stats = PhysicsStats::default();
            return;
        }
        let solver = self.solver.get_or_insert_with(Solver::default);
        let (joined, moved) = join(&mut self.bodies, &mut self.problems, solver, &wanted);
        if self.bodies.is_empty() {
            self.solver = None;
            self.statics = Statics::default();
            self.stats = PhysicsStats::default();
            return;
        }
        for entity in joined {
            if let Ok(mut entity) = world.get_entity_mut(entity)
                && !entity.contains::<InterpolateTransform>()
            {
                entity.insert(InterpolateTransform);
            }
        }
        for entity in moved {
            engine_core::reset_interpolation(world, entity);
        }
        let reach = reach(&self.bodies, solver);
        if let Some(collision) = world.get_resource::<CollisionWorld>() {
            let bodies = &self.bodies;
            self.statics.sync(solver, collision, reach, |entity| {
                bodies
                    .binary_search_by_key(&entity, |body| body.entity)
                    .is_ok()
            });
        }
        hold(&mut self.bodies, &self.statics, solver, world);
        let samples = if settings.enabled {
            push(&mut self.bodies, solver, world, &settings, time)
        } else {
            0
        };
        if settings.enabled {
            let _profile = engine_profile::span("physics", "solve");
            solver.step(
                time.delta_seconds,
                Vec3::from_array(settings.gravity),
                settings.iterations,
            );
        }
        write(&mut self.bodies, solver, world);
        let count =
            |keep: fn(&Body) -> bool| self.bodies.iter().filter(|body| keep(body)).count() as u32;
        self.stats = PhysicsStats {
            bodies: self.bodies.len() as u32,
            awake: count(|body| {
                body.settings.motion == BodyMotion::Dynamic
                    && !body.status.sleeping
                    && !body.status.held
            }),
            held: count(|body| body.status.held),
            wet: count(|body| body.status.water.is_some()),
            solids: self.statics.len() as u32,
            contacts: solver.contacts() as u32,
            water_samples: samples,
        };
    }
}

fn join(
    bodies: &mut Vec<Body>,
    problems: &mut Vec<(Entity, BodyProblem)>,
    solver: &mut Solver,
    wanted: &[Wanted],
) -> (Vec<Entity>, Vec<Entity>) {
    problems.clear();
    let mut joined = Vec::new();
    let mut moved = Vec::new();
    let mut old = std::mem::take(bodies).into_iter().peekable();
    for wanted in wanted {
        while let Some(left) = old.next_if(|body| body.entity < wanted.entity) {
            solver.remove_body(left.handle);
        }
        let held = old.next_if(|body| body.entity == wanted.entity);
        let ready = match (wanted.root, wanted.collider, wanted.pose) {
            (false, _, _) => Err(BodyProblem::NotRoot),
            (true, None, _) => Err(BodyProblem::NoCollider),
            (true, Some(_), None) => Err(BodyProblem::NoTransform),
            (true, Some(collider), Some(pose)) => Ok((collider, pose)),
        };
        let followed = ready.and_then(|(collider, pose)| match held {
            Some(mut body) => match body.follow(solver, wanted, collider, pose) {
                Ok(teleported) => {
                    if teleported {
                        moved.push(wanted.entity);
                    }
                    Ok(body)
                }
                Err(problem) => {
                    solver.remove_body(body.handle);
                    Err(problem)
                }
            },
            None => {
                let shape =
                    body_shape(collider.shape(), pose.scale).ok_or(BodyProblem::BadScale)?;
                if wanted.settings.motion == BodyMotion::Dynamic {
                    joined.push(wanted.entity);
                }
                Ok(Body::join(solver, wanted, collider, pose, &shape))
            }
        });
        match followed {
            Ok(body) => bodies.push(body),
            Err(problem) => problems.push((wanted.entity, problem)),
        }
    }
    for left in old {
        solver.remove_body(left.handle);
    }
    (joined, moved)
}

fn reach(bodies: &[Body], solver: &Solver) -> Option<(Vec3, Vec3)> {
    bodies
        .iter()
        .filter(|body| !body.status.held)
        .filter_map(|body| {
            let bounds = solver.colliders.get(body.collider)?.compute_aabb();
            let ahead = solver.bodies.get(body.handle)?.linvel() * AHEAD;
            Some((
                bounds.mins.min(bounds.mins + ahead) - Vec3::splat(MARGIN),
                bounds.maxs.max(bounds.maxs + ahead) + Vec3::splat(MARGIN),
            ))
        })
        .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

fn hold(bodies: &mut [Body], statics: &Statics, solver: &mut Solver, world: &World) {
    for body in bodies {
        let Some(rigid) = solver.bodies.get_mut(body.handle) else {
            continue;
        };
        let at = rigid.translation();
        let held = body.settings.motion == BodyMotion::Dynamic
            && !statics.grounded(at.x, at.z)
            && Ground::sample_in(world, at.x, at.z).is_some();
        if held != body.status.held {
            rigid.set_enabled(!held);
            body.status.held = held;
        }
    }
}

fn push(
    bodies: &mut [Body],
    solver: &mut Solver,
    world: &World,
    settings: &Physics,
    time: FixedTime,
) -> u32 {
    let _profile = engine_profile::span("physics", "push");
    let wind = world.get_resource::<Wind>();
    let mut samples = 0;
    for body in bodies {
        let Some(rigid) = solver.bodies.get_mut(body.handle) else {
            continue;
        };
        rigid.reset_forces(false);
        rigid.reset_torques(false);
        if body.settings.motion != BodyMotion::Dynamic || body.status.held {
            body.status.water = None;
            continue;
        }
        let motion = Motion {
            translation: rigid.translation(),
            rotation: *rigid.rotation(),
            center: rigid.center_of_mass(),
            velocity: rigid.linvel(),
            spin: rigid.angvel(),
            mass: rigid.mass(),
        };
        let pushed = coupling::push(
            world,
            settings,
            wind,
            time.elapsed_seconds as f32,
            time.delta_seconds,
            &body.form,
            body.settings.drag,
            &motion,
            &mut samples,
        );
        rigid.set_angular_damping(pushed.damping);
        let sleeping = rigid.is_sleeping();
        if !sleeping || pushed.running {
            rigid.add_force(pushed.force, sleeping);
            rigid.add_torque(pushed.torque, sleeping);
        }
        body.status.water = pushed.water;
    }
    samples
}

fn write(bodies: &mut [Body], solver: &Solver, world: &mut World) {
    for body in bodies {
        let Some(rigid) = solver.bodies.get(body.handle) else {
            continue;
        };
        body.status = BodyStatus {
            problem: None,
            mass: rigid.mass(),
            volume: body.form.volume,
            velocity: rigid.linvel(),
            spin: rigid.angvel(),
            sleeping: rigid.is_sleeping(),
            held: body.status.held,
            water: body.status.water,
        };
        if body.settings.motion != BodyMotion::Dynamic {
            continue;
        }
        let at = (rigid.translation(), *rigid.rotation());
        if at == body.written {
            continue;
        }
        if let Some(mut transform) = world.get_mut::<Transform>(body.entity) {
            transform.translation = at.0;
            transform.rotation = at.1;
        }
        body.written = at;
    }
}

pub(crate) fn step(world: &mut World, query: &mut QueryState<BodyQuery>) {
    let _profile = engine_profile::span("physics", "step");
    let wanted: Vec<Wanted> = query
        .iter(world)
        .map(|(entity, settings, collider, pose, parented)| Wanted {
            entity,
            settings: settings.sanitized(),
            collider: collider.copied(),
            pose: pose.copied(),
            root: !parented,
        })
        .collect();
    if wanted.is_empty()
        && world
            .get_resource::<PhysicsWorld>()
            .is_none_or(PhysicsWorld::idle)
    {
        return;
    }
    world.resource_scope(|world, mut physics: Mut<PhysicsWorld>| physics.tick(world, wanted));
}
