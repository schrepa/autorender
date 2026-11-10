use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    Applies, Error, ErrorCode, Filled, FixedTime, GlobalTransforms, Moves, Parent, Refusals,
    Transform,
};
use engine_places_simulation::physics::{
    BodyAsk, BodyChange, BodyMotion, BodyPush, Collide, Collider, Collision, RigidBody,
};
use engine_places_simulation::water::{WaterBody, WaterObstacle};
use engine_places_simulation::wind::Wind;
use engine_solids::{SolidField, SolidSource};
use glam::{Quat, Vec3};
use rapier3d::prelude as rp;

use crate::PhysicsBrick;
use crate::body::{self, BodyProblem, BodyStatus};
use crate::coupling::{self, AIR_SPIN, Form, Motion};
use crate::obstacles::sorted;
use crate::settings::{Physics, PhysicsStats};
use crate::shapes::body_shape;
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
    simulated: bool,
    written: (Vec3, Quat),
    status: BodyStatus,
}

pub(crate) struct Wanted {
    entity: Entity,
    settings: RigidBody,
    collider: Option<Collider>,
    pose: Option<Transform>,
    root: bool,
    moved: bool,
    refused: Option<BodyProblem>,
}

impl Wanted {
    fn ready(&self) -> Result<(Collider, Transform), BodyProblem> {
        match (self.root, self.collider, self.pose, &self.refused) {
            (false, _, _, _) => Err(BodyProblem::NotRoot),
            (true, None, _, _) => Err(BodyProblem::NoCollider),
            (true, Some(_), None, _) => Err(BodyProblem::NoTransform),
            (true, Some(_), Some(_), Some(problem)) => Err(problem.clone()),
            (true, Some(collider), Some(pose), None) => Ok((collider, pose)),
        }
    }

    fn simulated(&self) -> bool {
        simulates(self.settings.motion, self.moved)
    }
}

struct Surroundings<'a> {
    settings: &'a Physics,
    time: &'a FixedTime,
    wind: Option<&'a Wind>,
    collision: &'a dyn Collide,
    grounds: &'a [Arc<SolidField>],
    waters: &'a [&'a WaterBody],
}

fn simulates(motion: BodyMotion, moved: bool) -> bool {
    match motion {
        BodyMotion::Dynamic => moved,
        BodyMotion::Kinematic => false,
    }
}

fn body_type(simulated: bool) -> rp::RigidBodyType {
    if simulated {
        rp::RigidBodyType::Dynamic
    } else {
        rp::RigidBodyType::KinematicPositionBased
    }
}

fn solver_collider(
    settings: &RigidBody,
    collider: Collider,
    shape: &rp::SharedShape,
) -> rp::Collider {
    rp::ColliderBuilder::new(shape.clone())
        .density(settings.density)
        .friction(settings.friction)
        .restitution(settings.restitution)
        .collision_groups(groups(collider.layers()))
        .sensor(collider.is_sensor())
        .build()
}

impl Body {
    fn form(&self) -> Form {
        Form::of(self.shape.shape(), self.scale)
    }

    fn join(
        solver: &mut Solver,
        wanted: &Wanted,
        collider: Collider,
        pose: Transform,
        shape: &rp::SharedShape,
    ) -> Self {
        let settings = wanted.settings;
        let simulated = wanted.simulated();
        let body = rp::RigidBodyBuilder::new(body_type(simulated))
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
            simulated,
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
    ) -> Result<(), BodyProblem> {
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
            self.shape = collider;
            self.scale = pose.scale;
        }
        let Some(body) = solver.bodies.get_mut(self.handle) else {
            return Ok(());
        };
        let simulated = wanted.simulated();
        if simulated != self.simulated {
            body.set_body_type(body_type(simulated), true);
            self.simulated = simulated;
        }
        if settings.fast != self.settings.fast {
            body.enable_ccd(settings.fast);
        }
        self.settings = settings;
        let at = (pose.translation, pose.rotation);
        let target = rp::Pose::from_parts(pose.translation, pose.rotation);
        if !simulated {
            body.set_next_kinematic_position(target);
            self.written = at;
        } else if at != self.written {
            body.set_position(target, true);
            self.written = at;
        }
        Ok(())
    }
}

#[derive(Resource, Default)]
pub(crate) struct PhysicsWorld {
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
    Has<Parent>,
);

type Pushes<'w, 's> = Query<'w, 's, &'static BodyPush>;

impl PhysicsWorld {
    pub fn stats(&self) -> PhysicsStats {
        self.stats
    }

    pub fn status(&self, entity: Entity) -> Option<BodyStatus> {
        if let Ok(index) = self
            .bodies
            .binary_search_by_key(&entity, |body| body.entity)
        {
            return Some(self.bodies[index].status.clone());
        }
        self.problems
            .iter()
            .find(|(found, _)| *found == entity)
            .map(|(_, problem)| BodyStatus {
                problem: Some(problem.clone()),
                ..BodyStatus::default()
            })
    }

    fn problem(&self, entity: Entity) -> Option<&BodyProblem> {
        self.problems
            .iter()
            .find(|(found, _)| *found == entity)
            .map(|(_, problem)| problem)
    }

    pub fn holds(&self, entity: Entity) -> bool {
        self.bodies
            .binary_search_by_key(&entity, |body| body.entity)
            .is_ok()
    }

    pub fn solids(&self) -> impl Iterator<Item = WaterObstacle> + '_ {
        let solver = self.solver.as_ref();
        self.bodies.iter().filter_map(move |body| {
            let rigid = solver?.bodies.get(body.handle)?;
            let form = body.form();
            Some(WaterObstacle {
                center: rigid.translation(),
                axes: glam::Mat3::from_quat(*rigid.rotation())
                    * glam::Mat3::from_diagonal(form.half),
                round: form.round,
                velocity: rigid.linvel(),
            })
        })
    }

    fn idle(&self) -> bool {
        self.solver.is_none() && self.problems.is_empty()
    }

    fn tick(
        &mut self,
        (mut wanted, asks): (Vec<Wanted>, Vec<BodyAsk>),
        around: &Surroundings<'_>,
        moves: &mut Moves<'_, '_, PhysicsBrick>,
        pushes: &Pushes<'_, '_>,
    ) {
        wanted.sort_unstable_by_key(|wanted| wanted.entity);
        let settings = around.settings;
        let time = around.time;
        if self.solver.is_none() && wanted.iter().all(|wanted| wanted.ready().is_err()) {
            self.problems = wanted
                .iter()
                .filter_map(|wanted| Some((wanted.entity, wanted.ready().err()?)))
                .collect();
            self.stats = PhysicsStats {
                problems: self.problems.len() as u32,
                dropped_asks: asks.len() as u32,
                ..PhysicsStats::default()
            };
            return;
        }
        let solver = self.solver.get_or_insert_with(Solver::default);
        join(&mut self.bodies, &mut self.problems, solver, &wanted);
        if self.bodies.is_empty() {
            self.solver = None;
            self.statics = Statics::default();
            self.stats = PhysicsStats {
                problems: self.problems.len() as u32,
                dropped_asks: asks.len() as u32,
                ..PhysicsStats::default()
            };
            return;
        }
        let dropped = apply(&self.bodies, solver, asks);
        let reach = reach(&self.bodies, solver);
        let bodies = &self.bodies;
        self.statics.sync(solver, (around.collision, around.grounds), reach, |entity| {
            bodies
                .binary_search_by_key(&entity, |body| body.entity)
                .is_ok()
        });
        let samples = push(&mut self.bodies, solver, around, pushes);
        {
            let _profile = engine_core::profile::span("physics", "solve");
            solver.step(
                time.delta_seconds,
                Vec3::from_array(settings.gravity),
                settings.iterations,
            );
        }
        write(&mut self.bodies, solver, moves);
        let count =
            |keep: fn(&Body) -> bool| self.bodies.iter().filter(|body| keep(body)).count() as u32;
        self.stats = PhysicsStats {
            bodies: self.bodies.len() as u32,
            awake: count(|body| body.simulated && !body.status.sleeping),
            wet: count(|body| body.status.water.is_some()),
            solids: self.statics.len() as u32,
            contacts: solver.contacts() as u32,
            water_samples: samples,
            problems: self.problems.len() as u32,
            dropped_asks: dropped,
        };
    }
}

fn join(
    bodies: &mut Vec<Body>,
    problems: &mut Vec<(Entity, BodyProblem)>,
    solver: &mut Solver,
    wanted: &[Wanted],
) {
    problems.clear();
    let mut old = std::mem::take(bodies).into_iter().peekable();
    for wanted in wanted {
        while let Some(left) = old.next_if(|body| body.entity < wanted.entity) {
            solver.remove_body(left.handle);
        }
        let held = old.next_if(|body| body.entity == wanted.entity);
        let ready = wanted.ready();
        if let (Err(_), Some(body)) = (&ready, &held) {
            solver.remove_body(body.handle);
        }
        let followed = ready.and_then(|(collider, pose)| match held {
            Some(mut body) => match body.follow(solver, wanted, collider, pose) {
                Ok(()) => Ok(body),
                Err(problem) => {
                    solver.remove_body(body.handle);
                    Err(problem)
                }
            },
            None => {
                let shape =
                    body_shape(collider.shape(), pose.scale).ok_or(BodyProblem::BadScale)?;
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
}

fn apply(bodies: &[Body], solver: &mut Solver, asks: Vec<BodyAsk>) -> u32 {
    let mut dropped = 0;
    for BodyAsk { body, change } in asks {
        let rigid = bodies
            .binary_search_by_key(&body, |held| held.entity)
            .ok()
            .map(|index| &bodies[index])
            .filter(|held| held.simulated && body::admits(change))
            .and_then(|held| solver.bodies.get_mut(held.handle));
        let Some(rigid) = rigid else {
            dropped += 1;
            continue;
        };
        let value = Vec3::from_array(change.value());
        match change {
            BodyChange::Velocity(_) => rigid.set_linvel(value, true),
            BodyChange::Spin(_) => rigid.set_angvel(value, true),
            BodyChange::Impulse(_) => rigid.apply_impulse(value, true),
            BodyChange::AngularImpulse(_) => rigid.apply_torque_impulse(value, true),
        }
    }
    dropped
}

fn reach(bodies: &[Body], solver: &Solver) -> Option<(Vec3, Vec3)> {
    bodies
        .iter()
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

fn push(
    bodies: &mut [Body],
    solver: &mut Solver,
    around: &Surroundings<'_>,
    pushes: &Pushes<'_, '_>,
) -> u32 {
    let _profile = engine_core::profile::span("physics", "push");
    let mut samples = 0;
    for body in bodies {
        let Some(rigid) = solver.bodies.get_mut(body.handle) else {
            continue;
        };
        rigid.reset_forces(false);
        rigid.reset_torques(false);
        if !body.simulated {
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
            around.waters,
            around.settings,
            around.wind,
            around.time.elapsed_seconds as f32,
            around.time.delta_seconds,
            &body.form(),
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
        if let Ok(asked) = pushes.get(body.entity)
            && *asked != BodyPush::default()
        {
            rigid.add_force(Vec3::from_array(asked.force), true);
            rigid.add_torque(Vec3::from_array(asked.torque), true);
        }
    }
    samples
}

fn write(bodies: &mut [Body], solver: &Solver, moves: &mut Moves<'_, '_, PhysicsBrick>) {
    for body in bodies {
        let Some(rigid) = solver.bodies.get(body.handle) else {
            continue;
        };
        body.status = BodyStatus {
            problem: None,
            mass: rigid.mass(),
            volume: body.form().volume,
            velocity: rigid.linvel(),
            spin: rigid.angvel(),
            sleeping: rigid.is_sleeping(),
            water: body.status.water,
        };
        if !body.simulated {
            continue;
        }
        let at = (rigid.translation(), *rigid.rotation());
        if at == body.written {
            continue;
        }
        if let Some(mut transform) = moves.get_mut(body.entity) {
            transform.translation = at.0;
            transform.rotation = at.1;
        }
        body.written = at;
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "each is one of the tick's inputs"
)]
pub(crate) fn step(
    (settings, time, globals): (Res<Physics>, Res<FixedTime>, Res<GlobalTransforms>),
    (wind, wind_on): (Res<Wind>, Filled<Wind>),
    (collision, sources): (Res<Collision>, Query<&SolidSource>),
    waters: Query<&WaterBody>,
    bodies: Query<BodyQuery>,
    mut moves: Moves<PhysicsBrick>,
    mut asked: Applies<BodyAsk>,
    pushes: Pushes,
    mut physics: ResMut<PhysicsWorld>,
    (mut stats, mut refusals): (ResMut<PhysicsStats>, ResMut<Refusals>),
) {
    let _profile = engine_core::profile::span("physics", "step");
    let asks = asked.take();
    let wanted: Vec<Wanted> = bodies
        .iter()
        .filter(|&(entity, ..)| globals.active(entity))
        .map(|(entity, rigid, collider, parented)| Wanted {
            entity,
            settings: *rigid,
            collider: collider.copied(),
            pose: moves.local(entity).copied(),
            root: !parented,
            moved: moves.contains(entity),
            refused: body::refused(rigid, pushes.get(entity).ok()),
        })
        .collect();
    let mut held: Vec<(Entity, bool)> = wanted
        .iter()
        .map(|wanted| {
            let pinned = wanted.settings.motion == BodyMotion::Dynamic && !wanted.moved;
            (wanted.entity, pinned)
        })
        .collect();
    held.sort_unstable_by_key(|&(entity, _)| entity);
    if wanted.is_empty() && asks.is_empty() && physics.idle() {
        judge(&physics, &held, &mut refusals);
        return;
    }
    let waters = sorted(waters.iter());
    let grounds: Vec<Arc<SolidField>> = sources
        .iter()
        .map(|source| Arc::clone(&source.field))
        .collect();
    let around = Surroundings {
        settings: &settings,
        time: &time,
        wind: wind_on.on().then_some(&*wind),
        collision: &*collision.0,
        grounds: &grounds,
        waters: &waters,
    };
    physics.tick((wanted, asks), &around, &mut moves, &pushes);
    stats.set_if_neq(physics.stats());
    judge(&physics, &held, &mut refusals);
}

pub const BODY_PART: &str = "body";

fn judge(physics: &PhysicsWorld, held: &[(Entity, bool)], refusals: &mut Refusals) {
    refusals.keep_part(BODY_PART, |entity| {
        held.binary_search_by_key(&entity, |&(found, _)| found).is_ok()
    });
    for &(entity, pinned) in held {
        let verdict = match physics.problem(entity) {
            Some(problem) => Err(problem.error()),
            None if pinned => Err(Error::new(ErrorCode::UsageConflict, "$.rigid_body.motion")
                .got(BodyMotion::Dynamic)
                .nearest(BodyMotion::Kinematic)),
            None => Ok(()),
        };
        refusals.judge(entity, BODY_PART, verdict);
    }
}
