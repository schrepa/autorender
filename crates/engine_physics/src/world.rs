use bevy_ecs::prelude::*;
use engine_core::{Filled, FixedTime, InterpolateTransform, Parent, Transform};
use engine_places::ground::Ground;
use engine_places::physics::{BodyMotion, BodyPush, Collider, RigidBody};
use engine_places::water::{WaterBody, WaterObstacle};
use engine_places::wind::Wind;
use glam::{Quat, Vec2, Vec3};
use rapier3d::prelude as rp;

use crate::body::{self, BodyProblem, BodyStatus};
use crate::coupling::{self, AIR_SPIN, Form, Motion};
use crate::obstacles::sorted;
use crate::settings::{Physics, PhysicsStats};
use crate::shapes::body_shape;
use crate::solids::Solids;
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

struct Surroundings<'a> {
    settings: &'a Physics,
    time: &'a FixedTime,
    wind: Option<&'a Wind>,
    ground: &'a Ground,
    solids: &'a Solids,
    waters: &'a [&'a WaterBody],
}

fn body_type(motion: BodyMotion) -> rp::RigidBodyType {
    match motion {
        BodyMotion::Dynamic => rp::RigidBodyType::Dynamic,
        BodyMotion::Kinematic => rp::RigidBodyType::KinematicPositionBased,
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
    Option<&'static mut Transform>,
    Has<Parent>,
);

type Markers<'w, 's> = Query<'w, 's, &'static mut InterpolateTransform, With<RigidBody>>;
type Pushes<'w, 's> = Query<'w, 's, &'static mut BodyPush>;

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
        mut wanted: Vec<Wanted>,
        around: &Surroundings<'_>,
        poses: &mut Query<'_, '_, BodyQuery>,
        markers: &mut Markers<'_, '_>,
        pushes: &mut Pushes<'_, '_>,
    ) {
        wanted.sort_unstable_by_key(|wanted| wanted.entity);
        let settings = around.settings;
        let time = around.time;
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
        let moved = join(&mut self.bodies, &mut self.problems, solver, &wanted);
        if self.bodies.is_empty() {
            self.solver = None;
            self.statics = Statics::default();
            self.stats = PhysicsStats::default();
            return;
        }
        for entity in moved {
            if let Ok(mut marker) = markers.get_mut(entity) {
                marker.resets = marker.resets.wrapping_add(1);
            }
        }
        let reach = reach(&self.bodies, solver);
        if around.solids.collides() {
            let bodies = &self.bodies;
            self.statics.sync(solver, around.solids, reach, |entity| {
                bodies
                    .binary_search_by_key(&entity, |body| body.entity)
                    .is_ok()
            });
        }
        hold(&mut self.bodies, &self.statics, solver, around.ground);
        let samples = push(&mut self.bodies, solver, around, pushes);
        {
            let _profile = engine_core::profile::span("physics", "solve");
            solver.step(
                time.delta_seconds,
                Vec3::from_array(settings.gravity),
                settings.iterations,
            );
        }
        write(&mut self.bodies, solver, poses);
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
) -> Vec<Entity> {
    problems.clear();
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
    moved
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

fn hold(bodies: &mut [Body], statics: &Statics, solver: &mut Solver, ground: &Ground) {
    for body in bodies {
        let Some(rigid) = solver.bodies.get_mut(body.handle) else {
            continue;
        };
        let at = rigid.translation();
        let held = body.settings.motion == BodyMotion::Dynamic
            && !statics.grounded(at.x, at.z)
            && ground.sample(Vec2::new(at.x, at.z)).is_some();
        if held != body.status.held {
            rigid.set_enabled(!held);
            body.status.held = held;
        }
    }
}

fn asked_push(rigid: &mut rp::RigidBody, mut asked: Mut<'_, BodyPush>) {
    if *asked == BodyPush::default() {
        return;
    }
    let kept = body::sanitized_push(*asked);
    rigid.add_force(Vec3::from_array(kept.force), true);
    rigid.add_torque(Vec3::from_array(kept.torque), true);
    rigid.apply_impulse(Vec3::from_array(kept.impulse), true);
    rigid.apply_torque_impulse(Vec3::from_array(kept.angular_impulse), true);
    if asked.impulse != [0.0; 3] || asked.angular_impulse != [0.0; 3] {
        asked.impulse = [0.0; 3];
        asked.angular_impulse = [0.0; 3];
    }
}

fn push(
    bodies: &mut [Body],
    solver: &mut Solver,
    around: &Surroundings<'_>,
    pushes: &mut Pushes<'_, '_>,
) -> u32 {
    let _profile = engine_core::profile::span("physics", "push");
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
        if let Ok(asked) = pushes.get_mut(body.entity) {
            asked_push(rigid, asked);
        }
    }
    samples
}

fn write(bodies: &mut [Body], solver: &Solver, poses: &mut Query<'_, '_, BodyQuery>) {
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
        if let Ok((_, _, _, Some(mut transform), _)) = poses.get_mut(body.entity) {
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
    settings: Res<Physics>,
    time: Res<FixedTime>,
    wind: Res<Wind>,
    wind_on: Filled<Wind>,
    ground: Res<Ground>,
    solids: Res<Solids>,
    waters: Query<&WaterBody>,
    mut poses: Query<BodyQuery>,
    mut markers: Markers,
    mut pushes: Pushes,
    mut physics: ResMut<PhysicsWorld>,
    mut stats: ResMut<PhysicsStats>,
) {
    let _profile = engine_core::profile::span("physics", "step");
    let wanted: Vec<Wanted> = poses
        .iter()
        .map(|(entity, rigid, collider, pose, parented)| Wanted {
            entity,
            settings: body::sanitized(*rigid),
            collider: collider.copied(),
            pose: pose.copied(),
            root: !parented,
        })
        .collect();
    if wanted.is_empty() && physics.idle() {
        return;
    }
    let waters = sorted(waters.iter());
    let around = Surroundings {
        settings: &settings,
        time: &time,
        wind: wind_on.on().then_some(&*wind),
        ground: &ground,
        solids: &solids,
        waters: &waters,
    };
    physics.tick(wanted, &around, &mut poses, &mut markers, &mut pushes);
    stats.set_if_neq(physics.stats());
}
