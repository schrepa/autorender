//! The solver's state: Rapier's sets, pipeline and parameters, which only
//! this crate's modules touch.

use glam::Vec3;
use rapier3d::prelude as rp;

pub(crate) struct Solver {
    pub bodies: rp::RigidBodySet,
    pub colliders: rp::ColliderSet,
    pub islands: rp::IslandManager,
    joints: rp::ImpulseJointSet,
    multibodies: rp::MultibodyJointSet,
    pipeline: rp::PhysicsPipeline,
    broad: rp::BroadPhaseBvh,
    narrow: rp::NarrowPhase,
    ccd: rp::CCDSolver,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            bodies: rp::RigidBodySet::new(),
            colliders: rp::ColliderSet::new(),
            islands: rp::IslandManager::new(),
            joints: rp::ImpulseJointSet::new(),
            multibodies: rp::MultibodyJointSet::new(),
            pipeline: rp::PhysicsPipeline::new(),
            broad: rp::BroadPhaseBvh::new(),
            narrow: rp::NarrowPhase::new(),
            ccd: rp::CCDSolver::new(),
        }
    }
}

impl Solver {
    /// Advances every body `seconds` under `gravity`, its constraints
    /// solved over `iterations`.
    pub fn step(&mut self, seconds: f32, gravity: Vec3, iterations: u32) {
        let parameters = rp::IntegrationParameters {
            dt: seconds,
            num_solver_iterations: iterations.max(1) as usize,
            ..rp::IntegrationParameters::default()
        };
        self.pipeline.step(
            gravity,
            &parameters,
            &mut self.islands,
            &mut self.broad,
            &mut self.narrow,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joints,
            &mut self.multibodies,
            &mut self.ccd,
            &(),
            &(),
        );
    }

    /// Adds a fixed collider standing in the world.
    pub fn add_solid(&mut self, collider: rp::Collider) -> rp::ColliderHandle {
        self.colliders.insert(collider)
    }

    /// Removes a collider, of a body or of the world.
    pub fn remove_collider(&mut self, handle: rp::ColliderHandle) {
        self.colliders
            .remove(handle, &mut self.islands, &mut self.bodies, true);
    }

    /// Adds `body` with `collider`: their handles.
    pub fn add_body(
        &mut self,
        body: rp::RigidBody,
        collider: rp::Collider,
    ) -> (rp::RigidBodyHandle, rp::ColliderHandle) {
        let handle = self.bodies.insert(body);
        let collider = self
            .colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        (handle, collider)
    }

    /// Replaces body `handle`'s collider `old` with `collider`: the new
    /// collider's handle. Its mass follows.
    pub fn replace_collider(
        &mut self,
        handle: rp::RigidBodyHandle,
        old: rp::ColliderHandle,
        collider: rp::Collider,
    ) -> rp::ColliderHandle {
        self.remove_collider(old);
        let new = self
            .colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        if let Some(body) = self.bodies.get_mut(handle) {
            body.recompute_mass_properties_from_colliders(&self.colliders);
            body.wake_up(true);
        }
        new
    }

    /// Removes a body and its collider.
    pub fn remove_body(&mut self, handle: rp::RigidBodyHandle) {
        self.bodies.remove(
            handle,
            &mut self.islands,
            &mut self.colliders,
            &mut self.joints,
            &mut self.multibodies,
            true,
        );
    }

    /// Pairs of colliders touching.
    pub fn contacts(&self) -> usize {
        self.narrow
            .contact_pairs()
            .filter(|pair| pair.has_any_active_contact())
            .count()
    }
}
