//! Skins (decision 036): the joints whose poses deform a mesh, and the joint
//! palettes extraction prepares for renderers.

use std::sync::Arc;

use bevy_ecs::{
    prelude::{Component, Entity, World},
    query::QueryState,
};
use glam::Mat4;

use crate::{PoseRows, TransformIndex};

/// Deforms this entity's mesh by the displayed world poses of `joints`, as
/// glTF skins do (decision 036).
///
/// Each vertex of a mesh with joints and weights ([`crate::Mesh::with_skin`])
/// moves by the weighted sum of its joints' world matrices times their
/// inverse bind matrices; the entity's own pose does not move skinned
/// vertices. Vertex joint indices address `joints`, which need at least
/// [`crate::Mesh::joint_count`] entries, or the mesh draws rigidly with the
/// entity's pose. A joint without a valid world pose counts as the identity.
/// Meshes with joints are never frustum culled.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Skin {
    pub joints: Vec<Entity>,
    /// One matrix per joint, from the mesh's bind space into the joint's
    /// space; missing entries are the identity.
    pub inverse_binds: Arc<[Mat4]>,
}

impl Skin {
    pub fn new(joints: Vec<Entity>, inverse_binds: impl Into<Arc<[Mat4]>>) -> Self {
        Self {
            joints,
            inverse_binds: inverse_binds.into(),
        }
    }

    /// Joint `index`'s inverse bind matrix.
    pub fn inverse_bind(&self, index: usize) -> Mat4 {
        self.inverse_binds
            .get(index)
            .copied()
            .unwrap_or(Mat4::IDENTITY)
    }
}

/// One joint of a skinned instance's palette (decision 036), in
/// [`crate::RenderSnapshot::joints`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderJoint {
    /// The joint's hierarchy node in the index space of
    /// [`crate::GpuPresentation`] records, or `u32::MAX` without a valid
    /// pose. Set on GPU-presented frames, which compose the joint's
    /// displayed pose as they do slots'.
    pub node: u32,
    /// The joint's inverse bind matrix, in upload layout.
    pub inverse_bind: PoseRows,
    /// CPU-prepared frames: the joint's displayed world matrix times its
    /// inverse bind matrix, in upload layout. GPU-presented frames leave it
    /// unset.
    pub matrix: PoseRows,
}

/// Skinned entities' query and per-joint transform store hints.
pub(crate) struct SkinExtraction {
    pub(crate) query: QueryState<(Entity, &'static Skin)>,
    pub(crate) hints: Vec<TransformIndex>,
}

impl SkinExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, &Skin)>(),
            hints: Vec::new(),
        }
    }
}
