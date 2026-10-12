//! Skins: the joints whose poses deform a mesh, and the joint
//! palettes extraction prepares for renderers.

use std::sync::Arc;

use bevy_ecs::{
    change_detection::Ref,
    prelude::{Component, Entity, World},
    query::QueryState,
};
use glam::Mat4;

use crate::ticks::ScanClock;
use crate::{MeshHandle, TransformIndex};

/// Deforms this entity's mesh by the displayed world poses of `joints`, as
/// glTF skins do.
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

/// Skinned entities' query, per-joint transform store hints, and what the
/// last extraction listed.
pub(crate) struct SkinExtraction {
    pub(crate) query: QueryState<(Entity, Ref<'static, Skin>)>,
    pub(crate) hints: Vec<TransformIndex>,
    // Joints the skins listed, which decals' joints follow.
    pub(crate) joints: usize,
    // Slots given a joint palette by the last extraction, and scratch for
    // the next one's (slot, offset) pairs.
    pub(crate) skinned: Vec<usize>,
    pub(crate) scratch: Vec<(usize, u32)>,
    // What the last GPU-presented extraction read (each skinned entity's
    // slot and mesh, in query order; the asset, topology and validity
    // revisions) and when it scanned the skins: while all match and no
    // skin changed, its joints stand.
    pub(crate) inputs: Vec<(Entity, usize, Option<MeshHandle>)>,
    pub(crate) key: Option<((u64, u64), u64, u64)>,
    pub(crate) scans: ScanClock,
}

impl SkinExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, Ref<Skin>)>(),
            hints: Vec::new(),
            joints: 0,
            skinned: Vec::new(),
            scratch: Vec::new(),
            inputs: Vec::new(),
            key: None,
            scans: ScanClock::default(),
        }
    }
}
