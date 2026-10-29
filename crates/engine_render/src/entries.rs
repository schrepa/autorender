use crate::graph::Targets;
use crate::shaders::Entry;
use crate::shaders::entry::meshes;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshPass {
    Opaque(Targets),
    View,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassKind {
    pub pass: MeshPass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slots {
    Retained,
    Temporal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Streams {
    Basic,
    Extended,
}

impl PassKind {
    pub fn opaque(targets: Targets) -> Self {
        Self {
            pass: MeshPass::Opaque(targets),
        }
    }

    pub fn view() -> Self {
        Self {
            pass: MeshPass::View,
        }
    }

    pub fn targets(self) -> Option<Targets> {
        match self.pass {
            MeshPass::Opaque(targets) => Some(targets),
            MeshPass::View => None,
        }
    }

    pub fn slots(self) -> Slots {
        if self.targets() == Some(Targets::Temporal) {
            Slots::Temporal
        } else {
            Slots::Retained
        }
    }

    pub fn vertex_entry(self, streams: Streams) -> Entry {
        if self.targets() == Some(Targets::Debug) {
            return match streams {
                Streams::Basic => meshes::vs_debug_affine,
                Streams::Extended => meshes::vs_debug_affine_extended,
            };
        }
        match (self.slots(), streams) {
            (Slots::Retained, Streams::Basic) => meshes::vs_affine,
            (Slots::Retained, Streams::Extended) => meshes::vs_affine_extended,
            (Slots::Temporal, Streams::Basic) => meshes::vs_affine_temporal,
            (Slots::Temporal, Streams::Extended) => meshes::vs_affine_temporal_extended,
        }
    }
}

pub struct MeshEntries {
    pub fragment: fn(PassKind, bool) -> Entry,
    pub prepass: fn(PassKind, bool) -> Entry,
    pub card_vertex: fn(Targets) -> Entry,
    pub card_fragment: fn(Targets) -> Entry,
    pub card_bake: fn(bool, bool) -> (Entry, Entry),
}
