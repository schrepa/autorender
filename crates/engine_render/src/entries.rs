use crate::graph::Targets;
use crate::shaders::Entry;
use crate::shaders::entry::{meshes, terrain_shadow};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MeshPass {
    Opaque(Targets),
    Probe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PassKind {
    pub pass: MeshPass,
    pub lit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slots {
    Retained,
    Temporal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Streams {
    Basic,
    Extended,
}

impl PassKind {
    pub(crate) fn opaque(targets: Targets, lit: bool) -> Self {
        Self {
            pass: MeshPass::Opaque(targets),
            lit,
        }
    }

    pub(crate) fn probe(lit: bool) -> Self {
        Self {
            pass: MeshPass::Probe,
            lit,
        }
    }

    pub(crate) fn targets(self) -> Option<Targets> {
        match self.pass {
            MeshPass::Opaque(targets) => Some(targets),
            MeshPass::Probe => None,
        }
    }

    pub(crate) fn slots(self) -> Slots {
        if self.targets() == Some(Targets::Temporal) {
            Slots::Temporal
        } else {
            Slots::Retained
        }
    }

    pub(crate) fn vertex_entry(self, streams: Streams) -> Entry {
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

    pub(crate) fn fragment_entry(self, extended: bool) -> Entry {
        match (self.pass, self.lit, extended) {
            (MeshPass::Opaque(Targets::Canvas), false, false) => meshes::fs_main,
            (MeshPass::Opaque(Targets::Canvas), true, false) => meshes::fs_main_lit,
            (MeshPass::Opaque(Targets::Canvas), false, true) => meshes::fs_main_extended,
            (MeshPass::Opaque(Targets::Canvas), true, true) => meshes::fs_main_lit_extended,
            (MeshPass::Opaque(Targets::Post), false, false) => meshes::fs_post,
            (MeshPass::Opaque(Targets::Post), true, false) => meshes::fs_post_lit,
            (MeshPass::Opaque(Targets::Post), false, true) => meshes::fs_post_extended,
            (MeshPass::Opaque(Targets::Post), true, true) => meshes::fs_post_lit_extended,
            (MeshPass::Opaque(Targets::Temporal), false, false) => meshes::fs_temporal,
            (MeshPass::Opaque(Targets::Temporal), true, false) => meshes::fs_temporal_lit,
            (MeshPass::Opaque(Targets::Temporal), false, true) => meshes::fs_temporal_extended,
            (MeshPass::Opaque(Targets::Temporal), true, true) => meshes::fs_temporal_lit_extended,
            (MeshPass::Opaque(Targets::Debug), _, _) => meshes::fs_debug_mesh,
            (MeshPass::Probe, false, false) => meshes::fs_probe,
            (MeshPass::Probe, true, false) => meshes::fs_probe_lit,
            (MeshPass::Probe, false, true) => meshes::fs_probe_extended,
            (MeshPass::Probe, true, true) => meshes::fs_probe_lit_extended,
        }
    }

    pub(crate) fn prepass_entry(self, extended: bool) -> Entry {
        match (self.targets(), extended) {
            (Some(Targets::Debug), _) | (Some(Targets::Canvas | Targets::Post) | None, true) => {
                meshes::fs_masked_depth_extended
            }
            (Some(Targets::Temporal), true) => meshes::fs_masked_depth_temporal_extended,
            (Some(Targets::Canvas | Targets::Post | Targets::Temporal) | None, false) => {
                meshes::fs_masked_depth
            }
        }
    }
}

pub(crate) const SKY_VERTEX: Entry = meshes::vs_sky;

pub(crate) fn sky_fragment(targets: Option<Targets>) -> Entry {
    match targets {
        Some(Targets::Canvas) => meshes::fs_sky,
        Some(Targets::Post) => meshes::fs_sky_post,
        Some(Targets::Temporal) => meshes::fs_sky_temporal,
        Some(Targets::Debug) => meshes::fs_debug_sky,
        None => meshes::fs_probe_sky,
    }
}

pub(crate) fn impostor_vertex(targets: Targets) -> Entry {
    match targets {
        Targets::Canvas | Targets::Post => meshes::vs_impostor,
        Targets::Temporal => meshes::vs_impostor_temporal,
        Targets::Debug => meshes::vs_debug_impostor,
    }
}

pub(crate) fn impostor_fragment(targets: Targets, lit: bool) -> Entry {
    match (targets, lit) {
        (Targets::Canvas, false) => meshes::fs_impostor,
        (Targets::Canvas, true) => meshes::fs_impostor_lit,
        (Targets::Post, false) => meshes::fs_impostor_post,
        (Targets::Post, true) => meshes::fs_impostor_post_lit,
        (Targets::Temporal, false) => meshes::fs_impostor_temporal,
        (Targets::Temporal, true) => meshes::fs_impostor_temporal_lit,
        (Targets::Debug, _) => meshes::fs_debug_impostor,
    }
}

pub(crate) fn impostor_bake(extended: bool, shade: bool) -> (Entry, Entry) {
    match (extended, shade) {
        (false, false) => (meshes::vs_impostor_bake, meshes::fs_impostor_bake),
        (true, false) => (
            meshes::vs_impostor_bake_extended,
            meshes::fs_impostor_bake_extended,
        ),
        (false, true) => (meshes::vs_impostor_bake, meshes::fs_impostor_shade),
        (true, true) => (
            meshes::vs_impostor_bake_extended,
            meshes::fs_impostor_shade_extended,
        ),
    }
}

pub(crate) fn terrain(targets: Option<Targets>, lit: bool) -> (Entry, Entry) {
    match (targets, lit) {
        (Some(Targets::Canvas), false) => (meshes::vs_terrain, meshes::fs_terrain),
        (Some(Targets::Canvas), true) => (meshes::vs_terrain, meshes::fs_terrain_lit),
        (Some(Targets::Post), false) => (meshes::vs_terrain, meshes::fs_terrain_post),
        (Some(Targets::Post), true) => (meshes::vs_terrain, meshes::fs_terrain_post_lit),
        (Some(Targets::Temporal), false) => {
            (meshes::vs_terrain_temporal, meshes::fs_terrain_temporal)
        }
        (Some(Targets::Temporal), true) => {
            (meshes::vs_terrain_temporal, meshes::fs_terrain_temporal_lit)
        }
        (Some(Targets::Debug), _) => (meshes::vs_terrain, meshes::fs_debug_terrain),
        (None, false) => (meshes::vs_terrain, meshes::fs_terrain_probe),
        (None, true) => (meshes::vs_terrain, meshes::fs_terrain_probe_lit),
    }
}

pub(crate) fn water(targets: Targets) -> (Entry, Entry) {
    match targets {
        Targets::Canvas => (meshes::vs_water, meshes::fs_water),
        Targets::Post => (meshes::vs_water, meshes::fs_water_post),
        Targets::Temporal => (meshes::vs_water, meshes::fs_water_temporal),
        Targets::Debug => (meshes::vs_water, meshes::fs_debug_water),
    }
}

pub(crate) fn falls(targets: Targets) -> (Entry, Entry) {
    match targets {
        Targets::Canvas => (meshes::vs_fall, meshes::fs_fall),
        Targets::Post => (meshes::vs_fall, meshes::fs_fall_post),
        Targets::Temporal => (meshes::vs_fall_temporal, meshes::fs_fall_temporal),
        Targets::Debug => (meshes::vs_fall, meshes::fs_debug_fall),
    }
}

pub(crate) fn cover(targets: Targets, lit: bool) -> (Entry, Entry) {
    match (targets, lit) {
        (Targets::Canvas, false) => (meshes::vs_cover, meshes::fs_cover),
        (Targets::Canvas, true) => (meshes::vs_cover, meshes::fs_cover_lit),
        (Targets::Post, false) => (meshes::vs_cover, meshes::fs_cover_post),
        (Targets::Post, true) => (meshes::vs_cover, meshes::fs_cover_post_lit),
        (Targets::Temporal, false) => (meshes::vs_cover_temporal, meshes::fs_cover_temporal),
        (Targets::Temporal, true) => (meshes::vs_cover_temporal, meshes::fs_cover_temporal_lit),
        (Targets::Debug, _) => (meshes::vs_cover, meshes::fs_debug_cover),
    }
}

pub(crate) const TERRAIN_SHADOW_VERTEX: Entry = terrain_shadow::vs_terrain_shadow;
