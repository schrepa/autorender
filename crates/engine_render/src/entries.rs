//! The mesh shader's entry points (`gpu_data::SHADER`) by what they draw:
//! a pipeline names its vertex and fragment stages here, by exhaustive
//! matches over the entry points `crate::shaders::entry` declares.

use crate::render_graph::Targets;
use crate::shaders::Entry;
use crate::shaders::entry::{meshes, terrain_shadow};

/// A mesh pass: opaque into the frame's targets, transparent surfaces
/// accumulated, or a reflection probe face (
/// linear light, drawn mirrored).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MeshPass {
    Opaque(Targets),
    Transparent,
    Probe,
}

/// What a mesh pipeline draws: its pass and whether it shades local lights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PassKind {
    pub pass: MeshPass,
    pub lit: bool,
}

/// Where a mesh pipeline's slots are posed: as the frame poses them, or
/// in temporal frames also as the previous frame did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slots {
    Retained,
    Temporal,
}

/// The vertex streams a mesh draws from beyond positions, normals and
/// coordinates: none, a second one of colors and second coordinates,
/// or that and a skin stream of joints and weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Streams {
    Basic,
    Extended,
    Skinned,
}

impl PassKind {
    pub(crate) fn opaque(targets: Targets, lit: bool) -> Self {
        Self {
            pass: MeshPass::Opaque(targets),
            lit,
        }
    }

    pub(crate) fn transparent(lit: bool) -> Self {
        Self {
            pass: MeshPass::Transparent,
            lit,
        }
    }

    pub(crate) fn probe(lit: bool) -> Self {
        Self {
            pass: MeshPass::Probe,
            lit,
        }
    }

    /// The targets of an opaque pass; `None` for the others.
    pub(crate) fn targets(self) -> Option<Targets> {
        match self.pass {
            MeshPass::Opaque(targets) => Some(targets),
            MeshPass::Transparent | MeshPass::Probe => None,
        }
    }

    /// How this pass poses its slots.
    pub(crate) fn slots(self) -> Slots {
        if self.targets() == Some(Targets::Temporal) {
            Slots::Temporal
        } else {
            Slots::Retained
        }
    }

    /// The vertex stage drawing meshes of `streams` in this pass.
    pub(crate) fn vertex_entry(self, streams: Streams) -> Entry {
        match (self.slots(), streams) {
            (Slots::Retained, Streams::Basic) => meshes::vs_affine,
            (Slots::Retained, Streams::Extended) => meshes::vs_affine_extended,
            (Slots::Retained, Streams::Skinned) => meshes::vs_affine_skinned,
            (Slots::Temporal, Streams::Basic) => meshes::vs_affine_temporal,
            (Slots::Temporal, Streams::Extended) => meshes::vs_affine_temporal_extended,
            (Slots::Temporal, Streams::Skinned) => meshes::vs_affine_temporal_skinned,
        }
    }

    /// The fragment stage of this pass for meshes with a second vertex
    /// stream (`extended`) or without.
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
            (MeshPass::Transparent, false, false) => meshes::fs_transparent,
            (MeshPass::Transparent, true, false) => meshes::fs_transparent_lit,
            (MeshPass::Transparent, false, true) => meshes::fs_transparent_extended,
            (MeshPass::Transparent, true, true) => meshes::fs_transparent_lit_extended,
            (MeshPass::Probe, false, false) => meshes::fs_probe,
            (MeshPass::Probe, true, false) => meshes::fs_probe_lit,
            (MeshPass::Probe, false, true) => meshes::fs_probe_extended,
            (MeshPass::Probe, true, true) => meshes::fs_probe_lit_extended,
        }
    }
}

/// The sky: one triangle at the far plane (`meshes/sky.wgsl`).
pub(crate) const SKY_VERTEX: Entry = meshes::vs_sky;

/// The sky's fragment stage into `targets`, or a reflection probe face's
/// (`None`).
pub(crate) fn sky_fragment(targets: Option<Targets>) -> Entry {
    match targets {
        Some(Targets::Canvas) => meshes::fs_sky,
        Some(Targets::Post) => meshes::fs_sky_post,
        Some(Targets::Temporal) => meshes::fs_sky_temporal,
        None => meshes::fs_probe_sky,
    }
}

/// The vertex stage of virtual geometry's clusters drawn
/// into `targets`; their fragments are the opaque passes'.
pub(crate) fn virtual_vertex(targets: Targets) -> Entry {
    match targets {
        Targets::Canvas | Targets::Post => meshes::vs_virtual,
        Targets::Temporal => meshes::vs_virtual_temporal,
    }
}

/// The visibility pass's stages.
pub(crate) const VISIBILITY_VERTEX: Entry = meshes::vs_visibility;
pub(crate) const VISIBILITY_FRAGMENT: Entry = meshes::fs_visibility;
/// The vertex stage of the visibility buffer's resolve, over each bin's
/// tiles.
pub(crate) const RESOLVE_VERTEX: Entry = meshes::vs_resolve;

/// The resolve's fragment stage into `targets`, `lit` or not.
pub(crate) fn resolve_fragment(targets: Targets, lit: bool) -> Entry {
    match (targets, lit) {
        (Targets::Canvas, false) => meshes::fs_resolve,
        (Targets::Canvas, true) => meshes::fs_resolve_lit,
        (Targets::Post, false) => meshes::fs_resolve_post,
        (Targets::Post, true) => meshes::fs_resolve_post_lit,
        (Targets::Temporal, false) => meshes::fs_resolve_temporal,
        (Targets::Temporal, true) => meshes::fs_resolve_temporal_lit,
    }
}

/// The vertex stage of impostor cards drawn into `targets`.
pub(crate) fn impostor_vertex(targets: Targets) -> Entry {
    match targets {
        Targets::Canvas | Targets::Post => meshes::vs_impostor,
        Targets::Temporal => meshes::vs_impostor_temporal,
    }
}

/// The fragment stage of impostor cards into `targets`, `lit` or not.
pub(crate) fn impostor_fragment(targets: Targets, lit: bool) -> Entry {
    match (targets, lit) {
        (Targets::Canvas, false) => meshes::fs_impostor,
        (Targets::Canvas, true) => meshes::fs_impostor_lit,
        (Targets::Post, false) => meshes::fs_impostor_post,
        (Targets::Post, true) => meshes::fs_impostor_post_lit,
        (Targets::Temporal, false) => meshes::fs_impostor_temporal,
        (Targets::Temporal, true) => meshes::fs_impostor_temporal_lit,
    }
}

/// The vertex and fragment stages baking impostors of meshes with a second
/// vertex stream (`extended`) or without: their colors, or with `shade`
/// their frames' nearest depths.
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

/// Terrain's vertex and fragment stages into `targets`, or
/// a reflection probe face's (`None`), `lit` or not.
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
        (None, false) => (meshes::vs_terrain, meshes::fs_terrain_probe),
        (None, true) => (meshes::vs_terrain, meshes::fs_terrain_probe_lit),
    }
}

/// Water's vertex and fragment stages into `targets`.
pub(crate) fn water(targets: Targets) -> (Entry, Entry) {
    match targets {
        Targets::Canvas => (meshes::vs_water, meshes::fs_water),
        Targets::Post => (meshes::vs_water, meshes::fs_water_post),
        Targets::Temporal => (meshes::vs_water_temporal, meshes::fs_water_temporal),
    }
}

/// The sea's vertex and fragment stages into `targets`.
pub(crate) fn ocean(targets: Targets) -> (Entry, Entry) {
    match targets {
        Targets::Canvas => (meshes::vs_ocean, meshes::fs_ocean),
        Targets::Post => (meshes::vs_ocean, meshes::fs_ocean_post),
        Targets::Temporal => (meshes::vs_ocean_temporal, meshes::fs_ocean_temporal),
    }
}

/// Falls' sheets' vertex and fragment stages into `targets`.
pub(crate) fn falls(targets: Targets) -> (Entry, Entry) {
    match targets {
        Targets::Canvas => (meshes::vs_fall, meshes::fs_fall),
        Targets::Post => (meshes::vs_fall, meshes::fs_fall_post),
        Targets::Temporal => (meshes::vs_fall_temporal, meshes::fs_fall_temporal),
    }
}

/// Ground cover's vertex and fragment stages into
/// `targets`, `lit` or not.
pub(crate) fn cover(targets: Targets, lit: bool) -> (Entry, Entry) {
    match (targets, lit) {
        (Targets::Canvas, false) => (meshes::vs_cover, meshes::fs_cover),
        (Targets::Canvas, true) => (meshes::vs_cover, meshes::fs_cover_lit),
        (Targets::Post, false) => (meshes::vs_cover, meshes::fs_cover_post),
        (Targets::Post, true) => (meshes::vs_cover, meshes::fs_cover_post_lit),
        (Targets::Temporal, false) => (meshes::vs_cover_temporal, meshes::fs_cover_temporal),
        (Targets::Temporal, true) => (meshes::vs_cover_temporal, meshes::fs_cover_temporal_lit),
    }
}

/// Terrain's depth stage in shadow maps (`terrain_shadow.wgsl`).
pub(crate) const TERRAIN_SHADOW_VERTEX: Entry = terrain_shadow::vs_terrain_shadow;

/// Decals' boxes.
pub(crate) const DECAL_VERTEX: Entry = meshes::vs_decal;

/// Decals' fragment stage, `lit` or not.
pub(crate) fn decal_fragment(lit: bool) -> Entry {
    if lit {
        meshes::fs_decal_lit
    } else {
        meshes::fs_decal
    }
}
