//! The mesh shader's entry points (`gpu_data::SHADER`) as typed keys: what
//! a pipeline draws names its vertex and fragment stages here, by
//! exhaustive matches over string literals, and a test checks that the keys
//! name every entry point of the shader and nothing else.

use crate::render_graph::Targets;

/// A mesh pass: opaque into the frame's targets, transparent surfaces
/// accumulated (decision 026), or a reflection probe face (decision 043:
/// linear light, drawn mirrored).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MeshPass {
    Opaque(Targets),
    Transparent,
    Probe,
}

/// What a mesh pipeline draws: its pass and whether it shades local lights
/// (decision 023).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PassKind {
    pub pass: MeshPass,
    pub lit: bool,
}

/// Where a mesh pipeline's slots are posed: as the frame poses them, or
/// in temporal frames also as the previous frame did (decision 017).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slots {
    Retained,
    Temporal,
}

/// The vertex streams a mesh draws from beyond positions, normals and
/// coordinates: none, a second one of colors and second coordinates
/// (decision 034), or that and a skin stream of joints and weights
/// (decision 036).
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
    pub(crate) fn vertex_entry(self, streams: Streams) -> &'static str {
        match (self.slots(), streams) {
            (Slots::Retained, Streams::Basic) => "vs_affine",
            (Slots::Retained, Streams::Extended) => "vs_affine_extended",
            (Slots::Retained, Streams::Skinned) => "vs_affine_skinned",
            (Slots::Temporal, Streams::Basic) => "vs_affine_temporal",
            (Slots::Temporal, Streams::Extended) => "vs_affine_temporal_extended",
            (Slots::Temporal, Streams::Skinned) => "vs_affine_temporal_skinned",
        }
    }

    /// The fragment stage of this pass for meshes with a second vertex
    /// stream (`extended`) or without.
    pub(crate) fn fragment_entry(self, extended: bool) -> &'static str {
        match (self.pass, self.lit, extended) {
            (MeshPass::Opaque(Targets::Canvas), false, false) => "fs_main",
            (MeshPass::Opaque(Targets::Canvas), true, false) => "fs_main_lit",
            (MeshPass::Opaque(Targets::Canvas), false, true) => "fs_main_extended",
            (MeshPass::Opaque(Targets::Canvas), true, true) => "fs_main_lit_extended",
            (MeshPass::Opaque(Targets::Post), false, false) => "fs_post",
            (MeshPass::Opaque(Targets::Post), true, false) => "fs_post_lit",
            (MeshPass::Opaque(Targets::Post), false, true) => "fs_post_extended",
            (MeshPass::Opaque(Targets::Post), true, true) => "fs_post_lit_extended",
            (MeshPass::Opaque(Targets::Temporal), false, false) => "fs_temporal",
            (MeshPass::Opaque(Targets::Temporal), true, false) => "fs_temporal_lit",
            (MeshPass::Opaque(Targets::Temporal), false, true) => "fs_temporal_extended",
            (MeshPass::Opaque(Targets::Temporal), true, true) => "fs_temporal_lit_extended",
            (MeshPass::Transparent, false, false) => "fs_transparent",
            (MeshPass::Transparent, true, false) => "fs_transparent_lit",
            (MeshPass::Transparent, false, true) => "fs_transparent_extended",
            (MeshPass::Transparent, true, true) => "fs_transparent_lit_extended",
            (MeshPass::Probe, false, false) => "fs_probe",
            (MeshPass::Probe, true, false) => "fs_probe_lit",
            (MeshPass::Probe, false, true) => "fs_probe_extended",
            (MeshPass::Probe, true, true) => "fs_probe_lit_extended",
        }
    }
}

/// The sky: one triangle at the far plane (`meshes/sky.wgsl`).
pub(crate) const SKY_VERTEX: &str = "vs_sky";

/// The sky's fragment stage into `targets`, or a reflection probe face's
/// (`None`, decision 043).
pub(crate) fn sky_fragment(targets: Option<Targets>) -> &'static str {
    match targets {
        Some(Targets::Canvas) => "fs_sky",
        Some(Targets::Post) => "fs_sky_post",
        Some(Targets::Temporal) => "fs_sky_temporal",
        None => "fs_probe_sky",
    }
}

/// The vertex stage of virtual geometry's clusters (decision 044) drawn
/// into `targets`; their fragments are the opaque passes'.
pub(crate) fn virtual_vertex(targets: Targets) -> &'static str {
    match targets {
        Targets::Canvas | Targets::Post => "vs_virtual",
        Targets::Temporal => "vs_virtual_temporal",
    }
}

/// The visibility pass's stages (decision 045).
pub(crate) const VISIBILITY_VERTEX: &str = "vs_visibility";
pub(crate) const VISIBILITY_FRAGMENT: &str = "fs_visibility";
/// The vertex stage of the visibility buffer's resolve, over each bin's
/// tiles.
pub(crate) const RESOLVE_VERTEX: &str = "vs_resolve";

/// The resolve's fragment stage into `targets`, `lit` or not.
pub(crate) fn resolve_fragment(targets: Targets, lit: bool) -> &'static str {
    match (targets, lit) {
        (Targets::Canvas, false) => "fs_resolve",
        (Targets::Canvas, true) => "fs_resolve_lit",
        (Targets::Post, false) => "fs_resolve_post",
        (Targets::Post, true) => "fs_resolve_post_lit",
        (Targets::Temporal, false) => "fs_resolve_temporal",
        (Targets::Temporal, true) => "fs_resolve_temporal_lit",
    }
}

/// The vertex stage of impostor cards (decision 048) drawn into `targets`.
pub(crate) fn impostor_vertex(targets: Targets) -> &'static str {
    match targets {
        Targets::Canvas | Targets::Post => "vs_impostor",
        Targets::Temporal => "vs_impostor_temporal",
    }
}

/// The fragment stage of impostor cards into `targets`, `lit` or not.
pub(crate) fn impostor_fragment(targets: Targets, lit: bool) -> &'static str {
    match (targets, lit) {
        (Targets::Canvas, false) => "fs_impostor",
        (Targets::Canvas, true) => "fs_impostor_lit",
        (Targets::Post, false) => "fs_impostor_post",
        (Targets::Post, true) => "fs_impostor_post_lit",
        (Targets::Temporal, false) => "fs_impostor_temporal",
        (Targets::Temporal, true) => "fs_impostor_temporal_lit",
    }
}

/// The vertex and fragment stages baking impostors of meshes with a second
/// vertex stream (`extended`) or without.
pub(crate) fn impostor_bake(extended: bool) -> (&'static str, &'static str) {
    if extended {
        ("vs_impostor_bake_extended", "fs_impostor_bake_extended")
    } else {
        ("vs_impostor_bake", "fs_impostor_bake")
    }
}

/// Terrain's vertex and fragment stages (decision 051) into `targets`, or
/// a reflection probe face's (`None`), `lit` or not.
pub(crate) fn terrain(targets: Option<Targets>, lit: bool) -> (&'static str, &'static str) {
    match (targets, lit) {
        (Some(Targets::Canvas), false) => ("vs_terrain", "fs_terrain"),
        (Some(Targets::Canvas), true) => ("vs_terrain", "fs_terrain_lit"),
        (Some(Targets::Post), false) => ("vs_terrain", "fs_terrain_post"),
        (Some(Targets::Post), true) => ("vs_terrain", "fs_terrain_post_lit"),
        (Some(Targets::Temporal), false) => ("vs_terrain_temporal", "fs_terrain_temporal"),
        (Some(Targets::Temporal), true) => ("vs_terrain_temporal", "fs_terrain_temporal_lit"),
        (None, false) => ("vs_terrain", "fs_terrain_probe"),
        (None, true) => ("vs_terrain", "fs_terrain_probe_lit"),
    }
}

/// Terrain's depth stage in shadow maps (`terrain_shadow.wgsl`).
pub(crate) const TERRAIN_SHADOW_VERTEX: &str = "vs_terrain_shadow";

/// Decals' boxes (decision 039).
pub(crate) const DECAL_VERTEX: &str = "vs_decal";

/// Decals' fragment stage, `lit` or not.
pub(crate) fn decal_fragment(lit: bool) -> &'static str {
    if lit { "fs_decal_lit" } else { "fs_decal" }
}
