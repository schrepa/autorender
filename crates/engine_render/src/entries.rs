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

/// What a mesh pipeline draws: its pass, whether it shades local lights
/// (decision 023), and whether it draws retained slots (`affine`);
/// temporal frames and probe faces always do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PassKind {
    pub pass: MeshPass,
    pub lit: bool,
    pub affine: bool,
}

/// Where a mesh pipeline's instances come from: public instance records,
/// retained slots, or retained slots of temporal frames, which also pose
/// them as the previous frame did (decision 017).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slots {
    Instances,
    Retained,
    Temporal,
}

/// The vertex streams a mesh draws from beyond positions, normals and
/// coordinates: none, a second one of colors and second coordinates
/// (decision 034), or that and a skin stream of joints and weights
/// (decision 036; retained slots only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Streams {
    Basic,
    Extended,
    Skinned,
}

impl PassKind {
    pub(crate) fn opaque(targets: Targets, lit: bool, affine: bool) -> Self {
        Self {
            pass: MeshPass::Opaque(targets),
            lit,
            affine: affine || targets == Targets::Temporal,
        }
    }

    pub(crate) fn transparent(lit: bool, affine: bool) -> Self {
        Self {
            pass: MeshPass::Transparent,
            lit,
            affine,
        }
    }

    pub(crate) fn probe(lit: bool) -> Self {
        Self {
            pass: MeshPass::Probe,
            lit,
            affine: true,
        }
    }

    /// The targets of an opaque pass; `None` for the others.
    pub(crate) fn targets(self) -> Option<Targets> {
        match self.pass {
            MeshPass::Opaque(targets) => Some(targets),
            MeshPass::Transparent | MeshPass::Probe => None,
        }
    }

    /// Where this pass's instances come from.
    pub(crate) fn slots(self) -> Slots {
        if self.targets() == Some(Targets::Temporal) {
            Slots::Temporal
        } else if self.affine {
            Slots::Retained
        } else {
            Slots::Instances
        }
    }

    /// The vertex stage drawing meshes of `streams` in this pass; public
    /// instances never skin.
    pub(crate) fn vertex_entry(self, streams: Streams) -> &'static str {
        match (self.slots(), streams) {
            (Slots::Instances, Streams::Basic) => "vs_main",
            (Slots::Instances, Streams::Extended | Streams::Skinned) => "vs_main_extended",
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

/// Decals' boxes (decision 039).
pub(crate) const DECAL_VERTEX: &str = "vs_decal";

/// Decals' fragment stage, `lit` or not.
pub(crate) fn decal_fragment(lit: bool) -> &'static str {
    if lit { "fs_decal_lit" } else { "fs_decal" }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::gpu_data::SHADER;

    const TARGETS: [Targets; 3] = [Targets::Canvas, Targets::Post, Targets::Temporal];
    const STREAMS: [Streams; 3] = [Streams::Basic, Streams::Extended, Streams::Skinned];

    /// Every pass kind the renderer builds mesh pipelines for.
    fn pass_kinds() -> Vec<PassKind> {
        let mut kinds = Vec::new();
        for lit in [false, true] {
            for affine in [false, true] {
                kinds.extend(TARGETS.map(|targets| PassKind::opaque(targets, lit, affine)));
                kinds.push(PassKind::transparent(lit, affine));
            }
            kinds.push(PassKind::probe(lit));
        }
        kinds
    }

    /// Every entry point a key names, with its stage.
    fn named() -> BTreeSet<(&'static str, naga::ShaderStage)> {
        use naga::ShaderStage::{Fragment, Vertex};
        let mut named = BTreeSet::new();
        for kind in pass_kinds() {
            for streams in STREAMS {
                named.insert((kind.vertex_entry(streams), Vertex));
            }
            for extended in [false, true] {
                named.insert((kind.fragment_entry(extended), Fragment));
            }
        }
        named.insert((SKY_VERTEX, Vertex));
        named.insert((sky_fragment(None), Fragment));
        for targets in TARGETS {
            named.insert((sky_fragment(Some(targets)), Fragment));
            named.insert((virtual_vertex(targets), Vertex));
            named.insert((impostor_vertex(targets), Vertex));
            for lit in [false, true] {
                named.insert((resolve_fragment(targets, lit), Fragment));
                named.insert((impostor_fragment(targets, lit), Fragment));
            }
        }
        named.insert((VISIBILITY_VERTEX, Vertex));
        named.insert((VISIBILITY_FRAGMENT, Fragment));
        named.insert((RESOLVE_VERTEX, Vertex));
        for extended in [false, true] {
            let (vertex, fragment) = impostor_bake(extended);
            named.insert((vertex, Vertex));
            named.insert((fragment, Fragment));
        }
        named.insert((DECAL_VERTEX, Vertex));
        for lit in [false, true] {
            named.insert((decal_fragment(lit), Fragment));
        }
        named
    }

    #[test]
    fn keys_name_every_mesh_entry_point_and_nothing_else() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("WGSL syntax");
        let shader: BTreeSet<(&str, naga::ShaderStage)> = module
            .entry_points
            .iter()
            .map(|entry| (entry.name.as_str(), entry.stage))
            .collect();
        let named = named();
        let missing: Vec<_> = named.difference(&shader).collect();
        let unnamed: Vec<_> = shader.difference(&named).collect();
        assert!(
            missing.is_empty(),
            "keys naming no entry point: {missing:?}"
        );
        assert!(unnamed.is_empty(), "entry points no key names: {unnamed:?}");
    }

    #[test]
    fn passes_draw_their_slots() {
        assert_eq!(
            PassKind::opaque(Targets::Post, false, false).slots(),
            Slots::Instances
        );
        assert_eq!(
            PassKind::opaque(Targets::Post, false, true).slots(),
            Slots::Retained
        );
        // Temporal frames and probe faces draw retained slots.
        assert_eq!(
            PassKind::opaque(Targets::Temporal, true, false).slots(),
            Slots::Temporal
        );
        assert_eq!(PassKind::probe(false).slots(), Slots::Retained);
        assert_eq!(
            PassKind::transparent(false, false).vertex_entry(Streams::Skinned),
            "vs_main_extended"
        );
    }
}
