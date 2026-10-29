use engine_render::entries::{MeshEntries, MeshPass, PassKind};
use engine_render::graph::Targets;
use engine_render::shaders::Entry;

use crate::shaders::entry::meshes;

pub(crate) static ENTRIES: MeshEntries = MeshEntries {
    fragment,
    prepass,
    card_vertex,
    card_fragment,
    card_bake,
};

fn fragment(kind: PassKind, extended: bool) -> Entry {
    match (kind.pass, extended) {
        (MeshPass::Opaque(Targets::Canvas), false) => meshes::fs_main,
        (MeshPass::Opaque(Targets::Canvas), true) => meshes::fs_main_extended,
        (MeshPass::Opaque(Targets::Post), false) => meshes::fs_post,
        (MeshPass::Opaque(Targets::Post), true) => meshes::fs_post_extended,
        (MeshPass::Opaque(Targets::Temporal), false) => meshes::fs_temporal,
        (MeshPass::Opaque(Targets::Temporal), true) => meshes::fs_temporal_extended,
        (MeshPass::Opaque(Targets::Debug), _) => meshes::fs_debug_mesh,
        (MeshPass::View, false) => meshes::fs_probe,
        (MeshPass::View, true) => meshes::fs_probe_extended,
    }
}

fn prepass(kind: PassKind, extended: bool) -> Entry {
    match (kind.targets(), extended) {
        (Some(Targets::Debug), _) | (Some(Targets::Canvas | Targets::Post) | None, true) => {
            meshes::fs_masked_depth_extended
        }
        (Some(Targets::Temporal), true) => meshes::fs_masked_depth_temporal_extended,
        (Some(Targets::Canvas | Targets::Post | Targets::Temporal) | None, false) => {
            meshes::fs_masked_depth
        }
    }
}

fn card_vertex(targets: Targets) -> Entry {
    match targets {
        Targets::Canvas | Targets::Post => meshes::vs_impostor,
        Targets::Temporal => meshes::vs_impostor_temporal,
        Targets::Debug => meshes::vs_debug_impostor,
    }
}

fn card_fragment(targets: Targets) -> Entry {
    match targets {
        Targets::Canvas => meshes::fs_impostor,
        Targets::Post => meshes::fs_impostor_post,
        Targets::Temporal => meshes::fs_impostor_temporal,
        Targets::Debug => meshes::fs_debug_impostor,
    }
}

fn card_bake(extended: bool, shade: bool) -> (Entry, Entry) {
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
