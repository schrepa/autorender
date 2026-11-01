macro_rules! shader_modules {
    ($($path:literal in $brick:literal = $file:literal;)*) => {
        pub const MODULES: &[crate::linker::ShaderModule] = &[$(crate::linker::ShaderModule {
            path: $path,
            brick: $brick,
            source: include_str!($file),
        }),*];
    };
}

shader_modules! {
    "engine::color" in "core" = "color.wgsl";
    "engine::math" in "core" = "math.wgsl";
    "engine::fullscreen" in "core" = "fullscreen.wgsl";
    "engine::screen_points" in "core" = "screen_points.wgsl";
    "engine::mipmaps" in "core" = "mipmaps.wgsl";
    "engine::noise" in "core" = "noise.wgsl";
    "engine::view" in "core" = "frame/view.wgsl";
    "engine::shading" in "core" = "shading.wgsl";
    "meshes::declarations" in "core" = "meshes/declarations.wgsl";
    "meshes::vertex" in "core" = "meshes/vertex.wgsl";
    "meshes::surface" in "core" = "meshes/surface.wgsl";
    "meshes::impostor_frames" in "core" = "impostor_frames.wgsl";
    "engine::pyramid" in "core" = "depth_pyramid.wgsl";
    "engine::post::pixel" in "core" = "post/pixel.wgsl";
    "engine::post::frame" in "core" = "effects.wgsl";
    "engine::post::resolve_plain" in "core" = "post/resolve_plain.wgsl";
    "engine::post::levels" in "core" = "post/levels.wgsl";
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub module: &'static str,
    pub name: &'static str,
    pub stage: Stage,
}

macro_rules! entry_stage {
    (vertex) => {
        Stage::Vertex
    };
    (fragment) => {
        Stage::Fragment
    };
    (compute) => {
        Stage::Compute
    };
}

macro_rules! entry_points {
    ($($rust:ident {
        $($stage:ident: [$($name:ident @ $module:literal),* $(,)?]),* $(,)?
    })*) => {
        pub mod entry {
            $(
                #[expect(non_upper_case_globals, reason = "named as their WGSL functions")]
                pub mod $rust {
                    use super::super::{Entry, Stage};
                    $($(
                        pub const $name: Entry = Entry {
                            module: $module,
                            name: stringify!($name),
                            stage: entry_stage!($stage),
                        };
                    )*)*
                }
            )*
        }

        pub const ENTRIES: &[Entry] = &[$($($(entry::$rust::$name,)*)*)*];
    };
}

entry_points! {
    meshes {
        vertex: [
            vs_affine @ "meshes::vertex",
            vs_affine_temporal @ "meshes::vertex",
            vs_affine_extended @ "meshes::vertex",
            vs_affine_temporal_extended @ "meshes::vertex",
            vs_debug_affine @ "meshes::debug",
            vs_debug_affine_extended @ "meshes::debug",
        ],
    }
    presentation {
        compute: [
            shift_previous @ "presentation::cull",
            compose_parents @ "presentation::cull",
            cull @ "presentation::cull",
            fade_levels @ "presentation::cull",
            occlusion_test @ "presentation::cull",
            count_still @ "presentation::cull",
            probe_cull @ "presentation::cull",
            sum_groups @ "presentation::cull",
            scan_groups @ "presentation::cull",
            batch_args @ "presentation::cull",
            scatter @ "presentation::cull",
        ],
    }
    depth_pyramid {
        compute: [
            pyramid_first @ "engine::pyramid",
            pyramid_down @ "engine::pyramid",
            pyramid_first_nearest @ "engine::pyramid",
            pyramid_down_nearest @ "engine::pyramid",
        ],
    }
    mipmaps {
        vertex: [vs_mip @ "engine::mipmaps"],
        fragment: [fs_mip @ "engine::mipmaps"],
    }
    transparent {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
    }
    post {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [fs_levels_down @ "engine::post::levels"],
    }
}
