use engine_core::{Module, ShaderText, Source};

macro_rules! shader_modules {
    ($($path:literal = $file:literal;)*) => {
        pub const MODULES: &[crate::fixed_linker::ShaderModule] = &[$(crate::fixed_linker::ShaderModule {
            path: $path,
            brick: "core",
            source: include_str!($file),
        }),*];

        const TEXTS: &[ShaderText] = &[
            $(ShaderText::generated($path, include_str!($file)),)*
            crate::shared::generated(crate::shared::TARGETS_MODULE),
            crate::shared::generated(crate::shared::COMPOSITE_MODULE),
        ];
    };
}

shader_modules! {
    "engine::fullscreen" = "fullscreen.wgsl";
    "engine::color" = "color.wgsl";
    "engine::math" = "math.wgsl";
    "engine::screen_points" = "screen_points.wgsl";
    "engine::mipmaps" = "mipmaps.wgsl";
    "engine::noise" = "noise.wgsl";
    "engine::view" = "frame/view.wgsl";
    "engine::shading" = "shading.wgsl";
    "engine::mesh::declarations" = "meshes/declarations.wgsl";
    "engine::mesh::vertex" = "meshes/vertex.wgsl";
    "engine::mesh::impostor_frames" = "impostor_frames.wgsl";
    "engine::pyramid" = "depth_pyramid.wgsl";
    "engine::post::pixel" = "post/pixel.wgsl";
    "engine::post::frame" = "effects.wgsl";
    "engine::post::resolve_plain" = "post/resolve_plain.wgsl";
    "engine::post::levels" = "post/levels.wgsl";
}

pub static CORE: Source = Source::generated(crate::linker::CORE_PACKAGE, [0; 32], TEXTS);

pub const FULLSCREEN: engine_core::Entry =
    engine_core::Entry::generated(Module::generated(&CORE, 0), "vs_fullscreen");
pub const FS_COMPOSITE: engine_core::Entry = engine_core::Entry::generated(
    Module::generated(&CORE, (TEXTS.len() - 1) as u32),
    "fs_composite",
);

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
    };
}

entry_points! {
    mipmaps {
        vertex: [vs_mip @ "engine::mipmaps"],
        fragment: [fs_mip @ "engine::mipmaps"],
    }
}
