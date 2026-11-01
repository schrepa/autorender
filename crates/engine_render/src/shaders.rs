use std::fmt::Write as _;

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
pub const FS_DEBUG_VIEW: engine_core::Entry = engine_core::Entry::generated(
    Module::generated(&CORE, (TEXTS.len() - 1) as u32),
    "fs_debug_view",
);

pub const PRESENTATION_MODULE: &str = "engine::presentation";
pub const DEBUG_MODULE: &str = "engine::debug";

fn unsigned(source: &mut String, name: &str, value: u32) {
    let _ = writeln!(source, "const {name}: u32 = {value}u;");
}

fn float(source: &mut String, name: &str, value: f32) {
    let _ = writeln!(source, "const {name}: f32 = {:?};", f64::from(value));
}

pub fn presentation_source() -> String {
    use engine_core::presentation_bounds as bounds;
    let mut source = String::new();
    for (name, value) in [
        ("NO_PARENT", engine_core::NO_PARENT),
        ("NODE_VALID", engine_core::NODE_VALID),
        ("NODE_BLEND", engine_core::NODE_BLEND),
        ("NODE_ACTIVE", engine_core::NODE_ACTIVE),
        ("NODE_PARENT_ACTIVE", engine_core::NODE_PARENT_ACTIVE),
        ("IMPOSTOR_VERTICES", crate::IMPOSTOR_VERTICES),
    ] {
        unsigned(&mut source, name, value);
    }
    for (name, value) in [
        ("MAX_COEFFICIENT", bounds::MAX_COEFFICIENT),
        ("ROUNDING_MARGIN", bounds::ROUNDING_MARGIN),
        ("MAX_PRODUCT_BOUND", bounds::MAX_PRODUCT_BOUND),
        ("COEFFICIENT_LIMIT", bounds::COEFFICIENT_LIMIT),
        ("MIN_LINEAR_SCALE", bounds::MIN_LINEAR_SCALE),
        ("COMMON_DETERMINANT_MARGIN", bounds::COMMON_DETERMINANT_MARGIN),
        ("LINEAR_ROTATION_DOT", bounds::LINEAR_ROTATION_DOT_THRESHOLD),
    ] {
        float(&mut source, name, value);
    }
    source
}

pub fn debug_source() -> String {
    let mut source = String::new();
    for &(name, value) in crate::debug::DECLARED {
        unsigned(&mut source, name, value);
    }
    source
}

pub fn generated_core() -> [(&'static str, String); 2] {
    [
        (PRESENTATION_MODULE, presentation_source()),
        (DEBUG_MODULE, debug_source()),
    ]
}

#[doc(hidden)]
pub fn core_modules() -> Vec<(&'static str, String)> {
    TEXTS
        .iter()
        .filter(|text| !text.text().is_empty())
        .map(|text| (text.path(), text.text().to_owned()))
        .chain(generated_core())
        .collect()
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
    };
}

entry_points! {
    mipmaps {
        vertex: [vs_mip @ "engine::mipmaps"],
        fragment: [fs_mip @ "engine::mipmaps"],
    }
}
