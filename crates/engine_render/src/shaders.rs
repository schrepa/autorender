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
    "engine::impostor_bake" = "impostors/bake.wgsl";
    "engine::post::pixel" = "post/pixel.wgsl";
    "engine::post::frame" = "effects.wgsl";
    "engine::post::resolve_plain" = "post/resolve_plain.wgsl";
    "engine::post::levels" = "post/levels.wgsl";
}

pub static CORE: Source = Source::generated(crate::linker::CORE_PACKAGE, false, [0; 32], TEXTS);

pub const FULLSCREEN: engine_core::Entry =
    engine_core::Entry::generated(Module::generated(&CORE, 0), "vs_fullscreen");
pub const FS_COMPOSITE: engine_core::Entry = engine_core::Entry::generated(
    Module::generated(&CORE, (TEXTS.len() - 1) as u32),
    "fs_composite",
);
pub const FS_COMPOSITE_PLAIN: engine_core::Entry = engine_core::Entry::generated(
    Module::generated(&CORE, (TEXTS.len() - 1) as u32),
    "fs_composite_plain",
);
pub const FS_DEBUG_VIEW: engine_core::Entry = engine_core::Entry::generated(
    Module::generated(&CORE, (TEXTS.len() - 1) as u32),
    "fs_debug_view",
);

pub const PRESENTATION_MODULE: &str = "engine::presentation";
pub const DEBUG_MODULE: &str = "engine::debug";
pub const MATERIALS_MODULE: &str = "engine::materials";
pub const IMPOSTOR_FRAMES_MODULE: &str = "engine::mesh::impostor_frames";
pub const IMPOSTORS_MODULE: &str = "engine::impostors";
pub const MATH_MODULE: &str = "engine::math";
pub const VIEW_MODULE: &str = "engine::view";
pub const PYRAMID_MODULE: &str = "engine::pyramid";

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
        (
            "POSE_FLOATS",
            (size_of::<engine_core::NodePose>() / 4) as u32,
        ),
    ] {
        unsigned(&mut source, name, value);
    }
    for (name, value) in [
        ("MAX_COEFFICIENT", bounds::MAX_COEFFICIENT),
        ("ROUNDING_MARGIN", bounds::ROUNDING_MARGIN),
        ("MAX_PRODUCT_BOUND", bounds::MAX_PRODUCT_BOUND),
        ("COEFFICIENT_LIMIT", bounds::COEFFICIENT_LIMIT),
        ("MIN_LINEAR_SCALE", bounds::MIN_LINEAR_SCALE),
        (
            "COMMON_DETERMINANT_MARGIN",
            bounds::COMMON_DETERMINANT_MARGIN,
        ),
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

pub fn materials_source() -> String {
    use crate::gpu_layouts::constants::{WGSL_FLOAT, WGSL_UNSIGNED};
    let mut source = String::from(include_str!("meshes/materials.wgsl"));
    source.push_str(<crate::gpu_layouts::scene::MaterialUniform as engine_core::GpuStruct>::WGSL);
    for (name, value) in WGSL_UNSIGNED {
        unsigned(&mut source, name, value);
    }
    for (name, value) in WGSL_FLOAT {
        float(&mut source, name, value);
    }
    source
}

pub fn impostor_frames_source() -> String {
    let mut source = String::from(include_str!("impostor_frames.wgsl"));
    for (name, value) in crate::gpu_layouts::constants::CARD_FLAGS {
        unsigned(&mut source, name, value);
    }
    source.push_str(&crate::impostors::layout_source());
    source
}

pub fn math_source() -> String {
    let mut source = String::from(include_str!("math.wgsl"));
    float(&mut source, "GRAVITY", engine_core::math::GRAVITY);
    source
}

pub fn view_source() -> String {
    let mut source = String::from(include_str!("frame/view.wgsl"));
    float(
        &mut source,
        "DETAIL_ANISOTROPY",
        f32::from(crate::MATERIAL_ANISOTROPY),
    );
    source
}

pub fn generated_core() -> [(&'static str, String); 8] {
    [
        (PRESENTATION_MODULE, presentation_source()),
        (DEBUG_MODULE, debug_source()),
        (MATERIALS_MODULE, materials_source()),
        (IMPOSTOR_FRAMES_MODULE, impostor_frames_source()),
        (IMPOSTORS_MODULE, crate::impostors::atlas_source()),
        (MATH_MODULE, math_source()),
        (VIEW_MODULE, view_source()),
        (PYRAMID_MODULE, crate::pyramid::source()),
    ]
}

pub fn fixed_linker() -> crate::fixed_linker::Linker {
    let mut linker = crate::fixed_linker::Linker::new(MODULES);
    for (path, text) in generated_core() {
        linker.add_generated(path.to_owned(), "core", text);
    }
    linker
}

#[doc(hidden)]
pub fn core_modules() -> Vec<(&'static str, String)> {
    let generated = generated_core();
    let mut modules: Vec<(&'static str, String)> = TEXTS
        .iter()
        .filter(|text| !text.text().is_empty())
        .filter(|text| !generated.iter().any(|(path, _)| *path == text.path()))
        .map(|text| (text.path(), text.text().to_owned()))
        .collect();
    modules.extend(generated);
    modules
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
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
