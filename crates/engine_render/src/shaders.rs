use std::fmt::Write as _;

use engine_core::{Module, ShaderText, Source};

macro_rules! shader_modules {
    ($($path:literal = $file:literal;)*) => {
        const TEXTS: &[ShaderText] = &[
            $(ShaderText::generated($path, include_str!($file)),)*
            crate::shared::generated(crate::drawn::DRAWN_MODULE),
            crate::shared::generated(crate::drawn::DRAWN_CULL_MODULE),
            crate::shared::generated(crate::drawn::DRAWN_FADES_MODULE),
            crate::shared::generated(crate::shared::TARGETS_MODULE),
            crate::shared::generated(crate::shared::APPLIED_MODULE),
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
    "engine::standard" = "standard.wgsl";
    "engine::surface" = "surface.wgsl";
    "engine::transparency" = "transparency.wgsl";
    "engine::mesh::vertex" = "meshes/vertex.wgsl";
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
pub const FS_DEBUG_VIEW: engine_core::Entry = engine_core::Entry::generated(
    Module::generated(&CORE, (TEXTS.len() - 1) as u32),
    "fs_debug_view",
);

pub(crate) fn core_module(path: &str) -> Module {
    let index = TEXTS.iter().position(|text| text.path() == path);
    Module::generated(&CORE, index.unwrap_or_else(|| panic!("core_module_listed {path}")) as u32)
}

pub const PRESENTATION_MODULE: &str = "engine::presentation";
pub const DEBUG_MODULE: &str = "engine::debug";
pub const DECLARATIONS_MODULE: &str = "engine::mesh::declarations";
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
    let mut source = String::from(<engine_core::PoseWord as engine_core::GpuBits>::WGSL);
    source.push_str(engine_core::PRESENTATION_WGSL);
    for (name, value) in [
        ("NO_PARENT", engine_core::NO_PARENT),
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
        ("POSE_FLOAT_LIMIT", bounds::POSE_FLOAT_LIMIT),
    ] {
        float(&mut source, name, value);
    }
    source
}

pub fn debug_source() -> String {
    use engine_core::GpuBits;
    let mut source = String::from(<crate::debug::DebugRecord as GpuBits>::WGSL);
    unsigned(&mut source, "DEBUG_NO_MATERIAL", crate::debug::DEBUG_NO_MATERIAL);
    source.push_str(
        "
fn debug_kind_color(kind: u32) -> vec3<f32> {
    let hue = fract(f32(kind) * 0.618034 + 0.13);
    let k = (vec3<f32>(0.0, 2.0, 4.0) / 6.0 + hue) * 6.0;
    return 0.2 + 0.7 * saturate(abs(k % 6.0 - 3.0) - 1.0);
}
",
    );
    source
}

pub fn declarations_source() -> String {
    use crate::slots::{DrawnSlot, FADE_LEVEL_MAX, FadeWord, FadedSlot};
    use engine_core::GpuBits;
    let mut source = String::from(include_str!("meshes/declarations.wgsl"));
    for wgsl in [DrawnSlot::WGSL, FadedSlot::WGSL, FadeWord::WGSL] {
        source.push_str(wgsl);
    }
    unsigned(&mut source, "FADE_LEVEL_MAX", FADE_LEVEL_MAX);
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

pub fn generated_core() -> [(&'static str, String); 12] {
    let standard = crate::surface::MaterialLayout::standard();
    [
        (PRESENTATION_MODULE, presentation_source()),
        (DEBUG_MODULE, debug_source()),
        (DECLARATIONS_MODULE, declarations_source()),
        (crate::surface::MATERIALS_MODULE, standard.materials_source()),
        (crate::surface::SLOTS_MODULE, standard.slots_source(false)),
        (crate::surface::SURFACES_MODULE, standard.surfaces_source()),
        (MATH_MODULE, math_source()),
        (VIEW_MODULE, view_source()),
        (PYRAMID_MODULE, crate::pyramid::source()),
        (crate::drawn::DRAWN_MODULE, crate::drawn::source()),
        (crate::drawn::DRAWN_CULL_MODULE, crate::drawn::cull_source()),
        (crate::drawn::DRAWN_FADES_MODULE, crate::drawn::fades_source()),
    ]
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
