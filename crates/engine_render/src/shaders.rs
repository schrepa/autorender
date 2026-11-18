use std::fmt::Write as _;

use engine_core::{Module, ShaderText, Source};

macro_rules! shader_modules {
    ($($path:literal = $file:literal;)*) => {
        const TEXTS: &[ShaderText] = &[
            $(ShaderText::generated(
                $path,
                include_str!($file),
                concat!(env!("CARGO_MANIFEST_DIR"), "/src/", $file),
            ),)*
            crate::shared::generated(crate::drawn::DRAWN_MODULE),
            crate::shared::generated(crate::drawn::DRAWN_POSE_MODULE),
        ];
    };
}

shader_modules! {
    "engine::fullscreen" = "fullscreen.wgsl";
    "engine::color" = "color.wgsl";
    "engine::math" = "math.wgsl";
    "engine::mipmaps" = "mipmaps.wgsl";
    "engine::view" = "frame/view.wgsl";
    "engine::post::frame" = "effects.wgsl";
}

pub static CORE: Source = Source::generated(crate::linker::CORE_PACKAGE, false, [0; 32], TEXTS);

pub const FULLSCREEN: engine_core::Entry =
    engine_core::Entry::generated(Module::generated(&CORE, 0), "vs_fullscreen");

pub(crate) fn core_module(path: &str) -> Module {
    let index = TEXTS.iter().position(|text| text.path() == path);
    Module::generated(
        &CORE,
        index.unwrap_or_else(|| panic!("core_module_listed {path}")) as u32,
    )
}

pub const PRESENTATION_MODULE: &str = "engine::presentation";
pub const VIEW_MODULE: &str = "engine::view";

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

pub fn view_source() -> String {
    use engine_core::GpuStruct;
    let mut source = String::from(include_str!("frame/view.wgsl"));
    source.push_str(<crate::CameraUniform as GpuStruct>::WGSL);
    source
}

pub fn generated_core() -> [(&'static str, String); 4] {
    [
        (PRESENTATION_MODULE, presentation_source()),
        (VIEW_MODULE, view_source()),
        (crate::drawn::DRAWN_MODULE, crate::drawn::source()),
        (crate::drawn::DRAWN_POSE_MODULE, crate::drawn::pose_source()),
    ]
}
