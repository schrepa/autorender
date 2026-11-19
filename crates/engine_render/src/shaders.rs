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
            crate::shared::generated(crate::drawn::POSES_MODULE),
            crate::shared::generated(crate::drawn::DRAWN_POSE_MODULE),
        ];
    };
}

shader_modules! {
    "engine::fullscreen" = "fullscreen.wgsl";
    "engine::color" = "color.wgsl";
    "engine::math" = "math.wgsl";
    "engine::view" = "frame/view.wgsl";
    "engine::post::frame" = "effects.wgsl";
}

pub static CORE: Source = Source::generated(crate::linker::CORE_PACKAGE, false, TEXTS);

pub const FULLSCREEN: engine_core::Entry =
    engine_core::Entry::generated(Module::generated(&CORE, 0), "vs_fullscreen");

pub(crate) fn core_module(path: &str) -> Module {
    let index = TEXTS.iter().position(|text| text.path() == path);
    Module::generated(
        &CORE,
        index.unwrap_or_else(|| panic!("core_module_listed {path}")) as u32,
    )
}

pub const VIEW_MODULE: &str = "engine::view";

pub fn view_source() -> String {
    use engine_core::GpuStruct;
    let mut source = String::from(include_str!("frame/view.wgsl"));
    source.push_str(<crate::CameraUniform as GpuStruct>::WGSL);
    source
}

pub fn generated_core() -> [(&'static str, String); 4] {
    [
        (VIEW_MODULE, view_source()),
        (crate::drawn::DRAWN_MODULE, crate::drawn::source()),
        (crate::drawn::POSES_MODULE, engine_place::poses_wgsl()),
        (crate::drawn::DRAWN_POSE_MODULE, crate::drawn::pose_source()),
    ]
}
