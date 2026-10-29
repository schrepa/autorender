use std::fmt::Write as _;

use engine_core::{Error, ErrorCode, Expected};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PostStage {
    Scene,
    Light,
    Display,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PixelEffect {
    pub stage: PostStage,
    pub module: &'static str,
}

pub(crate) const PIXEL_MODULE: &str = "post::pixel";
pub(crate) const COLOR_MODULE: &str = "engine::color";
pub(crate) const COMPOSITE_ENTRY: &str = "fs_composite";

pub(crate) fn composite_path(resolve: &str) -> String {
    format!("generated::composite::{}", resolve.replace("::", "_"))
}

pub(crate) fn composite_source(
    resolve: &str,
    effects: &[PixelEffect],
    tonemap: Option<&str>,
) -> Result<String, Error> {
    if let Some(scene) = effects
        .iter()
        .find(|effect| effect.stage == PostStage::Scene)
    {
        return Err(Error::new(
            ErrorCode::UnsupportedFeature,
            format!("$.effects.{}", scene.module),
        )
        .expected(Expected::Supported)
        .got("scene"));
    }
    let mut imports = String::new();
    let mut body = String::new();
    let _ = writeln!(
        imports,
        "import package::{resolve}::{{resolve, output, Outputs}};"
    );
    let _ = writeln!(imports, "import package::{COLOR_MODULE}::encode_srgb;");
    let _ = writeln!(body, "    let resolved = resolve(position);");
    let _ = writeln!(body, "    var color = resolved.color;");
    let lights = effects
        .iter()
        .filter(|effect| effect.stage == PostStage::Light);
    for (index, effect) in lights.enumerate() {
        let _ = writeln!(
            imports,
            "import package::{}::apply as light_{index};",
            effect.module
        );
        let _ = writeln!(body, "    color = light_{index}(color, resolved.at);");
    }
    if let Some(tonemap) = tonemap {
        let _ = writeln!(imports, "import package::{tonemap}::apply as tonemap;");
        let _ = writeln!(body, "    color = tonemap(color, resolved.at);");
    }
    let _ = writeln!(body, "    var encoded = encode_srgb(color);");
    let displays = effects
        .iter()
        .filter(|effect| effect.stage == PostStage::Display);
    for (index, effect) in displays.enumerate() {
        let _ = writeln!(
            imports,
            "import package::{}::apply as display_{index};",
            effect.module
        );
        let _ = writeln!(body, "    encoded = display_{index}(encoded, resolved.at);");
    }
    let _ = writeln!(body, "    return output(resolved, encoded);");
    Ok(format!(
        "{imports}\n@fragment\nfn {COMPOSITE_ENTRY}(@builtin(position) position: vec4<f32>) -> Outputs {{\n{body}}}\n"
    ))
}
