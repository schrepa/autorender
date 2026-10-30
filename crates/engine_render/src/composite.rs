use std::fmt::Write as _;

use engine_core::{BrickRegistry, Error, ErrorCode, Expected};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PostStage {
    Resolve,
    Scene,
    Light,
    Tonemap,
    Display,
}

#[derive(Clone, Copy, Debug)]
pub struct PixelEffect {
    pub stage: PostStage,
    pub module: &'static str,
    pub reads: &'static [&'static str],
}

pub const COLOR_MODULE: &str = "engine::color";
pub const COMPOSITE_ENTRY: &str = "fs_composite";
pub const PLAIN_RESOLVE: &str = "post::resolve_plain";
pub const TRANSPARENT_COMPOSITE: &str = "transparent::composite";
pub const SCENE_MODULE: &str = "generated::scene";
pub const SCENE_FUNCTION: &str = "scene_light";
pub const PIXEL_MODULE: &str = "post::pixel";

pub fn composite_path(resolve: &str) -> String {
    format!("generated::composite::{}", resolve.replace("::", "_"))
}

pub fn composite_entry(resolve: &str) -> crate::shaders::Entry {
    crate::shaders::Entry {
        module: Box::leak(composite_path(resolve).into_boxed_str()),
        name: COMPOSITE_ENTRY,
        stage: crate::shaders::Stage::Fragment,
    }
}

pub fn pixel_effects(parts: &crate::parts::Parts) -> Vec<PixelEffect> {
    (0..parts.len())
        .flat_map(|index| parts.decl(index).effects.iter().copied())
        .collect()
}

pub fn effects_of(registry: &BrickRegistry) -> Vec<PixelEffect> {
    crate::render_bricks::render_bricks(registry)
        .flat_map(|(_, brick)| brick.parts.iter())
        .flat_map(|part| part.decl.effects.iter().copied())
        .collect()
}

pub fn effect_reads(effects: &[PixelEffect]) -> impl Iterator<Item = &'static str> + '_ {
    effects
        .iter()
        .flat_map(|effect| effect.reads.iter().copied())
}

pub fn scene_source(effects: &[PixelEffect]) -> String {
    let mut imports = String::new();
    let mut body = String::new();
    let _ = writeln!(
        imports,
        "import package::{PIXEL_MODULE}::{{PostPixel, SceneLight}};"
    );
    let _ = writeln!(body, "    var lit = light;");
    let scenes = effects
        .iter()
        .filter(|effect| effect.stage == PostStage::Scene);
    for (index, effect) in scenes.enumerate() {
        let _ = writeln!(
            imports,
            "import package::{}::apply as scene_{index};",
            effect.module
        );
        let _ = writeln!(body, "    lit = scene_{index}(lit, at);");
    }
    let _ = writeln!(body, "    return lit;");
    format!(
        "{imports}\nfn {SCENE_FUNCTION}(light: SceneLight, at: PostPixel) -> SceneLight {{\n{body}}}\n"
    )
}

pub fn resolves(effects: &[PixelEffect]) -> impl Iterator<Item = &'static str> + '_ {
    std::iter::once(PLAIN_RESOLVE).chain(
        effects
            .iter()
            .filter(|effect| effect.stage == PostStage::Resolve)
            .map(|effect| effect.module),
    )
}

pub fn tonemap_effect(effects: &[PixelEffect]) -> Result<Option<&'static str>, Error> {
    let mut tonemaps = effects
        .iter()
        .filter(|effect| effect.stage == PostStage::Tonemap);
    let first = tonemaps.next();
    if let (Some(first), Some(second)) = (first, tonemaps.next()) {
        return Err(Error::new(ErrorCode::SlotFilledTwice, "$.effects.tonemap")
            .expected(Expected::Unique)
            .got(second.module)
            .nearest(first.module));
    }
    Ok(first.map(|effect| effect.module))
}

pub fn composites(effects: &[PixelEffect]) -> Result<Vec<(String, &'static str, String)>, Error> {
    tonemap_effect(effects)?;
    let mut made = Vec::new();
    for resolve in resolves(effects) {
        made.push((
            composite_path(resolve),
            crate::parts::CORE_BRICK,
            composite_source(resolve, effects, None),
        ));
    }
    made.push((
        composite_path(TRANSPARENT_COMPOSITE),
        crate::parts::CORE_BRICK,
        transparent_source(effects),
    ));
    Ok(made)
}

pub fn composite_source(resolve: &str, effects: &[PixelEffect], tonemap: Option<&str>) -> String {
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
    displays(effects, "encoded", "resolved.at", &mut imports, &mut body);
    let _ = writeln!(body, "    return output(resolved, encoded);");
    format!(
        "{imports}\n@fragment\nfn {COMPOSITE_ENTRY}(@builtin(position) position: vec4<f32>) -> Outputs {{\n{body}}}\n"
    )
}

pub fn transparent_source(effects: &[PixelEffect]) -> String {
    let mut imports = String::new();
    let mut body = String::new();
    let _ = writeln!(
        imports,
        "import package::{TRANSPARENT_COMPOSITE}::{{output, weighted}};"
    );
    let _ = writeln!(imports, "import package::{COLOR_MODULE}::encode_srgb;");
    let _ = writeln!(body, "    let drawn = weighted(position);");
    let _ = writeln!(
        body,
        "    if drawn.reveal >= 1.0 {{\n        discard;\n    }}"
    );
    let _ = writeln!(body, "    let encoded = encode_srgb(drawn.color);");
    let _ = writeln!(body, "    var shown = encoded;");
    displays(effects, "shown", "drawn.at", &mut imports, &mut body);
    let _ = writeln!(body, "    return output(drawn, encoded, shown);");
    format!(
        "{imports}\n@fragment\nfn {COMPOSITE_ENTRY}(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {{\n{body}}}\n"
    )
}

fn displays(
    effects: &[PixelEffect],
    color: &str,
    at: &str,
    imports: &mut String,
    body: &mut String,
) {
    let displays = effects
        .iter()
        .filter(|effect| effect.stage == PostStage::Display);
    for (index, effect) in displays.enumerate() {
        let _ = writeln!(
            imports,
            "import package::{}::apply as display_{index};",
            effect.module
        );
        let _ = writeln!(body, "    {color} = display_{index}({color}, {at});");
    }
}
