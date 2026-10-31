use std::fmt::Write as _;

use engine_core::{Absent, BrickRegistry, Face, List, Place};

use crate::faces::face_module;

pub const COLOR_MODULE: &str = "engine::color";
pub const COMPOSITE_ENTRY: &str = "fs_composite";
pub const PLAIN_RESOLVE: &str = "post::resolve_plain";
pub const TRANSPARENT_COMPOSITE: &str = "transparent::composite";

const fn post_list(contract: &'static str) -> Face {
    Face {
        cpu: None,
        gpu: None,
        wgsl: Some(contract),
        absent: Absent::Nobody,
    }
}

pub struct PostScene;

impl List for PostScene {
    const NAME: &'static str = "post_scene";
    const FACE: Face = post_list(include_str!("places/post_scene.wgsl"));
}

pub struct PostLight;

impl List for PostLight {
    const NAME: &'static str = "post_light";
    const FACE: Face = post_list(include_str!("places/post_light.wgsl"));
}

pub struct PostDisplay;

impl List for PostDisplay {
    const NAME: &'static str = "post_display";
    const FACE: Face = post_list(include_str!("places/post_display.wgsl"));
}

pub static POST_PLACES: &[Place] = &[
    Place::list::<PostScene>(),
    Place::list::<PostLight>(),
    Place::list::<PostDisplay>(),
];

pub fn list_reads(registry: &BrickRegistry, lists: &[&str]) -> Vec<&'static str> {
    let mut reads: Vec<&'static str> = Vec::new();
    for list in lists {
        for (_, joined) in registry.joined(list) {
            for &read in joined.wgsl.iter().flat_map(|member| member.reads) {
                if !reads.contains(&read) {
                    reads.push(read);
                }
            }
        }
    }
    reads
}

pub fn post_reads(registry: &BrickRegistry) -> Vec<&'static str> {
    let lists: Vec<&str> = POST_PLACES.iter().map(|place| place.name).collect();
    list_reads(registry, &lists)
}

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

pub fn resolves(declared: impl IntoIterator<Item = &'static str>) -> Vec<&'static str> {
    let mut resolves = vec![PLAIN_RESOLVE];
    for resolve in declared {
        if !resolves.contains(&resolve) {
            resolves.push(resolve);
        }
    }
    resolves
}

pub fn composites(resolves: &[&'static str]) -> Vec<(String, &'static str, String)> {
    let mut made: Vec<(String, &'static str, String)> = resolves
        .iter()
        .map(|resolve| {
            (
                composite_path(resolve),
                crate::parts::CORE_BRICK,
                composite_source(resolve),
            )
        })
        .collect();
    made.push((
        composite_path(TRANSPARENT_COMPOSITE),
        crate::parts::CORE_BRICK,
        transparent_source(),
    ));
    made
}

fn folded<L: List>(imports: &mut String) {
    let _ = writeln!(
        imports,
        "import package::{}::apply as {};",
        face_module(L::NAME),
        L::NAME
    );
}

pub fn composite_source(resolve: &str) -> String {
    let mut imports = String::new();
    let mut body = String::new();
    let _ = writeln!(
        imports,
        "import package::{resolve}::{{resolve, output, Outputs}};"
    );
    let _ = writeln!(imports, "import package::{COLOR_MODULE}::encode_srgb;");
    folded::<PostLight>(&mut imports);
    folded::<PostDisplay>(&mut imports);
    let _ = writeln!(body, "    let resolved = resolve(position);");
    let _ = writeln!(body, "    var color = resolved.color;");
    let _ = writeln!(body, "    color = {}(color, resolved.at);", PostLight::NAME);
    let _ = writeln!(body, "    var encoded = encode_srgb(color);");
    let _ = writeln!(
        body,
        "    encoded = {}(encoded, resolved.at);",
        PostDisplay::NAME
    );
    let _ = writeln!(body, "    return output(resolved, encoded);");
    format!(
        "{imports}\n@fragment\nfn {COMPOSITE_ENTRY}(@builtin(position) position: vec4<f32>) -> Outputs {{\n{body}}}\n"
    )
}

pub fn transparent_source() -> String {
    let mut imports = String::new();
    let mut body = String::new();
    let _ = writeln!(
        imports,
        "import package::{TRANSPARENT_COMPOSITE}::{{output, weighted}};"
    );
    let _ = writeln!(imports, "import package::{COLOR_MODULE}::encode_srgb;");
    folded::<PostDisplay>(&mut imports);
    let _ = writeln!(body, "    let drawn = weighted(position);");
    let _ = writeln!(
        body,
        "    if drawn.reveal >= 1.0 {{\n        discard;\n    }}"
    );
    let _ = writeln!(body, "    let encoded = encode_srgb(drawn.color);");
    let _ = writeln!(body, "    var shown = encoded;");
    let _ = writeln!(body, "    shown = {}(shown, drawn.at);", PostDisplay::NAME);
    let _ = writeln!(body, "    return output(drawn, encoded, shown);");
    format!(
        "{imports}\n@fragment\nfn {COMPOSITE_ENTRY}(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {{\n{body}}}\n"
    )
}
