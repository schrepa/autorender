use std::fmt::Write as _;

use engine_core::{AttachPoint, Error, ErrorCode, PassViews, PlaceDecl, ShaderText, Source};

use crate::install::{PassDecl, Work};
use crate::machine::OutputDecl;
use crate::{DebugModes, DebugShown, GpuPart, Install, Lit, Pass, Prepare, Resolve};

pub(crate) const COMPOSITE: &str = "composite";
pub(crate) const DEBUG_VIEW: &str = "debug_view";
pub(crate) const OUTPUT_PREFIX: &str = "output_";
pub(crate) const POST: &str = "post";
pub(crate) const LINEAR: &str = "linear";
pub(crate) const SCENE: &str = "scene";
pub(crate) const DEPTH: &str = "depth_texture";
pub(crate) const TARGETS_MODULE: &str = "generated::targets";
pub(crate) const COMPOSITE_MODULE: &str = "generated::composite";
pub(crate) const APPLIED_MODULE: &str = "generated::applied";
pub(crate) const PASSES_MODULE: &str = "generated::passes";
pub(crate) const KINDS_MODULE: &str = "generated::kinds";
pub(crate) const TRANSPARENCY_MODULE: &str = "engine::transparency";
pub(crate) const COVER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub(crate) const MOST_APPLIED: usize = 128;
pub(crate) static PLACES: &[PlaceDecl] = &[
    PlaceDecl::slot::<Resolve>(),
    PlaceDecl::slot::<Lit>(),
    PlaceDecl::slot::<DebugShown>(),
    PlaceDecl::list::<DebugModes>(),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Color,
    Display,
}

#[derive(Clone, Debug)]
pub(crate) struct AppliedModule {
    pub(crate) pass: &'static str,
    pub(crate) views: PassViews,
    pub(crate) gate: Option<engine_registry::GateId>,
    pub(crate) module: String,
    pub(crate) stage: Stage,
}

pub(crate) fn stage_of(
    at: AttachPoint,
    module: &str,
    text: &str,
) -> Result<Stage, Error> {
    let path = format!("$.applied.{module}");
    let contract = crate::places::contract(text)
        .map_err(|error| Error::new(ErrorCode::ShaderLink, path.clone()).got(error))?;
    let first = contract
        .answers
        .iter()
        .find(|answer| answer.name == "apply" && !answer.constant)
        .and_then(|answer| answer.parameters.first())
        .map(|(_, ty)| ty.replace(' ', ""));
    match (at, first.as_deref()) {
        (AttachPoint::Post, Some("vec3<f32>")) => Ok(Stage::Color),
        (AttachPoint::Display, Some("vec3<f32>")) => Ok(Stage::Display),
        (
            AttachPoint::Depth
            | AttachPoint::Inputs
            | AttachPoint::Opaque
            | AttachPoint::Sky
            | AttachPoint::Transparent
            | AttachPoint::Post
            | AttachPoint::Display
            | AttachPoint::Overlay,
            _,
        ) => Err(Error::new(ErrorCode::ShaderLink, path)
            .expected(engine_core::Expected::Supported)
            .got(format!("{}: {}", at.name(), first.unwrap_or_default()))),
    }
}

pub(crate) fn applied_source(applied: &[AppliedModule]) -> String {
    let mut imports = String::from(
        "import package::engine::post::frame::post;\nimport package::engine::post::pixel::PostPixel;\n",
    );
    let mut bodies = [String::new(), String::new()];
    for (bit, member) in applied.iter().enumerate() {
        let _ = writeln!(
            imports,
            "import package::{}::apply as apply_{bit};",
            member.module
        );
        let word = ["x", "y", "z", "w"][bit / 32 % 4];
        let mask = 1_u32 << (bit % 32);
        let body = match member.stage {
            Stage::Color => &mut bodies[0],
            Stage::Display => &mut bodies[1],
        };
        let _ = writeln!(
            body,
            "    if (post.applied.{word} & {mask}u) != 0u {{\n        folded = apply_{bit}(folded, at);\n    }}"
        );
    }
    let [color, display] = bodies;
    format!(
        "{imports}\nfn applied_color(color: vec3<f32>, at: PostPixel) -> vec3<f32> {{\n    var folded = color;\n{color}    return folded;\n}}\n\nfn applied_display(color: vec3<f32>, at: PostPixel) -> vec3<f32> {{\n    var folded = color;\n{display}    return folded;\n}}\n"
    )
}

pub(crate) fn composites(pass: &PassDecl) -> bool {
    pass.name == COMPOSITE || pass.name == DEBUG_VIEW
}

pub(crate) fn draws_canvas(pass: &PassDecl) -> bool {
    composites(pass) || pass.layer == Some(AttachPoint::Overlay)
}

pub(crate) struct ScenePart {
    passes: Vec<Pass<Self>>,
}

impl GpuPart for ScenePart {
    type Reads = ();
    const NAME: &'static str = "scene";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let listed: Vec<engine_core::ScenePass> = cx
            .core
            .passes
            .listed()
            .iter()
            .filter(|listed| listed.bit.is_some() && listed.pass.views != PassViews::Owned)
            .filter(|listed| {
                cx.core
                    .draws
                    .iter()
                    .any(|member| (listed.pass.takes)(member.kind))
            })
            .copied()
            .collect();
        let mut passes = Vec::with_capacity(listed.len());
        for listed in listed {
            let pass = cx.core.pass(listed.name, Work::Render);
            cx.core.draws_scene(pass, listed.name, listed.pass);
            passes.push(Pass(pass, std::marker::PhantomData));
        }
        Ok(Self { passes })
    }

    fn prepare(&mut self, (): (), cx: &mut Prepare<'_, Self>) {
        for pass in &self.passes {
            cx.run(*pass);
        }
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut crate::Encode<'_, Self>) -> u32 {
        cx.render(crate::Attach::Declared).scene(self)
    }
}

pub(crate) struct Composite {
    pass: Pass<Self>,
    pipeline: crate::Pipeline<Self, crate::Own>,
    debug_pass: Pass<Self>,
    debug_pipeline: crate::Pipeline<Self, crate::Own>,
}

fn fullscreen(fragment: engine_core::Entry) -> crate::RenderPipeline<'static> {
    crate::RenderPipeline {
        vertex: crate::shaders::FULLSCREEN,
        fragment: Some(fragment),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: crate::DepthUse::None,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}

impl GpuPart for Composite {
    type Reads = ();
    const NAME: &'static str = "post";
    const LAZY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/post-", "engine/post."];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let pass = cx
            .pass(COMPOSITE)
            .label("engine/post-composite")
            .span("post_composite")
            .id();
        let pipeline = cx.render_pipeline(pass, &fullscreen(crate::shaders::FS_COMPOSITE), &[])?;
        let debug_pass = cx.pass(DEBUG_VIEW).after(pass).id();
        let debug_pipeline =
            cx.render_pipeline(debug_pass, &fullscreen(crate::shaders::FS_DEBUG_VIEW), &[])?;
        Ok(Self {
            pass,
            pipeline,
            debug_pass,
            debug_pipeline,
        })
    }

    fn prepare(&mut self, (): (), _cx: &mut Prepare<'_, Self>) {}

    fn view(&mut self, view: &engine_core::ViewFacts, cx: &mut Prepare<'_, Self>) {
        if view.debug {
            cx.run(self.debug_pass);
        } else {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut crate::Encode<'_, Self>) -> u32 {
        let pipeline = if pass == self.debug_pass {
            &self.debug_pipeline
        } else {
            &self.pipeline
        };
        let mut draw = cx.render(crate::Attach::Declared);
        draw.draw(
            pipeline,
            crate::Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
        0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PostUniform {
    pub(crate) inverse_view_projection: [[f32; 4]; 4],
    pub(crate) view_projection: [[f32; 4]; 4],
    pub(crate) eye: [f32; 4],
    pub(crate) size: [f32; 4],
    pub(crate) extent: [f32; 4],
    pub(crate) detail: [f32; 4],
    pub(crate) kept: [f32; 4],
    pub(crate) applied: [u32; 4],
}

pub(crate) fn post_uniform(
    view_projection: glam::Mat4,
    position: glam::Vec3,
    detail: glam::Vec3,
    render: (u32, u32),
    display: (u32, u32),
) -> PostUniform {
    let (width, height) = (render.0 as f32, render.1 as f32);
    let (wide, tall) = (display.0 as f32, display.1 as f32);
    let focal = view_projection.row(1).truncate().length();
    PostUniform {
        inverse_view_projection: crate::renderer::inverse_or_identity(view_projection)
            .to_cols_array_2d(),
        view_projection: view_projection.to_cols_array_2d(),
        eye: position.extend(focal * height * 0.5).to_array(),
        size: [width, height, 1.0 / width, 1.0 / height],
        extent: [wide, tall, 1.0 / wide, 1.0 / tall],
        detail: detail.extend(0.0).to_array(),
        kept: [0.0; 4],
        applied: [0; 4],
    }
}

fn output_type(format: wgpu::TextureFormat) -> String {
    let channels = format.components();
    let scalar = match format.sample_type(None, None) {
        Some(wgpu::TextureSampleType::Uint) => "u32",
        Some(wgpu::TextureSampleType::Sint) => "i32",
        Some(_) | None => "f32",
    };
    match channels {
        1 => scalar.to_owned(),
        2 => format!("vec2<{scalar}>"),
        _ => format!("vec4<{scalar}>"),
    }
}

fn sampled_type(format: wgpu::TextureFormat) -> &'static str {
    match format.sample_type(None, None) {
        Some(wgpu::TextureSampleType::Uint) => "texture_2d<u32>",
        Some(wgpu::TextureSampleType::Sint) => "texture_2d<i32>",
        Some(_) | None => "texture_2d<f32>",
    }
}

pub(crate) fn targets_source(outputs: &[OutputDecl]) -> String {
    let mut imports = String::from(
        "import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};\n",
    );
    let mut body = String::new();
    for (index, output) in outputs.iter().enumerate() {
        if let Some(module) = &output.module {
            let _ = writeln!(
                imports,
                "import package::{module}::{} as member_{index};",
                output.name
            );
        }
        let _ = writeln!(
            body,
            "@group(0) @binding(0) var {OUTPUT_PREFIX}{}: {};",
            output.name,
            sampled_type(output.format)
        );
    }
    for depth in [false, true] {
        let name = if depth {
            "DepthViewTargets"
        } else {
            "ViewTargets"
        };
        let _ = writeln!(
            body,
            "\nstruct {name} {{\n    @location(0) color: vec4<f32>,"
        );
        for (index, output) in outputs.iter().enumerate() {
            let _ = writeln!(
                body,
                "    @location({}) {}: {},",
                index + 1,
                output.name,
                output_type(output.format)
            );
        }
        if depth {
            let _ = writeln!(body, "    @builtin(frag_depth) depth: f32,");
        }
        let _ = writeln!(body, "}}");
    }
    let _ = writeln!(
        body,
        "\nfn view_targets(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> ViewTargets {{"
    );
    let _ = writeln!(
        body,
        "    var targets: ViewTargets;\n    targets.color = vec4<f32>(shaded.linear, 1.0);"
    );
    for (index, output) in outputs.iter().enumerate() {
        if output.module.is_some() {
            let _ = writeln!(
                body,
                "    targets.{} = member_{index}(point, shaded, at);",
                output.name
            );
        }
    }
    let _ = writeln!(
        body,
        "    return targets;\n}}\n\nfn blended_targets(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> ViewTargets {{\n    var targets = view_targets(point, shaded, at);\n    targets.color = vec4<f32>(shaded.linear * shaded.opacity, shaded.opacity);\n    return targets;\n}}"
    );
    if outputs
        .iter()
        .any(|output| output.name == engine_core::TargetName::TRANSPARENCY.name())
    {
        let _ = writeln!(
            body,
            "\nfn blended_cover(pixel: vec2<i32>) -> f32 {{\n    let covered = min(pixel, vec2<i32>(textureDimensions({OUTPUT_PREFIX}transparency)) - 1);\n    return textureLoad({OUTPUT_PREFIX}transparency, covered, 0).r;\n}}"
        );
    }
    let _ = writeln!(
        body,
        "\nfn depth_view_targets(point: SurfacePoint, shaded: Shaded, at: ViewPixel, depth: f32) -> DepthViewTargets {{"
    );
    let _ = writeln!(
        body,
        "    let targets = view_targets(point, shaded, at);\n    var out: DepthViewTargets;\n    out.color = targets.color;"
    );
    for output in outputs {
        let _ = writeln!(body, "    out.{name} = targets.{name};", name = output.name);
    }
    let _ = writeln!(
        body,
        "    out.depth = depth;\n    return out;\n}}"
    );
    format!("{imports}\n{body}")
}

pub(crate) fn composite_source() -> String {
    "import package::engine::post::frame::{depth_texture, post};
import package::engine::post::resolve_plain::{post_pixel, scene_color};
import package::engine::color::encode_srgb;
import package::generated::applied::{applied_color, applied_display};
import package::face::resolve::resolved;
import package::face::debug::{DebugShade, DebugTargets, shown};
import package::face::debug_modes::shade_kind;

@group(0) @binding(0) var debug_ids: texture_2d<u32>;
@group(0) @binding(0) var debug_count: texture_2d<f32>;
@group(0) @binding(0) var debug_words: texture_2d<u32>;

fn uncovered(pixel: vec2<i32>) -> bool {
    return post.kept.x > 0.5 && textureLoad(depth_texture, pixel, 0) <= 0.0;
}

@fragment
fn fs_debug_view(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    var targets: DebugTargets;
    targets.ids = textureLoad(debug_ids, pixel, 0);
    targets.count = textureLoad(debug_count, pixel, 0).r;
    targets.words = textureLoad(debug_words, pixel, 0).xy;
    var shade: DebugShade;
    return shown(shade_kind(shade, targets), targets);
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let at = post_pixel(position);
    if uncovered(at.pixel) {
        discard;
    }
    var light = resolved(at);
    if light.a == 0.0 {
        light = scene_color(at);
    }
    let color = applied_display(applied_color(light.rgb, at), at);
    return vec4<f32>(encode_srgb(color), 1.0);
}
"
    .to_owned()
}

pub(crate) fn core_source() -> &'static Source {
    &crate::shaders::CORE
}

pub(crate) const fn generated(path: &'static str) -> ShaderText {
    ShaderText::generated(path, "")
}
