use std::fmt::Write as _;

use engine_core::{AttachPoint, Error, ErrorCode, PassViews, PlaceDecl, ShaderText, Source};

use crate::install::{PassDecl, Work};
use crate::machine::OutputDecl;
use crate::{DebugModes, DebugShown, GpuPart, Install, Lit, Pass, Prepare, Resolve};

pub(crate) const COMPOSITE: &str = "composite";
pub(crate) const POST: &str = "post";
pub(crate) const LINEAR: &str = "linear";
pub(crate) const SCENE: &str = "scene";
pub(crate) const DEPTH: &str = "depth_texture";
pub(crate) const WRITES_MODULE: &str = "generated::writes";
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
            | AttachPoint::LateOpaque
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

const FOLDS: [(Stage, &str); 2] = [(Stage::Color, "post"), (Stage::Display, "display")];

fn steps(order: &engine_core::FoldOrder, stage: Stage) -> &[String] {
    match stage {
        Stage::Color => &order.post,
        Stage::Display => &order.display,
    }
}

pub(crate) fn ordered(
    applied: Vec<AppliedModule>,
    order: Option<&engine_core::FoldOrder>,
) -> Vec<AppliedModule> {
    let Some(order) = order else {
        return applied;
    };
    let mut left: Vec<Option<AppliedModule>> = applied.into_iter().map(Some).collect();
    let mut sorted = Vec::with_capacity(left.len());
    for (stage, _) in FOLDS {
        for step in steps(order, stage) {
            let found = left.iter_mut().find(|member| {
                member
                    .as_ref()
                    .is_some_and(|member| member.stage == stage && member.pass == step.as_str())
            });
            if let Some(member) = found.and_then(Option::take) {
                sorted.push(member);
            }
        }
    }
    sorted.extend(left.into_iter().flatten());
    sorted
}

pub(crate) fn unknown_steps(applied: &[AppliedModule], order: &engine_core::FoldOrder) -> Vec<Error> {
    let mut unknown = Vec::new();
    for (stage, field) in FOLDS {
        let known: Vec<&'static str> = applied
            .iter()
            .filter(|member| member.stage == stage)
            .map(|member| member.pass)
            .collect();
        for (at, step) in steps(order, stage).iter().enumerate() {
            if known.contains(&step.as_str()) && !steps(order, stage)[..at].contains(step) {
                continue;
            }
            unknown.push(
                Error::new(
                    ErrorCode::UnknownVariant,
                    format!(
                        "$.settings.{}.{field}[{at}]",
                        <engine_core::FoldOrder as engine_core::Section>::KEY
                    ),
                )
                .got(step)
                .variants(known.iter().copied()),
            );
        }
    }
    unknown
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
    pass.name == COMPOSITE
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
                    .any(|member| member.kind.joins(listed.name, &listed.pass))
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
}

fn fullscreen(fragment: engine_core::Entry) -> crate::RenderPipeline<'static> {
    crate::RenderPipeline {
        vertex: crate::shaders::FULLSCREEN,
        fragment: Some(fragment),
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
        Ok(Self { pass, pipeline })
    }

    fn prepare(&mut self, (): (), _cx: &mut Prepare<'_, Self>) {}

    fn view(&mut self, _view: &engine_core::ViewFacts, cx: &mut Prepare<'_, Self>) {
        cx.run(self.pass);
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut crate::Encode<'_, Self>) -> u32 {
        let mut draw = cx.render(crate::Attach::Declared);
        draw.draw(
            &self.pipeline,
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
    pub(crate) kept: [u32; 4],
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
        kept: [0; 4],
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

pub(crate) fn pass_module(colored: bool, written: &[&OutputDecl]) -> (String, String) {
    let mut key = String::new();
    let mut imports =
        String::from("import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};\n");
    let (mut fields, mut filled) = (String::new(), String::new());
    let (mut copied, mut blended) = (String::new(), String::new());
    if colored {
        key.push_str(engine_core::TargetName::COLOR.name());
        fields.push_str("    @location(0) color: vec4<f32>,\n");
        filled.push_str("    written.color = vec4<f32>(shaded.linear, 1.0);\n");
        copied.push_str("    out.color = written.color;\n");
        blended.push_str(
            "    written.color = vec4<f32>(shaded.linear * shaded.opacity, shaded.opacity);\n",
        );
    }
    for (location, output) in (u32::from(colored)..).zip(written) {
        let name = output.name;
        key.push('+');
        key.push_str(name);
        let _ = writeln!(
            fields,
            "    @location({location}) {name}: {},",
            output_type(output.format)
        );
        let _ = writeln!(copied, "    out.{name} = written.{name};");
        if let Some(module) = &output.module {
            let _ = writeln!(imports, "import package::{module}::{name} as written_{location};");
            let _ = writeln!(filled, "    written.{name} = written_{location}(point, shaded, at);");
        }
    }
    let text = format!(
        "{imports}\nstruct Outputs {{\n{fields}}}\n\nstruct DepthOutputs {{\n{fields}    @builtin(frag_depth) depth: f32,\n}}\n\nfn outputs(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> Outputs {{\n    var written: Outputs;\n{filled}    return written;\n}}\n\nfn blended_outputs(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> Outputs {{\n    var written = outputs(point, shaded, at);\n{blended}    return written;\n}}\n\nfn depth_outputs(point: SurfacePoint, shaded: Shaded, at: ViewPixel, depth: f32) -> DepthOutputs {{\n    let written = outputs(point, shaded, at);\n    var out: DepthOutputs;\n{copied}    out.depth = depth;\n    return out;\n}}\n"
    );
    (key, text)
}

pub(crate) fn composite_source() -> String {
    "import package::engine::post::frame::{depth_texture, post};
import package::engine::post::resolve_plain::{post_pixel, scene_color};
import package::engine::color::encode_srgb;
import package::generated::applied::{applied_color, applied_display};
import package::face::resolve::resolved;

fn uncovered(pixel: vec2<i32>) -> bool {
    return post.kept.x != 0u && textureLoad(depth_texture, pixel, 0) <= 0.0;
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
    ShaderText::generated(path, "", "")
}
