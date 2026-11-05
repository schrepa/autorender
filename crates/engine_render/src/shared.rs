use std::fmt::Write as _;

use bevy_ecs::prelude::Res;
use bevy_ecs::system::SystemParamItem;
use engine_core::{Error, PlaceDecl, ShaderText, Source};

use crate::install::Work;
use crate::machine::OutputDecl;
use crate::{
    DebugModes, DebugShown, Draws, GpuPart, Install, Layer, MainView, Overlay, OverlayStack,
    OverlaySteps, Pass, PostDisplay, PostLight, PostScene, PostTone, Prepare, Resolve, ViewOutputs,
};

pub(crate) const COMPOSITE: &str = "composite";
pub(crate) const DEBUG_VIEW: &str = "debug_view";
pub(crate) const OVERLAY: &str = "overlay";
pub(crate) const DEBUG_TARGETS: [&str; 3] = ["debug_ids", "debug_count", "debug_terrain"];
pub(crate) const OUTPUT_PREFIX: &str = "output_";
pub(crate) const POST: &str = "post";
pub(crate) const LINEAR: &str = "linear";
pub(crate) const SCENE: &str = "scene";
pub(crate) const DEPTH: &str = "depth_texture";
pub(crate) const TARGETS_MODULE: &str = "generated::targets";
pub(crate) const COMPOSITE_MODULE: &str = "generated::composite";
pub(crate) const LAYERS: [Layer; 7] = [
    Layer::Occluders,
    Layer::Opaque,
    Layer::Late,
    Layer::Backdrop,
    Layer::Media,
    Layer::Over,
    Layer::Blend,
];

pub(crate) static PLACES: &[PlaceDecl] = &[
    PlaceDecl::list::<ViewOutputs>(),
    PlaceDecl::slot::<Resolve>(),
    PlaceDecl::list::<PostScene>(),
    PlaceDecl::list::<PostLight>(),
    PlaceDecl::slot::<PostTone>(),
    PlaceDecl::list::<PostDisplay>(),
    PlaceDecl::slot::<DebugShown>(),
    PlaceDecl::list::<DebugModes>(),
    PlaceDecl::slot::<OverlaySteps>(),
];

pub(crate) fn draws_canvas(pass: &str) -> bool {
    pass == COMPOSITE || pass == DEBUG_VIEW || pass == OVERLAY
}

pub(crate) struct Layers {
    passes: Vec<(Layer, Pass<Self>)>,
    draws: Draws<MainView>,
}

impl GpuPart for Layers {
    type Reads = ();

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let members: Vec<(u32, Layer)> = cx
            .core
            .members::<crate::registration::DrawRecord<MainView>>()
            .map(|(part, record)| (part, record.layer))
            .collect();
        let mut passes = Vec::new();
        for layer in LAYERS {
            let drawn: Vec<u32> = members
                .iter()
                .filter(|(_, at)| *at == layer)
                .map(|(part, _)| *part)
                .collect();
            if drawn.is_empty() {
                continue;
            }
            let pass = cx.core.pass(layer_name(layer), Work::Shared(layer));
            if let Some(declared) = cx.core.pass_mut(pass) {
                declared.members = drawn;
            }
            passes.push((layer, Pass(pass, std::marker::PhantomData)));
        }
        let Some(&(_, first)) = passes.first() else {
            return Ok(Self {
                passes,
                draws: Draws {
                    members: Box::new([]),
                },
            });
        };
        let draws = cx.claim::<MainView>(first)?;
        for (_, pass) in passes.iter().skip(1) {
            if let Some(declared) = cx.core.pass_mut(pass.0) {
                declared.claims.push(std::any::TypeId::of::<MainView>());
            }
        }
        Ok(Self { passes, draws })
    }

    fn prepare(&mut self, (): (), cx: &mut Prepare<'_, Self>) {
        for (_, pass) in &self.passes {
            cx.run(*pass);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut crate::Encode<'_, Self>) -> u32 {
        let Some(&(layer, _)) = self.passes.iter().find(|(_, known)| *known == pass) else {
            return 0;
        };
        let mut draw = cx.render(crate::Attach::Declared, None);
        self.draws.draw::<Self>(&mut draw, layer, &())
    }
}

fn layer_name(layer: Layer) -> &'static str {
    match layer {
        Layer::Occluders => "occluders",
        Layer::Opaque => "opaque",
        Layer::Late => "late",
        Layer::Backdrop => "backdrop",
        Layer::Media => "media",
        Layer::Over => "over",
        Layer::Blend => "blend",
    }
}

pub(crate) struct Composite {
    pass: Pass<Self>,
    pipeline: crate::Pipeline<Self, crate::Own>,
    plain: crate::Pipeline<Self, crate::Own>,
    debug_pass: Pass<Self>,
    debug_pipeline: crate::Pipeline<Self, crate::Own>,
    overlay: Pass<Self>,
    overlay_draws: Draws<Overlay>,
    stack_draws: Draws<OverlayStack>,
    stack_height: u32,
}

impl Composite {
    fn stack_step(&self, draw: &mut crate::Draw<'_, Self, crate::Own>, step: Option<u32>) -> u32 {
        LAYERS
            .into_iter()
            .map(|layer| self.stack_draws.draw::<Self>(draw, layer, &step))
            .sum()
    }
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
    type Reads = Res<'static, OverlaySteps>;
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
        let plain =
            cx.render_pipeline(pass, &fullscreen(crate::shaders::FS_COMPOSITE_PLAIN), &[])?;
        let debug_pass = cx.pass(DEBUG_VIEW).after(pass).id();
        let debug_pipeline =
            cx.render_pipeline(debug_pass, &fullscreen(crate::shaders::FS_DEBUG_VIEW), &[])?;
        let overlay = cx
            .pass(OVERLAY)
            .label("engine/post-overlay")
            .after(pass)
            .after(debug_pass)
            .id();
        let overlay_draws = cx.claim::<Overlay>(overlay)?;
        let stack_draws = cx.claim::<OverlayStack>(overlay)?;
        Ok(Self {
            pass,
            pipeline,
            plain,
            debug_pass,
            debug_pipeline,
            overlay,
            overlay_draws,
            stack_draws,
            stack_height: 0,
        })
    }

    fn prepare(&mut self, stack: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        self.stack_height = stack.count;
        if cx.core.debug {
            cx.run(self.debug_pass);
        } else {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut crate::Encode<'_, Self>) -> u32 {
        if pass == self.overlay {
            let mut draw = cx.render(crate::Attach::Declared, None);
            let mut drawn: u32 = (0..self.stack_height)
                .map(|step| self.stack_step(&mut draw, Some(step)))
                .sum();
            drawn += LAYERS
                .into_iter()
                .map(|layer| self.overlay_draws.draw::<Self>(&mut draw, layer, &()))
                .sum::<u32>();
            return drawn + self.stack_step(&mut draw, None);
        }
        let pipeline = if pass == self.debug_pass {
            &self.debug_pipeline
        } else if cx.core.temporal {
            &self.pipeline
        } else {
            &self.plain
        };
        let mut draw = cx.render(crate::Attach::Declared, None);
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
    pub(crate) previous_view_projection: [[f32; 4]; 4],
    pub(crate) previous_eye: [f32; 4],
    pub(crate) detail: [f32; 4],
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
        previous_view_projection: view_projection.to_cols_array_2d(),
        previous_eye: position.extend(0.0).to_array(),
        detail: detail.extend(0.0).to_array(),
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
        "import package::engine::shading::{Shaded, ViewPixel};\nimport package::engine::post::pixel::{PostPixel, SceneLight};\n",
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
        for (index, output) in outputs
            .iter()
            .filter(|output| !output.layers.is_empty())
            .enumerate()
        {
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
        "\nfn view_targets(shaded: Shaded, at: ViewPixel) -> ViewTargets {{"
    );
    let _ = writeln!(
        body,
        "    var targets: ViewTargets;\n    targets.color = vec4<f32>(shaded.linear, shaded.ambient);"
    );
    for (index, output) in outputs.iter().enumerate() {
        if !output.layers.is_empty() && output.module.is_some() {
            let _ = writeln!(
                body,
                "    targets.{} = member_{index}(shaded, at);",
                output.name
            );
        }
    }
    let _ = writeln!(
        body,
        "    return targets;\n}}\n\nfn depth_view_targets(shaded: Shaded, at: ViewPixel, depth: f32) -> DepthViewTargets {{"
    );
    let _ = writeln!(
        body,
        "    let targets = view_targets(shaded, at);\n    var out: DepthViewTargets;\n    out.color = targets.color;"
    );
    for output in outputs.iter().filter(|output| !output.layers.is_empty()) {
        let _ = writeln!(body, "    out.{name} = targets.{name};", name = output.name);
    }
    let _ = writeln!(
        body,
        "    out.depth = depth;\n    return out;\n}}\n\nstruct CompositeOut {{\n    @location(0) canvas: vec4<f32>,"
    );
    for (location, output) in (1..).zip(outputs.iter().filter(|output| output.layers.is_empty())) {
        let _ = writeln!(
            body,
            "    @location({location}) {}: {},",
            output.name,
            output_type(output.format)
        );
    }
    let _ = writeln!(
        body,
        "}}\n\nfn composite_out(color: vec4<f32>, light: SceneLight, at: PostPixel) -> CompositeOut {{\n    var out: CompositeOut;\n    out.canvas = color;"
    );
    for (index, output) in outputs.iter().enumerate() {
        if output.layers.is_empty() && output.module.is_some() {
            let _ = writeln!(body, "    out.{} = member_{index}(light, at);", output.name);
        }
    }
    let _ = writeln!(body, "    return out;\n}}");
    format!("{imports}\n{body}")
}

pub(crate) fn composite_source() -> String {
    "import package::engine::post::pixel::{PostPixel, SceneLight};
import package::engine::post::resolve_plain::scene_light;
import package::engine::color::encode_srgb;
import package::face::post_scene::apply as post_scene;
import package::face::resolve::{resolve, resolved_at};
import package::face::post_light::apply as post_light;
import package::face::post_tone::apply as post_tone;
import package::face::post_display::apply as post_display;
import package::generated::targets::{CompositeOut, composite_out};
import package::face::debug::{DebugShade, DebugTargets, shown};
import package::face::debug_modes::shade_kind;

@group(0) @binding(0) var debug_ids: texture_2d<u32>;
@group(0) @binding(0) var debug_count: texture_2d<f32>;
@group(0) @binding(0) var debug_terrain: texture_2d<u32>;

@fragment
fn fs_debug_view(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    var targets: DebugTargets;
    targets.ids = textureLoad(debug_ids, pixel, 0);
    targets.count = textureLoad(debug_count, pixel, 0).r;
    targets.terrain = textureLoad(debug_terrain, pixel, 0).xy;
    var shade: DebugShade;
    return shown(shade_kind(shade, targets), targets);
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> CompositeOut {
    let started = scene_light(position);
    let light = resolve(started.at, started.light);
    let at = resolved_at(started.at);
    let color = post_tone(post_light(light.color, at), at);
    var encoded = encode_srgb(color);
    encoded = post_display(encoded, at);
    return composite_out(vec4<f32>(encoded, 1.0), light, at);
}

@fragment
fn fs_composite_plain(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let started = scene_light(position);
    let light = post_scene(started.light, started.at);
    let color = post_tone(post_light(light.color, started.at), started.at);
    var encoded = encode_srgb(color);
    encoded = post_display(encoded, started.at);
    return vec4<f32>(encoded, 1.0);
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
