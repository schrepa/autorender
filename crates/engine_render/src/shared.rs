use std::fmt::Write as _;

use engine_core::{Error, PlaceDecl, ShaderText, Source};

use crate::install::Work;
use crate::machine::OutputDecl;
use crate::{
    DebugModes, DebugShown, Draws, GpuPart, Install, Layer, MainView, Pass, PostDisplay, PostLight,
    PostScene, Prepare, Resolve, ViewOutputs,
};

pub(crate) const COMPOSITE: &str = "composite";
pub(crate) const OUTPUT_PREFIX: &str = "output_";
pub(crate) const POST: &str = "post";
pub(crate) const LINEAR: &str = "linear";
pub(crate) const SCENE: &str = "scene";
pub(crate) const GLOW: &str = "glow";
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
    PlaceDecl::list::<PostDisplay>(),
    PlaceDecl::slot::<DebugShown>(),
    PlaceDecl::list::<DebugModes>(),
];

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
        self.draws.draw(&mut draw, layer, &())
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
}

impl GpuPart for Composite {
    type Reads = ();

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let pass = cx.pass(COMPOSITE).id();
        let pipeline = cx.render_pipeline(
            pass,
            &crate::RenderPipeline {
                vertex: crate::shaders::FULLSCREEN,
                fragment: Some(crate::shaders::FS_COMPOSITE),
                debug: None,
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: crate::DepthUse::None,
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &[],
        )?;
        Ok(Self { pass, pipeline })
    }

    fn prepare(&mut self, (): (), cx: &mut Prepare<'_, Self>) {
        if !cx.core.debug {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut crate::Encode<'_, Self>) -> u32 {
        let mut draw = cx.render(crate::Attach::Declared, None);
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
        inverse_view_projection: crate::renderer::inverse_or_identity(view_projection).to_cols_array_2d(),
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
        "import package::engine::shading::{Shaded, ViewPixel};\nimport package::engine::post::pixel::PostPixel;\n",
    );
    let mut body = String::new();
    for (index, output) in outputs.iter().enumerate() {
        if let Some(module) = &output.module {
            let _ = writeln!(imports, "import package::{module}::{} as member_{index};", output.name);
        }
        let _ = writeln!(
            body,
            "@group(0) @binding(0) var {OUTPUT_PREFIX}{}: {};",
            output.name,
            sampled_type(output.format)
        );
    }
    let _ = writeln!(body, "\nstruct ViewTargets {{\n    @location(0) scene: vec4<f32>,\n    @location(1) glow: vec2<f32>,");
    for (index, output) in outputs.iter().enumerate() {
        let _ = writeln!(body, "    @location({}) {}: {},", index + 2, output.name, output_type(output.format));
    }
    let _ = writeln!(body, "}}\n\nfn view_targets(shaded: Shaded, at: ViewPixel) -> ViewTargets {{");
    let _ = writeln!(body, "    var targets: ViewTargets;\n    targets.scene = vec4<f32>(shaded.color, shaded.ambient);\n    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);");
    for (index, output) in outputs.iter().enumerate() {
        if !output.layers.is_empty() && output.module.is_some() {
            let _ = writeln!(body, "    targets.{} = member_{index}(shaded, at);", output.name);
        }
    }
    let _ = writeln!(body, "    return targets;\n}}\n\nstruct CompositeOut {{\n    @location(0) canvas: vec4<f32>,");
    let mut location = 1;
    for output in outputs.iter().filter(|output| output.layers.is_empty()) {
        let _ = writeln!(body, "    @location({location}) {}: {},", output.name, output_type(output.format));
        location += 1;
    }
    let _ = writeln!(body, "}}\n\nfn composite_out(color: vec4<f32>, at: PostPixel) -> CompositeOut {{\n    var out: CompositeOut;\n    out.canvas = color;");
    for (index, output) in outputs.iter().enumerate() {
        if output.layers.is_empty() && output.module.is_some() {
            let _ = writeln!(body, "    out.{} = member_{index}(color, at);", output.name);
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
import package::face::resolve::resolve;
import package::face::post_light::apply as post_light;
import package::face::post_display::apply as post_display;
import package::generated::targets::{CompositeOut, composite_out};

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> CompositeOut {
    let started = scene_light(position);
    var light = post_scene(started.light, started.at);
    light = resolve(started.at, light);
    let color = post_light(light.color, started.at);
    var encoded = encode_srgb(color);
    encoded = post_display(encoded, started.at);
    return composite_out(vec4<f32>(encoded, 1.0), started.at);
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
