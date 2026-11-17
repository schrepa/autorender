use engine_core::{AttachPoint, Error, ShaderText, Source};

use crate::install::{PassDecl, Work};
use crate::{GpuPart, Install, Pass, Prepare};

pub(crate) const COMPOSITE: &str = "composite";
pub(crate) const POST: &str = "post";
pub(crate) const SCENE: &str = "scene";
pub(crate) const DEPTH: &str = "depth_texture";
pub(crate) const WRITES_MODULE: &str = "generated::writes";
pub(crate) const COMPOSITE_MODULE: &str = "generated::composite";
pub(crate) const PASSES_MODULE: &str = "generated::passes";
pub(crate) const KINDS_MODULE: &str = "generated::kinds";

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
            .filter(|listed| listed.bit.is_some())
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

impl GpuPart for Composite {
    type Reads = ();
    const NAME: &'static str = "post";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let pass = cx
            .pass(COMPOSITE)
            .label("engine/post-composite")
            .span("post_composite")
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &crate::RenderPipeline {
                vertex: crate::shaders::FULLSCREEN,
                fragment: Some(crate::shaders::FS_COMPOSITE),
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
                material: None,
            },
        );
        0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PostUniform {
    pub(crate) size: [f32; 4],
    pub(crate) kept: [u32; 4],
}

pub(crate) fn post_uniform(render: (u32, u32)) -> PostUniform {
    let (width, height) = (render.0 as f32, render.1 as f32);
    PostUniform {
        size: [width, height, 1.0 / width, 1.0 / height],
        kept: [0; 4],
    }
}

const SHADED: &str = "
fn point_shaded(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {
    return shade(point, pixel);
}

fn slot_shaded(input: SurfaceInput, point: SurfacePoint, pixel: vec2<f32>) -> Shaded {
    return slot_output(input, point, shade(point, pixel));
}
";

pub(crate) fn pass_module() -> (String, String) {
    let text = format!(
        "import package::engine::shading::{{Shaded, SurfaceInput, SurfacePoint}};\nimport package::engine::slots::{{shade, slot_output}};\n\nstruct Outputs {{\n    @location(0) color: vec4<f32>,\n}}\n\nfn outputs(point: SurfacePoint, shaded: Shaded) -> Outputs {{\n    var written: Outputs;\n    written.color = vec4<f32>(shaded.linear, 1.0);\n    return written;\n}}\n\nfn blended_outputs(point: SurfacePoint, shaded: Shaded) -> Outputs {{\n    var written = outputs(point, shaded);\n    written.color = vec4<f32>(shaded.linear * shaded.opacity, shaded.opacity);\n    return written;\n}}\n{SHADED}"
    );
    (engine_core::TargetName::COLOR.name().to_owned(), text)
}

pub(crate) fn composite_source() -> String {
    "import package::engine::post::frame::{depth_texture, post, scene};
import package::engine::color::encode_srgb;

fn uncovered(pixel: vec2<i32>) -> bool {
    return post.kept.x != 0u && textureLoad(depth_texture, pixel, 0) <= 0.0;
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = min(vec2<i32>(position.xy), vec2<i32>(post.size.xy) - 1);
    if uncovered(pixel) {
        discard;
    }
    return vec4<f32>(encode_srgb(textureLoad(scene, pixel, 0).rgb), 1.0);
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
