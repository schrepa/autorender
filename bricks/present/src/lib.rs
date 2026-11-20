#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

engine::brick!();

use std::marker::PhantomData;

use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{Error, ErrorCode, Filter, Sampler, ViewFacts, ViewSkip, Wrap};
use engine_render::{Access, Encode, FULLSCREEN, Own, PerBatch, SharedTarget, TargetClear};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const PRESENTED: TargetName = TargetName("presented");

pub trait Scale: Resource {
    fn fraction(&self) -> f32;
}

const PRESENT_PASS: &str = "present";

const COVER_PASS: &str = "cover";

const LINEAR_CLAMP: Sampler = Sampler {
    wrap_u: Wrap::ClampToEdge,
    wrap_v: Wrap::ClampToEdge,
    wrap_w: Wrap::ClampToEdge,
    mag_filter: Filter::Linear,
    min_filter: Filter::Linear,
    mipmap_filter: Some(Filter::Linear),
    compare: None,
    anisotropy: 1,
};

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Present {
    pub antialias: bool,
}

gpu_struct! {
    struct PresentFrame {
        antialias: [f32; 4],
    }
}

pub struct PresentBrick<S>(PhantomData<fn() -> S>);

impl<S: Scale> Brick for PresentBrick<S> {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Present>("present")
                    .schema(engine_core::schema_of::<Present>),
            )
            .part::<Presenting<S>>();
    }
}

struct Presenting<S> {
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    cover: engine_render::Pass<Self>,
    frame: Buffer<Self>,
    sent: Option<PresentFrame>,
    several: bool,
    skipped: Vec<Entity>,
    scale: PhantomData<fn() -> S>,
}

fn fullscreen(fragment: engine_core::Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: FULLSCREEN,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[(None, wgpu::ColorWrites::ALL)],
        constants: &[],
        per_batch: PerBatch::NONE,
    }
}

impl<S: Scale> GpuPart for Presenting<S> {
    type Reads = (
        Query<'static, 'static, &'static Present>,
        Query<'static, 'static, Entity, With<ViewSkip>>,
        Option<Res<'static, S>>,
    );
    const NAME: &'static str = "present";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::present::canvas;
        cx.shared_target(SharedTarget {
            name: PRESENTED,
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            clear: TargetClear::Color([0.0, 0.0, 0.0, 1.0]),
        });
        cx.frame(canvas::PRESENTED, Bind::Target(PRESENTED))?;
        let frame = cx.buffer("present_frame", Lifetime::Kept);
        let sampler = cx.sampler(LINEAR_CLAMP);
        let pass = cx
            .pass(PRESENT_PASS)
            .label("present")
            .span("present")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &fullscreen(canvas::FS_PRESENT),
            &[
                (canvas::PRESENT_FRAME, Bind::Buffer(frame)),
                (canvas::PRESENTED_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        let cover = cx
            .pass(COVER_PASS)
            .label("cover")
            .span("cover")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        Ok(Self {
            pass,
            pipeline,
            cover,
            frame,
            sent: None,
            several: false,
            skipped: Vec::new(),
            scale: PhantomData,
        })
    }

    fn prepare(
        &mut self,
        (presents, skipped, scale): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.skipped.clear();
        self.skipped.extend(&skipped);
        if let Some(scale) = scale {
            cx.scale_target(PRESENTED, scale.fraction());
        }
        let mut found = presents.iter();
        let antialias = found.next().is_some_and(|present| present.antialias);
        let several = found.next().is_some();
        if several && !self.several {
            cx.report(
                Error::new(ErrorCode::NameTwice, "$.entities.present").got(presents.iter().count()),
            );
        }
        self.several = several;
        let shown = PresentFrame {
            antialias: [f32::from(u8::from(antialias)), 0.0, 0.0, 0.0],
        };
        let bytes = size_of::<PresentFrame>() as u64;
        if cx.allocate_buffer(self.frame, bytes) == Allocation::Fresh || self.sent != Some(shown) {
            cx.write(self.frame, 0, &[shown]);
            self.sent = Some(shown);
        }
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.skipped.contains(&view.view) {
            cx.run(self.cover);
        } else {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass == self.cover {
            let _ = cx.render();
            return;
        }
        cx.render().draw(
            &self.pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        );
    }
}
