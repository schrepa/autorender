include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod atlas;
mod fonts;
mod operations;
mod smokes;
mod typeset;

use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Error, ErrorCode, Runs, SceneComponents, SceneCx, Schedule,
    Shown, Smoke, Source, View, gpu_struct, settings, stats,
};
use engine_places::text::{DEFAULT_TEXT_SIZE, FontId, Text, TextAlign};
use engine_render::{
    Bind, Buffer, DepthUse, Draw, GpuPart, Install, Layer, Lifetime, OverlayStack, Pipeline,
    Prepare, RenderBricks, RenderPipeline, Shape, Texture, wgpu,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use typeset::{Typesetter, ensure};

const GENERATED: &[&str] = &[<GlyphInstance as engine_core::GpuStruct>::WGSL];

settings! {
    #[section("text")]
    #[serde(deny_unknown_fields)]
    pub struct TextSettings {
        pub enabled: bool = true => any,
        pub atlas_size: u32 = 2048 => range(256, 8192),
        pub max_glyphs: u32 = 65_536 => range(1, 2_000_000),
    }
}

stats! {
    #[section("text_stats")]
    pub struct TextStats {
        pub runs: u32,
        pub glyphs_drawn: u32,
        pub glyphs_cut: u32,
        pub atlas_resets: u32,
        pub fonts: u32,
        pub failed_fonts: u32,
    }
}

gpu_struct! {
    pub struct GlyphInstance {
        pub place: [f32; 4],
        pub rect: [f32; 4],
        pub color: [f32; 4],
    }
}

pub(crate) type GlyphRange = (Option<u32>, Range<u32>);

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Label {
    pub text: String,
    pub font: Option<String>,
    pub size: f32,
    pub color: [f32; 4],
    pub anchor: [f32; 2],
    pub pivot: [f32; 2],
    pub offset: [f32; 2],
    pub max_width: Option<f32>,
    pub align: TextAlign,
    pub visible: bool,
    #[serde(skip)]
    #[schemars(skip)]
    bytes: Option<Arc<[u8]>>,
    #[serde(skip)]
    #[schemars(skip)]
    face: Option<FontId>,
}

impl Default for Label {
    fn default() -> Self {
        Self {
            text: String::new(),
            font: None,
            size: DEFAULT_TEXT_SIZE,
            color: [1.0; 4],
            anchor: [0.0; 2],
            pivot: [0.0; 2],
            offset: [0.0; 2],
            max_width: None,
            align: TextAlign::Left,
            visible: true,
            bytes: None,
            face: None,
        }
    }
}

fn check_label(label: &Label, path: &str) -> Result<(), Error> {
    if label.size.is_finite() && label.size > 0.0 {
        Ok(())
    } else {
        Err(Error::new(ErrorCode::InvalidNumber, format!("{path}.size")).got(label.size))
    }
}

fn prepare_label(mut label: Label, cx: &SceneCx<'_>, path: &str) -> Result<Label, Error> {
    if let Some(uri) = &label.font {
        label.bytes = cx.buffer(uri, &format!("{path}.font"))?;
    }
    Ok(label)
}

fn label_files(label: &Label) -> Vec<String> {
    label.font.iter().cloned().collect()
}

fn label_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Label)).unwrap_or(Value::Null)
}

fn label_example() -> Value {
    json!({
        "text": "Hello",
        "size": 18.0,
        "anchor": [0.5, 0.0],
        "pivot": [0.5, 0.0],
        "offset": [0.0, 12.0],
    })
}

fn install(mut text: ResMut<Text>) {
    ensure(&mut text);
}

fn sync(
    (text, settings, view): (Res<Text>, Res<TextSettings>, Res<View>),
    mut labels: Query<&mut Label>,
    mut stats: ResMut<TextStats>,
) {
    let Some(typesetter) = Typesetter::of(&text) else {
        return;
    };
    let mut state = typesetter.lock();
    state.sync(&settings, &view);
    for mut label in labels.iter_mut().filter(|label| label.bytes.is_some()) {
        let bytes = label.bytes.take();
        label.face = bytes.and_then(|bytes| state.load_font(bytes));
    }
    stats.set_if_neq(state.stats());
}

pub struct TextBrick;

impl Brick for TextBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[
        Smoke {
            name: "text_measure",
            runs: Runs::Headless,
            run: smokes::measure,
        },
        Smoke {
            name: "text_drawn",
            runs: Runs::Gpu,
            run: smokes::drawn,
        },
    ];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<TextSettings>()
            .stats::<TextStats>()
            .fill::<Text>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<Label>("text")
                    .check(check_label)
                    .prepare(prepare_label)
                    .files(label_files)
                    .schema(label_schema)
                    .example(label_example),
                None,
            )
            .systems(Schedule::FixedPreSimulation, install)
            .systems(Schedule::FrameStart, (install, sync))
            .operation::<operations::TextMeasure>()
            .observation::<operations::TextRuns>()
            .generated(GENERATED)
            .draws::<TextGpu, OverlayStack>(Layer::Over, TextGpu::draw_glyphs);
    }
}

struct TextGpu {
    pipeline: Pipeline<Self, OverlayStack>,
    instances: Buffer<Self>,
    atlas: Texture<Self>,
    staged: Vec<GlyphInstance>,
    ranges: Vec<GlyphRange>,
}

impl GpuPart for TextGpu {
    type Reads = (
        Query<'static, 'static, &'static Label, Shown>,
        Res<'static, Text>,
        Res<'static, TextSettings>,
        Res<'static, View>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::text::draw;
        let instances = cx.buffer("instances", Lifetime::Kept);
        let atlas = cx.texture("atlas", wgpu::TextureFormat::R8Unorm, Lifetime::Kept);
        let pipeline = cx.render_pipeline(
            OverlayStack,
            &RenderPipeline {
                vertex: draw::VS_TEXT,
                fragment: Some(draw::FS_TEXT),
                debug: None,
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                writes: wgpu::ColorWrites::ALL,
            },
            &[
                (draw::INSTANCES, Bind::Buffer(instances)),
                (draw::ATLAS, Bind::Texture(atlas)),
            ],
        )?;
        Ok(Self {
            pipeline,
            instances,
            atlas,
            staged: Vec::new(),
            ranges: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (labels, text, settings, view): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.ranges.clear();
        let Some(typesetter) = Typesetter::of(&text) else {
            return;
        };
        let mut state = typesetter.lock();
        if cx.fresh() {
            state.forget_atlas();
        }
        state.frame(
            labels.iter(),
            &settings,
            &view,
            &mut self.staged,
            &mut self.ranges,
        );
        let uploads = state.take_uploads();
        if self.staged.is_empty() && uploads.is_empty() {
            return;
        }
        let extent = |width, height| wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let side = state.atlas_side();
        cx.allocate(self.atlas, extent(side, side), 1);
        for upload in &uploads {
            cx.write_texture(
                self.atlas,
                wgpu::Origin3d {
                    x: upload.x,
                    y: upload.y,
                    z: 0,
                },
                &upload.coverage,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.width),
                    rows_per_image: Some(upload.height),
                },
                extent(upload.width, upload.height),
            );
        }
        if self.staged.is_empty() {
            return;
        }
        cx.grow_buffer(
            self.instances,
            std::mem::size_of_val(self.staged.as_slice()) as u64,
        );
        cx.write(self.instances, 0, &self.staged);
        cx.want_overlay();
    }
}

impl TextGpu {
    fn draw_glyphs(&self, step: &Option<u32>, draw: &mut Draw<'_, Self, OverlayStack>) -> u32 {
        let mut drawn = 0;
        for (_, glyphs) in self.ranges.iter().filter(|(held, _)| held == step) {
            draw.draw(
                &self.pipeline,
                Shape::Pulled {
                    vertices: 0..6,
                    instances: glyphs.clone(),
                },
            );
            drawn += 1;
        }
        drawn
    }
}
