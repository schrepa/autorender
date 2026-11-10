#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod atlas;
mod fonts;
mod operations;
mod typeset;

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::world::World;
use engine_core::{
    AttachPoint, Attributes, Brick, Bricks, ComponentKind, DrawKind, Drawn, DrawnPoses, DrawnTable,
    Error, ErrorCode, GlobalTransforms, KindDecl, KindSkip, PassName, PassPurpose, PassViews,
    SceneComponents, SceneCx, SceneForm, ScenePass, Schedule, SlotSet, Source, TargetName, Views,
    Writes, gpu_struct, never, settings, stats,
};
use engine_places_rendering::text::{DEFAULT_TEXT_SIZE, FontId, Glyph, Text, TextAlign};
use engine_render::{
    Allocation, Attach, Bind, Buffer, DepthUse, Draw, Encode, GpuPart, Install, Lifetime, Own,
    Pass, Pipeline, Prepare, RenderBricks, RenderPipeline, Scene, SceneDraw, Shape, Texture, wgpu,
};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use typeset::{Placed, Typesetter, ensure};

pub const TEXT: DrawKind = DrawKind("text");

const GENERATED: &[&str] = &[
    <GlyphInstance as engine_core::GpuStruct>::WGSL,
    <LabelGlyph as engine_core::GpuStruct>::WGSL,
];

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

gpu_struct! {
    pub struct LabelGlyph {
        pub origin: [f32; 4],
        pub previous: [f32; 4],
        pub place: [f32; 4],
        pub rect: [f32; 4],
        pub color: [f32; 4],
        pub drawn: [u32; 4],
    }
}

struct WorldLabel {
    entity: Entity,
    slot: u32,
    origin: Vec3,
    previous: Vec3,
    scale: f32,
}

pub(crate) struct TextPass;

impl PassName for TextPass {
    const NAME: &'static str = "text";
    const PASS: engine_core::Pass = engine_core::Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Main,
        at: AttachPoint::Overlay,
        purpose: PassPurpose::Scene,
        takes: never,
    };
}

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorldText(pub Label);

impl WorldText {
    pub fn bundle(self, node: Entity) -> (Label, Drawn) {
        (self.0, Drawn::procedural(TEXT, node))
    }
}

fn in_world(world: &World, entity: Entity) -> bool {
    world.get::<Drawn>(entity).is_some_and(|drawn| drawn.kind == TEXT)
}

impl SceneForm for WorldText {
    type Component = Label;

    fn insert(self, world: &mut World, entity: Entity) -> Result<(), Error> {
        let held = world.get::<Drawn>(entity).cloned();
        let (label, row) = self.bundle(entity);
        let drawn = held.map_or(row, |held| Drawn {
            kind: TEXT,
            geometry: None,
            material: None,
            node: entity,
            ..held
        });
        world.entity_mut(entity).insert((label, drawn));
        Ok(())
    }

    fn read(world: &World, entity: Entity) -> Option<Self> {
        world
            .get::<Label>(entity)
            .filter(|_| in_world(world, entity))
            .cloned()
            .map(Self)
    }

    fn remove(world: &mut World, entity: Entity) {
        if in_world(world, entity) {
            world.entity_mut(entity).remove::<(Label, Drawn)>();
        }
    }
}

fn install(mut text: ResMut<Text>) {
    ensure(&mut text);
}

fn sync(
    (text, settings, views): (Res<Text>, Res<TextSettings>, Res<Views>),
    mut labels: Query<&mut Label>,
    mut stats: ResMut<TextStats>,
) {
    let Some(typesetter) = Typesetter::of(&text) else {
        return;
    };
    let mut state = typesetter.lock();
    state.sync(&settings, views.frame());
    state.advance();
    for mut label in labels.iter_mut().filter(|label| label.bytes.is_some()) {
        let bytes = label.bytes.take();
        label.face = bytes.and_then(|bytes| state.load_font(bytes));
    }
    stats.set_if_neq(state.stats());
}

pub struct TextBrick;

impl Brick for TextBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<TextSettings>()
            .stats::<TextStats>()
            .errors(&[typeset::INVALID_FONT])
            .fill::<Text>(Some(shaders::text::faces::text::MODULE))
            .join::<SceneComponents>(
                ComponentKind::of::<Label>("text")
                    .check(check_label)
                    .prepare(prepare_label)
                    .files(label_files)
                    .schema(label_schema),
                None,
            )
            .join::<SceneComponents>(
                ComponentKind::form::<WorldText>("world_text")
                    .check(|text, path| check_label(&text.0, path))
                    .prepare(|text, cx, path| prepare_label(text.0, cx, path).map(WorldText))
                    .files(|text| label_files(&text.0))
                    .schema(label_schema),
                None,
            )
            .systems(Schedule::FixedPreSimulation, install)
            .systems(Schedule::FrameStart, (install, sync))
            .operation::<operations::TextMeasure>()
            .observation::<operations::TextRuns>()
            .generated(GENERATED)
            .pass::<TextPass>()
            .draw_kind(KindDecl {
                kind: TEXT,
                needs: Attributes::NONE,
                skips: &[
                    KindSkip {
                        pass: "shadow",
                        reason: "labels cast no shadows",
                    },
                    KindSkip {
                        pass: "probe",
                        reason: "labels stay out of probe captures",
                    },
                ],
            })
            .draws::<TextGpu>(TEXT, TextGpu::draw);
    }
}

struct TextGpu {
    pass: Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    placings: Vec<(&'static str, Pipeline<Self, Scene>)>,
    instances: Buffer<Self>,
    labels: Buffer<Self>,
    atlas: Texture<Self>,
    placed: Vec<Glyph>,
    staged: Vec<GlyphInstance>,
    worlds: Vec<WorldLabel>,
    placed_world: Vec<(u32, Glyph)>,
    labeled: Vec<LabelGlyph>,
}

impl TextGpu {
    fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if self.labeled.is_empty() {
            return 0;
        }
        let Some((_, placing)) = self
            .placings
            .iter()
            .find(|(name, _)| *name == scene.pass.name())
        else {
            return 0;
        };
        draw.draw(
            placing,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.labeled.len() as u32,
            },
        );
        1
    }

    fn place_worlds(
        &mut self,
        labels: &Query<'_, '_, (Entity, &'static Label, Option<&'static Drawn>)>,
        (table, globals, poses): (&DrawnTable, &GlobalTransforms, &DrawnPoses),
    ) {
        self.worlds.clear();
        for (slot, entity, drawn) in table.rows(TEXT) {
            let (Ok(_), Some(pose)) = (labels.get(entity), poses.drawn_pose(globals, drawn.node))
            else {
                continue;
            };
            let previous = poses.previous_drawn_pose(globals, drawn.node).unwrap_or(pose);
            self.worlds.push(WorldLabel {
                entity,
                slot,
                origin: pose.w_axis.truncate(),
                previous: previous.w_axis.truncate(),
                scale: pose.x_axis.truncate().length(),
            });
        }
    }

    fn stage_worlds(&mut self) {
        self.labeled.clear();
        let worlds = &self.worlds;
        self.labeled
            .extend(self.placed_world.iter().filter_map(|(index, glyph)| {
                let world = worlds.get(*index as usize)?;
                Some(LabelGlyph {
                    origin: world.origin.extend(world.scale).to_array(),
                    previous: world.previous.extend(world.scale).to_array(),
                    place: glyph.place,
                    rect: glyph.texel,
                    color: glyph.color,
                    drawn: [world.slot, 0, 0, 0],
                })
            }));
    }
}

impl GpuPart for TextGpu {
    type Reads = (
        (
            Query<'static, 'static, (Entity, &'static Label, Option<&'static Drawn>)>,
            Res<'static, DrawnTable>,
            Res<'static, GlobalTransforms>,
            Res<'static, DrawnPoses>,
        ),
        Res<'static, Text>,
        Res<'static, TextSettings>,
        Res<'static, Views>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::text::{draw, faces::text as face};
        let instances = cx.buffer("instances", Lifetime::Kept);
        let labels = cx.buffer("labels", Lifetime::Kept);
        let atlas = cx.texture("atlas", wgpu::TextureFormat::R8Unorm, Lifetime::Kept);
        cx.frame(face::TEXT_ATLAS, Bind::Texture(atlas))?;
        let pass = cx
            .pass(TextPass::NAME)
            .at(TextPass::PASS.at)
            .span(TextPass::NAME)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: draw::VS_TEXT,
                fragment: Some(draw::FS_TEXT),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                writes: wgpu::ColorWrites::ALL,
            },
            &[(draw::INSTANCES, Bind::Buffer(instances))],
        )?;
        let listed: Vec<ScenePass> = cx
            .passes()
            .listed()
            .iter()
            .filter(|listed| {
                listed.bit.is_some()
                    && cx.passes().takes(listed, TEXT)
                    && listed.pass.views != PassViews::Owned
                    && listed.pass.at != AttachPoint::Transparent
            })
            .copied()
            .collect();
        let mut placings = Vec::with_capacity(listed.len());
        for listed in listed {
            let fragment = if listed.pass.writes.contains(TargetName::COLOR) {
                draw::FS_LABEL
            } else {
                draw::FS_LABEL_DEPTH
            };
            let placing = cx.render_pipeline(
                Scene::named(listed.name),
                &RenderPipeline {
                    vertex: draw::VS_LABEL,
                    fragment: Some(fragment),
                    vertices: &[],
                    primitive: wgpu::PrimitiveState::default(),
                    depth: DepthUse::Write {
                        bias: wgpu::DepthBiasState::default(),
                    },
                    blend: None,
                    writes: wgpu::ColorWrites::ALL,
                },
                &[(draw::LABELS, Bind::Buffer(labels))],
            )?;
            placings.push((listed.name, placing));
        }
        Ok(Self {
            pass,
            pipeline,
            placings,
            instances,
            labels,
            atlas,
            placed: Vec::new(),
            staged: Vec::new(),
            worlds: Vec::new(),
            placed_world: Vec::new(),
            labeled: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        ((labels, table, globals, poses), text, settings, views): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.staged.clear();
        self.labeled.clear();
        let Some(typesetter) = Typesetter::of(&text) else {
            return;
        };
        self.place_worlds(&labels, (&table, &globals, &poses));
        let mut state = typesetter.lock();
        let screen = labels
            .iter()
            .filter(|&(entity, _, drawn)| {
                drawn.is_none_or(|drawn| drawn.kind != TEXT) && globals.drawn(entity)
            })
            .map(|(_, label, _)| (label, Placed::Screen));
        let worlds = self
            .worlds
            .iter()
            .enumerate()
            .filter_map(|(index, world)| {
                let (_, label, _) = labels.get(world.entity).ok()?;
                Some((label, Placed::World(index as u32)))
            });
        let shown = state.frame(
            screen.chain(worlds),
            &settings,
            views.frame(),
            (&mut self.placed, &mut self.placed_world),
        );
        self.staged.extend(self.placed.iter().map(|glyph| GlyphInstance {
            place: glyph.place,
            rect: glyph.texel,
            color: glyph.color,
        }));
        self.stage_worlds();
        let mut uploads = state.take_uploads();
        if shown == 0 && uploads.is_empty() {
            return;
        }
        let extent = |width, height| wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let side = state.atlas_side();
        if cx.allocate(self.atlas, extent(side, side), 1) == Allocation::Fresh {
            uploads = state.refill_atlas();
        }
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
        if !self.labeled.is_empty() {
            let bytes = std::mem::size_of_val(self.labeled.as_slice()) as u64;
            let (Allocation::Fresh | Allocation::Kept) = cx.grow_buffer(self.labels, bytes);
            cx.write(self.labels, 0, &self.labeled);
        }
        if self.staged.is_empty() {
            return;
        }
        let bytes = std::mem::size_of_val(self.staged.as_slice()) as u64;
        let (Allocation::Fresh | Allocation::Kept) = cx.grow_buffer(self.instances, bytes);
        cx.write(self.instances, 0, &self.staged);
        cx.run(self.pass);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.pass {
            return 0;
        }
        let mut draw = cx.render(Attach::Declared);
        draw.draw(
            &self.pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.staged.len() as u32,
            },
        );
        1
    }
}
