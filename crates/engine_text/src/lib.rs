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
    AlphaMode, Asset, AssetLibrary, AssetResidency, AssetUses, AttachPoint, Attributes, Brick,
    Bricks, ComponentKind, Cull, DrawKind, Drawn, DrawnPoses, DrawnTable, DropReason, Error,
    ErrorCode, GlobalTransforms, KindDecl, KindSkip, LightingModel, Material, MaterialHandle,
    PassName, PassPurpose, PassRole, PassViews, Refusals, SceneComponents, SceneCx, SceneForm,
    Schedule, SlotSet, Source, Spawner, SurfaceSlot, Views, Writes, gpu_struct, never, settings,
    stats,
};
use engine_places_rendering::text::{DEFAULT_TEXT_SIZE, FontId, Glyph, Text, TextAlign};
use engine_render::{
    Allocation, Attach, Bind, Buffer, Draw, Encode, FaceCull, Form, FormColor, FormDepth,
    FormFragment, FormSide, FragmentPass, GpuPart, Install, KindForm, Lifetime, OwnFragments, Pass,
    PassFragment, Prepare, RenderBricks, Scene, SceneDraw, Shape, SlotModule, Texture, Varyings,
    wgpu,
};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use typeset::{Placed, Typesetter, ensure};

pub const TEXT: DrawKind = DrawKind("text");

pub const LABEL_PART: &str = "label";

const GLYPH: &str = "glyph";

const GLYPH_CUTOFF: f32 = 0.5;

const SCREEN_FORM: KindForm = KindForm {
    kind: TEXT,
    placed: FormSide::pulled(shaders::text::draw::VS_TEXT),
    owned: None,
    varyings: Varyings::Own(&OwnFragments::NONE),
};

const SCREEN: PassFragment = PassFragment {
    pass: FragmentPass::Own(TextPass::NAME),
    side: None,
    fragment: FormFragment::Entry {
        entry: shaders::text::draw::FS_TEXT,
        cull: FaceCull::None,
        depth: FormDepth::Ignored,
        color: FormColor::Blended(wgpu::BlendState::ALPHA_BLENDING),
    },
};

const LABEL_FORM: KindForm = KindForm {
    kind: TEXT,
    placed: FormSide {
        vertex: shaders::text::label::VS_LABEL,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide {
        vertex: shaders::text::label::VS_LABEL_VIEW,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    }),
    varyings: Varyings::SurfaceExtra,
};

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
    pub fn bundle(self) -> (Label, Drawn) {
        (self.0, Drawn::procedural(TEXT))
    }
}

fn in_world(world: &World, entity: Entity) -> bool {
    world
        .get::<Drawn>(entity)
        .is_some_and(|drawn| drawn.kind == TEXT)
}

impl SceneForm for WorldText {
    type Component = Label;

    fn insert(self, world: &mut World, entity: Entity) -> Result<(), Error> {
        let held = world.get::<Drawn>(entity).cloned();
        let (label, row) = self.bundle();
        let drawn = held.map_or(row, |held| Drawn {
            kind: TEXT,
            geometry: None,
            material: None,
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

enum LabelSurface {
    Unmade,
    Made(MaterialHandle),
    Refused(Error),
}

#[derive(Resource)]
struct LabelMaterial(LabelSurface);

impl Default for LabelMaterial {
    fn default() -> Self {
        Self(LabelSurface::Unmade)
    }
}

impl LabelMaterial {
    fn made(&self) -> Option<MaterialHandle> {
        match self.0 {
            LabelSurface::Made(material) => Some(material),
            LabelSurface::Unmade | LabelSurface::Refused(_) => None,
        }
    }

    fn held(&self, residency: &AssetResidency) -> LabelHeld {
        match &self.0 {
            LabelSurface::Made(made) if residency.material_resident(*made) => {
                LabelHeld::Resident(*made)
            }
            LabelSurface::Made(made) => residency
                .refusal(*made, LABEL_MATERIAL_PATH)
                .map_or(LabelHeld::Waiting, LabelHeld::Refused),
            LabelSurface::Unmade => LabelHeld::Waiting,
            LabelSurface::Refused(error) => LabelHeld::Refused(error.clone()),
        }
    }
}

const LABEL_MATERIAL: &str = "the label material";
const LABEL_MATERIAL_PATH: &str = "$.label.material";

enum LabelHeld {
    Waiting,
    Resident(MaterialHandle),
    Refused(Error),
}

#[derive(Component)]
struct LabelMaterialHeld;

fn label_material() -> Result<Material, Error> {
    Ok(Material::new([1.0; 4])?
        .with_slot(SurfaceSlot::Opacity.name(), GLYPH)?
        .with_alpha_mode(AlphaMode::Mask {
            cutoff: GLYPH_CUTOFF,
        })?
        .with_cull(Cull::None)
        .with_lighting(LightingModel::Unlit))
}

fn hold_label_material(
    labels: Query<(Entity, &Drawn), With<Label>>,
    (mut material, mut assets): (ResMut<LabelMaterial>, ResMut<AssetLibrary>),
    mut refusals: ResMut<Refusals>,
    residency: Res<AssetResidency>,
    held: Query<(Entity, &AssetUses), With<LabelMaterialHeld>>,
    mut spawner: Spawner<TextBrick>,
) {
    let live = match &material.0 {
        LabelSurface::Made(made) => assets.material(*made).is_some(),
        LabelSurface::Refused(_) => true,
        LabelSurface::Unmade => false,
    };
    if !live {
        if !labels.iter().any(|(_, drawn)| drawn.kind == TEXT) {
            return;
        }
        material.0 = match label_material().and_then(|made| assets.insert_material(made)) {
            Ok(made) => LabelSurface::Made(made),
            Err(error) => LabelSurface::Refused(error),
        };
    }
    let placed: Vec<Entity> = labels
        .iter()
        .filter(|(_, drawn)| drawn.kind == TEXT)
        .map(|(entity, _)| entity)
        .collect();
    for &entity in &placed {
        let verdict = match &material.0 {
            LabelSurface::Refused(error) => Err(error.clone()),
            LabelSurface::Made(_) | LabelSurface::Unmade => Ok(()),
        };
        refusals.judge(entity, LABEL_PART, verdict);
    }
    refusals.keep_part(LABEL_PART, |entity| placed.contains(&entity));
    let wanted: Vec<Asset> = material.made().map(Asset::Material).into_iter().collect();
    match held.single() {
        Ok((entity, uses)) => {
            if uses.assets() != wanted.as_slice() {
                spawner.insert(entity, AssetUses::new(&residency, wanted));
            }
        }
        Err(_) => {
            spawner.spawn((LabelMaterialHeld, AssetUses::new(&residency, wanted)));
        }
    }
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
            .resource::<LabelMaterial>()
            .systems(Schedule::ViewSync, hold_label_material)
            .operation::<operations::TextMeasure>()
            .observation::<operations::TextRuns>()
            .generated(GENERATED)
            .pass::<TextPass>()
            .material_slot(SlotModule {
                slot: SurfaceSlot::Opacity.name(),
                name: GLYPH,
                module: shaders::text::glyph::MODULE,
                maps: &[],
                values: &[],
                reach: 0.0,
            })
            .draw_kind(KindDecl {
                kind: TEXT,
                needs: Attributes::NONE,
                streams: &[],
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
    screen: Form<Self>,
    form: Form<Self>,
    material: LabelHeld,
    instances: Buffer<Self>,
    labels: Buffer<Self>,
    atlas: Texture<Self>,
    placed: Vec<Glyph>,
    staged: Vec<GlyphInstance>,
    worlds: Vec<WorldLabel>,
    unlabeled: u32,
    placed_world: Vec<(u32, Glyph)>,
    labeled: Vec<LabelGlyph>,
}

impl TextGpu {
    fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        draw.drop_rows(DropReason::RowIncomplete, self.unlabeled);
        if self.labeled.is_empty() {
            return if !self.worlds.is_empty() {
                draw.skip("labels place no glyphs")
            } else if self.unlabeled > 0 {
                0
            } else {
                draw.skip("no drawn label is placed")
            };
        }
        if scene.role == PassRole::Blended {
            return draw.skip("labels are masked, never blended");
        }
        let material = match &self.material {
            LabelHeld::Resident(material) => *material,
            LabelHeld::Waiting => return draw.wait(LABEL_MATERIAL),
            LabelHeld::Refused(error) => return draw.refuse(LABEL_MATERIAL, error),
        };
        let Some(pipeline) = draw.form(&self.form) else {
            return draw.skip("pass has no label form");
        };
        draw.draw(
            &pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.labeled.len() as u32,
                material: Some(material),
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
        self.unlabeled = 0;
        for (slot, entity, _) in table.rows(TEXT) {
            if labels.get(entity).is_err() {
                self.unlabeled += 1;
                continue;
            }
            let Some(pose) = poses.drawn_pose(globals, entity) else {
                continue;
            };
            let previous = poses.previous_drawn_pose(globals, entity).unwrap_or(pose);
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
        (Res<'static, LabelMaterial>, Res<'static, AssetResidency>),
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::text::{draw, faces::text as face, label};
        let instances = cx.buffer("instances", Lifetime::Kept);
        let labels = cx.buffer("labels", Lifetime::Kept);
        let atlas = cx.texture("atlas", wgpu::TextureFormat::R8Unorm, Lifetime::Kept);
        cx.frame(face::TEXT_ATLAS, Bind::Texture(atlas))?;
        let pass = cx
            .pass(TextPass::NAME)
            .at(TextPass::PASS.at)
            .span(TextPass::NAME)
            .id();
        let screen = cx.form_owning(
            &SCREEN_FORM,
            &[SCREEN],
            &[(draw::INSTANCES, Bind::Buffer(instances))],
        )?;
        let form = cx.form(&LABEL_FORM, &[(label::LABELS, Bind::Buffer(labels))])?;
        Ok(Self {
            pass,
            screen,
            form,
            material: LabelHeld::Waiting,
            instances,
            labels,
            atlas,
            placed: Vec::new(),
            staged: Vec::new(),
            worlds: Vec::new(),
            unlabeled: 0,
            placed_world: Vec::new(),
            labeled: Vec::new(),
        })
    }

    fn prepare(&mut self, reads: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        let ((labels, table, globals, poses), text, settings, views, (material, residency)) = reads;
        self.staged.clear();
        self.labeled.clear();
        self.unlabeled = 0;
        self.material = material.held(&residency);
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
        let worlds = self.worlds.iter().enumerate().filter_map(|(index, world)| {
            let (_, label, _) = labels.get(world.entity).ok()?;
            Some((label, Placed::World(index as u32)))
        });
        let shown = state.frame(
            screen.chain(worlds),
            &settings,
            views.frame(),
            (&mut self.placed, &mut self.placed_world),
        );
        self.staged
            .extend(self.placed.iter().map(|glyph| GlyphInstance {
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
        let Some(pipeline) = draw.form(&self.screen) else {
            return 0;
        };
        draw.draw(
            &pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.staged.len() as u32,
                material: None,
            },
        );
        1
    }
}
