engine::brick!(glimmer);

use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{
    AlphaMode, Asset, AssetLibrary, AssetResidency, AssetUses, AttachPoint, Attributes, Brick,
    Bricks, Camera, Clear, Cull, Cursor, DrawKind, Drawn, DrawnPoses, DrawnTable, EMISSIVE_MAP,
    Error, Events, Feed, FrameTime, GlobalTransforms, GpuStruct, Interest, KindDecl, LightingModel,
    Material, MaterialHandle, Node, Pass, PassName, PassPurpose, PassRole, PassViews, Rect,
    Refusals, Requests, Schedule, SlotSet, Source, Spawner, SurfaceSlot, Ticket, Transform, View,
    Writes, gpu_struct, settings, stats,
};
use engine_render::{
    Allocation, Bind, Buffer, DepthUse, Draw, Form, FormSide, GpuPart, Install, KindForm, Lifetime,
    LightingModule, Pipeline, Prepare, RenderBricks, RenderPipeline, Scene, SceneDraw, Shape,
    SlotModule, Varyings, wgpu,
};

pub const GLIMMER: DrawKind = DrawKind("glimmer");

const RING: Scene = Scene::of::<GlimmerPass>();

const GENERATED: &[&str] = &[<GlimmerInstance as GpuStruct>::WGSL];

pub const GLIMMER_PART: &str = "glimmer";

const LOOK: &str = "glimmer";

const DISC_CUTOFF: f32 = 0.5;

const FORM: KindForm = KindForm {
    kind: GLIMMER,
    placed: FormSide {
        vertex: shaders::glimmer::draw::VS_GLIMMER,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide {
        vertex: shaders::glimmer::draw::VS_GLIMMER_VIEW,
        streams: &[],
        depth_bias: FormSide::UNBIASED,
    }),
    varyings: Varyings::SurfaceExtra,
};

settings! {
    #[section("glimmer")]
    pub struct Glimmer {
        pub enabled: bool = true => any,
        pub size: f32 = 0.25 => range(0.01, 100.0),
        pub ring: [f32; 4] = [1.0, 0.85, 0.3, 1.0] => range(0.0, 1.0),
        pub events: u32 = 256 => at_least(1),
    }
}

stats! {
    #[section("glimmer_stats")]
    pub struct GlimmerStats {
        pub ready: u32,
        pub lit: u32,
        pub missed: u32,
    }
}

gpu_struct! {
    pub struct GlimmerInstance {
        pub place: [f32; 4],
        pub previous: [f32; 4],
        pub color: [f32; 4],
        pub ring: [f32; 4],
        pub drawn: [u32; 4],
    }
}

pub struct GlimmerPass;

impl PassName for GlimmerPass {
    const NAME: &'static str = "glimmer";
    const PASS: Pass = Pass {
        writes: Writes::LIT,
        reads: SlotSet::POSITION,
        views: PassViews::Main,
        at: AttachPoint::Opaque,
        purpose: PassPurpose::Scene,
        takes: ringed,
    };
}

fn ringed(kind: &KindDecl) -> bool {
    kind.kind == GLIMMER
}

#[derive(Component)]
pub struct GlimmerView;

#[derive(Resource)]
pub struct GlimmerBook {
    sparks: Requests<Entity, f64>,
    held: BTreeMap<Entity, Interest<Entity>>,
    lit: Feed<Entity>,
    seen: Cursor,
}

impl Default for GlimmerBook {
    fn default() -> Self {
        Self {
            sparks: Requests::default(),
            held: BTreeMap::new(),
            lit: Feed::new(1),
            seen: Cursor::default(),
        }
    }
}

pub struct GlimmerBrick;

impl Brick for GlimmerBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(enabled)
            .settings::<Glimmer>()
            .stats::<GlimmerStats>()
            .resource::<GlimmerBook>()
            .resource::<GlimmerMaterial>()
            .draw_kind(KindDecl {
                kind: GLIMMER,
                needs: Attributes::NONE,
                streams: &[],
                skips: &[],
            })
            .pass::<GlimmerPass>()
            .material_slot(SlotModule {
                slot: SurfaceSlot::Emissive.name(),
                name: LOOK,
                module: shaders::glimmer::glow::MODULE,
                maps: &[EMISSIVE_MAP],
                values: &[LOOK],
                reach: 0.0,
            })
            .material_slot(SlotModule {
                slot: SurfaceSlot::Opacity.name(),
                name: LOOK,
                module: shaders::glimmer::disc::MODULE,
                maps: &[],
                values: &[],
                reach: 0.0,
            })
            .lighting_model(LightingModule {
                name: LOOK,
                module: shaders::glimmer::bands::MODULE,
            })
            .generated(GENERATED)
            .draws::<GlimmerGpu>(GLIMMER, GlimmerGpu::draw)
            .systems(Schedule::FrameStart, open_view)
            .systems(Schedule::ViewSync, hold_material)
            .systems(Schedule::Presentation, (ask, answer));
    }
}

fn enabled(settings: Res<Glimmer>) -> bool {
    settings.enabled
}

fn open_view(
    table: Res<DrawnTable>,
    opened: Query<(), With<GlimmerView>>,
    mut spawner: Spawner<GlimmerBrick>,
) {
    if !opened.is_empty() || table.rows(GLIMMER).next().is_none() {
        return;
    }
    let camera = spawner.spawn((
        Transform::looking_at(Vec3::new(0.0, 40.0, 30.0), Vec3::ZERO),
        Camera::default(),
        Node::default(),
    ));
    spawner.spawn((
        GlimmerView,
        View {
            viewport: Rect {
                x: 0.75,
                y: 0.0,
                width: 0.25,
                height: 0.25,
            },
            clear: Clear::DepthOnly,
            order: 1,
            ..View::new(camera)
        },
    ));
}

fn ask(drawn: Query<(Entity, &Drawn)>, mut book: ResMut<GlimmerBook>) {
    let GlimmerBook { sparks, held, .. } = &mut *book;
    held.retain(|entity, _| {
        drawn
            .get(*entity)
            .is_ok_and(|(_, thing)| thing.kind == GLIMMER)
    });
    for (entity, thing) in &drawn {
        if thing.kind == GLIMMER && !held.contains_key(&entity) {
            held.insert(entity, sparks.ask(entity));
        }
    }
}

fn answer(
    time: Res<FrameTime>,
    settings: Res<Glimmer>,
    mut book: ResMut<GlimmerBook>,
    mut stats: ResMut<GlimmerStats>,
) {
    let GlimmerBook {
        sparks, lit, seen, ..
    } = &mut *book;
    sparks.settle(time.real_elapsed);
    let asked: Vec<(Entity, Ticket)> = sparks
        .asked()
        .map(|(entity, ticket)| (*entity, ticket))
        .collect();
    lit.resize(settings.events as usize);
    for (entity, ticket) in asked {
        if sparks.ready(&entity, ticket, time.real_elapsed) {
            lit.push(entity);
        }
    }
    match lit.events(seen) {
        Events::Listed(listed) => stats.lit += listed.len() as u32,
        Events::Missed(missed) => stats.missed += missed as u32,
    }
    stats.ready = sparks.census().ready;
}

enum GlimmerSurface {
    Unmade,
    Made(MaterialHandle),
    Refused(Error),
}

#[derive(Resource)]
struct GlimmerMaterial(GlimmerSurface);

impl Default for GlimmerMaterial {
    fn default() -> Self {
        Self(GlimmerSurface::Unmade)
    }
}

impl GlimmerMaterial {
    fn made(&self) -> Option<MaterialHandle> {
        match self.0 {
            GlimmerSurface::Made(material) => Some(material),
            GlimmerSurface::Unmade | GlimmerSurface::Refused(_) => None,
        }
    }
}

#[derive(Component)]
struct GlimmerMaterialHeld;

fn disc_material() -> Result<Material, Error> {
    Ok(Material::new([1.0; 4])?
        .with_slot(SurfaceSlot::Opacity.name(), LOOK)?
        .with_alpha_mode(AlphaMode::Mask {
            cutoff: DISC_CUTOFF,
        })?
        .with_cull(Cull::None)
        .with_lighting(LightingModel::Unlit))
}

fn hold_material(
    (table, residency): (Res<DrawnTable>, Res<AssetResidency>),
    (mut material, mut assets): (ResMut<GlimmerMaterial>, ResMut<AssetLibrary>),
    mut refusals: ResMut<Refusals>,
    held: Query<(Entity, &AssetUses), With<GlimmerMaterialHeld>>,
    mut spawner: Spawner<GlimmerBrick>,
) {
    let live = match &material.0 {
        GlimmerSurface::Made(made) => assets.material(*made).is_some(),
        GlimmerSurface::Refused(_) => true,
        GlimmerSurface::Unmade => false,
    };
    if !live {
        if table.rows(GLIMMER).next().is_none() {
            return;
        }
        material.0 = match disc_material().and_then(|made| assets.insert_material(made)) {
            Ok(made) => GlimmerSurface::Made(made),
            Err(error) => GlimmerSurface::Refused(error),
        };
    }
    let rows: Vec<Entity> = table.rows(GLIMMER).map(|(_, entity, _)| entity).collect();
    for &entity in &rows {
        let verdict = match &material.0 {
            GlimmerSurface::Refused(error) => Err(error.clone()),
            GlimmerSurface::Made(_) | GlimmerSurface::Unmade => Ok(()),
        };
        refusals.judge(entity, GLIMMER_PART, verdict);
    }
    refusals.keep_part(GLIMMER_PART, |entity| rows.contains(&entity));
    let wanted: Vec<Asset> = material.made().map(Asset::Material).into_iter().collect();
    match held.single() {
        Ok((entity, uses)) => {
            if uses.assets() != wanted.as_slice() {
                spawner.insert(entity, AssetUses::new(&residency, wanted));
            }
        }
        Err(_) => {
            spawner.spawn((GlimmerMaterialHeld, AssetUses::new(&residency, wanted)));
        }
    }
}

struct GlimmerGpu {
    instances: Buffer<Self>,
    form: Form<Self>,
    ring: Pipeline<Self, Scene>,
    material: Option<MaterialHandle>,
    staged: Vec<GlimmerInstance>,
}

impl GpuPart for GlimmerGpu {
    type Reads = (
        Res<'static, DrawnTable>,
        (Res<'static, GlobalTransforms>, Res<'static, DrawnPoses>),
        Res<'static, Glimmer>,
        (Res<'static, GlimmerMaterial>, Res<'static, AssetResidency>),
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::glimmer::draw;
        let instances = cx.buffer("glimmers", Lifetime::Kept);
        let bound = [(draw::GLIMMERS, Bind::Buffer(instances))];
        let form = cx.form(&FORM, &bound)?;
        let ring = cx.render_pipeline(
            RING,
            &RenderPipeline {
                vertex: draw::VS_RING,
                fragment: Some(draw::FS_RING),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::Test,
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &bound,
        )?;
        Ok(Self {
            instances,
            form,
            ring,
            material: None,
            staged: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (table, (globals, poses), settings, (material, residency)): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.material = material
            .made()
            .filter(|made| residency.material_resident(*made));
        self.staged.clear();
        for (slot, entity, thing) in table.rows(GLIMMER) {
            let Some(pose) = poses.drawn_pose(&globals, entity) else {
                continue;
            };
            let previous = poses.previous_drawn_pose(&globals, entity).unwrap_or(pose);
            let size = thing
                .values
                .floats
                .first()
                .copied()
                .unwrap_or(settings.size);
            let [red, green, blue, alpha] = thing.values.tint;
            let [glow_red, glow_green, glow_blue] = thing.values.emission;
            self.staged.push(GlimmerInstance {
                place: pose.w_axis.truncate().extend(size).to_array(),
                previous: previous.w_axis.truncate().extend(size).to_array(),
                color: [red + glow_red, green + glow_green, blue + glow_blue, alpha],
                ring: settings.ring,
                drawn: [slot, 0, 0, 0],
            });
        }
        let bytes = self.staged.len().max(1) * std::mem::size_of::<GlimmerInstance>();
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.instances, bytes as u64);
        cx.write(self.instances, 0, &self.staged);
    }
}

impl GlimmerGpu {
    fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let count = self.staged.len() as u32;
        if count == 0 {
            return draw.skip("no glimmer has a pose yet");
        }
        if scene.pass == RING {
            draw.draw(
                &self.ring,
                Shape::Pulled {
                    vertices: 0..6,
                    instances: 0..count,
                    material: None,
                },
            );
            return count;
        }
        if scene.role == PassRole::Blended {
            return draw.skip("glimmers are masked, never blended");
        }
        let Some(material) = self.material else {
            return draw.skip("the glimmer material is not resident yet");
        };
        let Some(pipeline) = draw.form(&self.form) else {
            return draw.skip("pass has no glimmer form");
        };
        draw.draw(
            &pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..count,
                material: Some(material),
            },
        );
        count
    }
}
