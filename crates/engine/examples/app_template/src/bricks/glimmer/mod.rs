engine::brick!(glimmer);

use std::collections::BTreeMap;
use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{
    AssetResidency, AttachPoint, Attributes, BlendedPass, Brick, Bricks, Camera, Clear, Cursor,
    DrawKind, Drawn, DrawnPoses, DrawnTable, EMISSIVE_MAP, Error, ErrorCode, Events, Expected,
    Feed, FrameTime, GlobalTransforms, GpuStruct, Interest, KindDecl, KindSkip, MaterialHandle,
    Node, Pass, PassName, PassPurpose, PassViews, Rect, Refusals, Requests, Schedule, SlotSet,
    Source, Spawner, SurfaceSlot, Ticket, Transform, View, Writes, gpu_struct, settings, stats,
};
use engine_render::{
    Allocation, Bind, Buffer, Draw, FaceCull, Form, FormColor, FormDepth, FormFragment, FormSide,
    FragmentPass, GpuPart, Install, KindForm, Lifetime, LightingModule, PassFragment, PointInput,
    Prepare, RenderBricks, Scene, SceneDraw, Shape, SlotModule, Varyings,
};

pub const GLIMMER: DrawKind = DrawKind("glimmer");

const GENERATED: &[&str] = &[<GlimmerInstance as GpuStruct>::WGSL];

pub const LOOK: &str = "glimmer";

const ADDED_LIGHT: PointInput = PointInput::new("added_light", "vec3<f32>", "vec3<f32>(0.0)");

pub const GLIMMER_PART: &str = "glimmer";

const FORM: KindForm = KindForm {
    kind: GLIMMER,
    placed: FormSide::pulled(shaders::glimmer::draw::VS_GLIMMER),
    owned: Some(FormSide::pulled(shaders::glimmer::draw::VS_GLIMMER_VIEW)),
    varyings: Varyings::SurfaceExtra,
};

const RING: PassFragment = PassFragment {
    pass: FragmentPass::Scene(Scene::of::<GlimmerPass>()),
    side: Some(FormSide::pulled(shaders::glimmer::draw::VS_RING)),
    fragment: FormFragment::Entry {
        entry: shaders::glimmer::draw::FS_RING,
        cull: FaceCull::None,
        depth: FormDepth::Tested,
        color: FormColor::Written,
    },
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

const SKIPS: &[KindSkip] = &[KindSkip::pass::<BlendedPass>(
    "glimmers are masked, never blended",
)];

pub struct GlimmerBrick;

impl Brick for GlimmerBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(enabled)
            .settings::<Glimmer>()
            .stats::<GlimmerStats>()
            .resource::<GlimmerBook>()
            .draw_kind(KindDecl {
                kind: GLIMMER,
                needs: Attributes::NONE,
                streams: &[],
                skips: SKIPS,
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
            .point_input(ADDED_LIGHT)
            .generated(GENERATED)
            .draws::<GlimmerGpu>(GLIMMER, GlimmerGpu::draw)
            .systems(Schedule::FrameStart, (open_view, judge))
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

fn judge(drawn: Query<(Entity, &Drawn)>, mut refusals: ResMut<Refusals>) {
    refusals.keep_part(GLIMMER_PART, |entity| {
        drawn
            .get(entity)
            .is_ok_and(|(_, thing)| thing.kind == GLIMMER)
    });
    for (entity, thing) in &drawn {
        if thing.kind == GLIMMER {
            refusals.judge(entity, GLIMMER_PART, glimmer_verdict(thing));
        }
    }
}

fn glimmer_verdict(thing: &Drawn) -> Result<(), Error> {
    match thing.material {
        Some(_) => Ok(()),
        None => Err(Error::new(ErrorCode::MissingField, "$.drawn.material")
            .expected(Expected::Present)
            .got(GLIMMER.name())),
    }
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

struct GlimmerGpu {
    instances: Buffer<Self>,
    form: Form<Self>,
    runs: Vec<(MaterialHandle, Range<u32>)>,
    staged: Vec<GlimmerInstance>,
}

impl GpuPart for GlimmerGpu {
    type Reads = (
        Res<'static, DrawnTable>,
        (Res<'static, GlobalTransforms>, Res<'static, DrawnPoses>),
        Res<'static, Glimmer>,
        Res<'static, AssetResidency>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let instances = cx.buffer("glimmers", Lifetime::Kept);
        let bound = [(shaders::glimmer::draw::GLIMMERS, Bind::Buffer(instances))];
        Ok(Self {
            instances,
            form: cx.form_owning(&FORM, &[RING], &bound)?,
            runs: Vec::new(),
            staged: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (table, (globals, poses), settings, residency): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let mut placed: Vec<(MaterialHandle, GlimmerInstance)> = Vec::new();
        for (slot, entity, thing) in table.rows(GLIMMER) {
            let held = thing
                .material
                .filter(|made| residency.material_resident(*made));
            let (Some(pose), Some(material)) = (poses.drawn_pose(&globals, entity), held) else {
                continue;
            };
            let previous = poses.previous_drawn_pose(&globals, entity).unwrap_or(pose);
            let size = *thing.values.floats.first().unwrap_or(&settings.size);
            let [red, green, blue, alpha] = thing.values.tint;
            let [glow_red, glow_green, glow_blue] = thing.values.emission;
            let instance = GlimmerInstance {
                place: pose.w_axis.truncate().extend(size).to_array(),
                previous: previous.w_axis.truncate().extend(size).to_array(),
                color: [red + glow_red, green + glow_green, blue + glow_blue, alpha],
                ring: settings.ring,
                drawn: [slot, 0, 0, 0],
            };
            placed.push((material, instance));
        }
        placed.sort_by_key(|(material, _)| *material);
        self.staged.clear();
        self.runs.clear();
        for (material, instance) in placed {
            let at = self.staged.len() as u32;
            self.staged.push(instance);
            match self.runs.last_mut() {
                Some((known, run)) if *known == material => run.end = at + 1,
                Some(_) | None => self.runs.push((material, at..at + 1)),
            }
        }
        let bytes = self.staged.len().max(1) * std::mem::size_of::<GlimmerInstance>();
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.instances, bytes as u64);
        cx.write(self.instances, 0, &self.staged);
    }
}

impl GlimmerGpu {
    fn draw(&self, _: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        if self.runs.is_empty() {
            return draw.skip("no glimmer has a pose and a resident material yet");
        }
        let Some(pipeline) = draw.form(&self.form) else {
            return draw.skip("pass has no glimmer form");
        };
        for (material, instances) in &self.runs {
            draw.draw(
                &pipeline,
                Shape::Pulled {
                    vertices: 0..6,
                    instances: instances.clone(),
                    material: Some(*material),
                },
            );
        }
        self.staged.len() as u32
    }
}
