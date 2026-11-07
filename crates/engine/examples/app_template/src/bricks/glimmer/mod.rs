include!(concat!(env!("OUT_DIR"), "/bricks/glimmer.rs"));

use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{
    AttachPoint, Attributes, Brick, Bricks, Camera, Clear, Cursor, DrawKind, Drawn, DrawnPoses,
    DrawnTable, Error, Events, Feed, FrameTime, GlobalTransforms, GpuStruct, Interest, KindDecl,
    Node, Pass, PassName, PassViews, Rect, Requests, ScenePass, Schedule, SlotSet, Source,
    Spawner, SurfaceSlot, TargetName, Ticket, Transform, View, Writes, gpu_struct, settings,
    stats,
};
use engine_render::{
    Bind, Buffer, DepthUse, Draw, GpuPart, Install, LightingModule, Lifetime, Pipeline, Prepare,
    RenderBricks, RenderPipeline, Scene, SceneDraw, Shape, SlotModule, wgpu,
};

pub const GLIMMER: DrawKind = DrawKind("glimmer");

const RING: Scene = Scene::of::<GlimmerPass>();

const GENERATED: &[&str] = &[<GlimmerInstance as GpuStruct>::WGSL];

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
        #[pending] pub asked: u32,
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
        takes: ringed,
    };
}

fn ringed(kind: DrawKind) -> bool {
    kind == GLIMMER
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
            .draw_kind(KindDecl {
                kind: GLIMMER,
                vertex: shaders::glimmer::draw::VS_GLIMMER.name(),
                needs: Attributes::NONE,
            })
            .pass::<GlimmerPass>()
            .material_slot(SlotModule {
                slot: SurfaceSlot::Emissive,
                name: "glimmer",
                module: shaders::glimmer::glow::MODULE,
                maps: &[],
                values: &["glimmer"],
                reach: 0.0,
            })
            .lighting_model(LightingModule {
                name: "glimmer",
                module: shaders::glimmer::bands::MODULE,
            })
            .generated(GENERATED)
            .draws::<GlimmerGpu>(GLIMMER, GlimmerGpu::draw)
            .systems(Schedule::FrameStart, open_view)
            .systems(Schedule::Presentation, (ask, answer));
    }
}

fn enabled(settings: Res<Glimmer>) -> bool {
    settings.enabled
}

fn open_view(opened: Query<(), With<GlimmerView>>, mut spawner: Spawner<GlimmerBrick>) {
    if !opened.is_empty() {
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
    let census = sparks.census();
    stats.asked = census.asked;
    stats.ready = census.ready;
}

struct GlimmerGpu {
    instances: Buffer<Self>,
    solids: Vec<(&'static str, Pipeline<Self, Scene>)>,
    ring: Pipeline<Self, Scene>,
    staged: Vec<GlimmerInstance>,
}

fn quad(
    vertex: engine_core::Entry,
    fragment: engine_core::Entry,
    depth: DepthUse,
) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex,
        fragment: Some(fragment),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}

impl GpuPart for GlimmerGpu {
    type Reads = (
        Res<'static, DrawnTable>,
        (Res<'static, GlobalTransforms>, Res<'static, DrawnPoses>),
        Res<'static, Glimmer>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::glimmer::draw;
        let instances = cx.buffer("glimmers", Lifetime::Kept);
        let written = DepthUse::Write {
            bias: wgpu::DepthBiasState::default(),
        };
        let listed: Vec<ScenePass> = cx
            .passes()
            .listed()
            .iter()
            .filter(|listed| {
                listed.bit.is_some()
                    && listed.name != GlimmerPass::NAME
                    && (listed.pass.takes)(GLIMMER)
                    && listed.pass.views != PassViews::Owned
                    && listed.pass.at != AttachPoint::Transparent
            })
            .copied()
            .collect();
        let mut solids = Vec::with_capacity(listed.len());
        for listed in listed {
            let fragment = if listed.pass.writes.contains(TargetName::COLOR) {
                draw::FS_GLIMMER
            } else {
                draw::FS_GLIMMER_DEPTH
            };
            let solid = cx.render_pipeline(
                Scene::named(listed.name),
                &quad(draw::VS_GLIMMER, fragment, written),
                &[(draw::GLIMMERS, Bind::Buffer(instances))],
            )?;
            solids.push((listed.name, solid));
        }
        let ring = cx.render_pipeline(
            RING,
            &quad(draw::VS_RING, draw::FS_RING, DepthUse::Test),
            &[(draw::GLIMMERS, Bind::Buffer(instances))],
        )?;
        Ok(Self {
            instances,
            solids,
            ring,
            staged: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (table, (globals, poses), settings): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.staged.clear();
        for (slot, _, thing) in table.rows(GLIMMER) {
            let Some(pose) = poses.drawn_pose(&globals, thing.node) else {
                continue;
            };
            let previous = poses
                .previous_drawn_pose(&globals, thing.node)
                .unwrap_or(pose);
            let size = thing.values.floats.first().copied().unwrap_or(settings.size);
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
        cx.allocate_buffer(self.instances, bytes as u64);
        cx.write(self.instances, 0, &self.staged);
    }
}

impl GlimmerGpu {
    fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let solid = self
            .solids
            .iter()
            .find(|(name, _)| *name == scene.pass.name())
            .map(|(_, solid)| solid);
        let pipeline = if scene.pass == RING {
            &self.ring
        } else if let Some(solid) = solid {
            solid
        } else {
            return 0;
        };
        let count = self.staged.len() as u32;
        draw.draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..count,
            },
        );
        count
    }
}
