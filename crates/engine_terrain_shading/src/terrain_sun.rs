use bevy_ecs::prelude::*;
use engine_core::{Cursor, DeviceLimits, Error, FrameTime, Readable, Readback, Views};
use engine_open_world_kit::ground::TerrainTiles;
use engine_places_rendering::sky::SkyLight;
use engine_places_rendering::sun::Sun;
use engine_render::{
    AfterRead, Allocation, Bind, Buffer, ComputePipeline, Count, Encode, Install, Lifetime, Pass,
    Prepare,
};
use engine_tiles::TileKey;
use glam::{Vec2, Vec3, Vec3Swizzles};

use crate::gpu::{
    HorizonsPiece, SUN_GROUP, SUN_LEVEL_LAYERS, SUN_LEVELS, SUN_SNAPS, SUN_STATE, SunJob,
    SunTerrain,
};
use crate::settings::TerrainShading;
use crate::state::{Atlas, AtlasMoves, TileId, Upload};
use crate::terrain::TerrainGpu;
use crate::terrain_horizons::Horizons;

const LEVELS: usize = SUN_LEVELS as usize;
const DROPPED: u32 = 0;
const MARCHED: u32 = 1;
const COUNTED: u32 = 2;
const COUNT_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Window {
    origin: Vec2,
    texel: f32,
    heading: Vec2,
    half: u32,
}

impl Window {
    fn reaches(&self, side: u32, cone: f32, (low, high, spacing): (Vec2, Vec2, f32)) -> bool {
        let along = self.heading;
        let across = along.perp();
        let span = |first: Vec2, last: Vec2, axis: Vec2| {
            [
                first,
                Vec2::new(last.x, first.y),
                Vec2::new(first.x, last.y),
                last,
            ]
            .map(|corner| corner.dot(axis))
            .into_iter()
            .fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(least, most), value| {
                    (
                        engine_math::min(least, value),
                        engine_math::max(most, value),
                    )
                },
            )
        };
        let last = self.origin + Vec2::splat(self.texel * side as f32);
        let margin = 2.0 * self.texel + 2.0 * spacing;
        let (start, _) = span(self.origin, last, along);
        let (left, right) = span(self.origin, last, across);
        let (_, farthest) = span(low, high, along);
        let (near, far) = span(low, high, across);
        if farthest < start - margin {
            return false;
        }
        let wide = engine_math::max(self.texel, cone * (farthest - start)) + margin;
        far >= left - wide && near <= right + wide
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    side: u32,
    base: u32,
    levels: u32,
    texel: f32,
    cone: f32,
    tall: f32,
    slack: f32,
    steps: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SunBuild {
    pub(crate) id: u32,
    pub(crate) windows: Vec<(u32, Window)>,
    pub(crate) base: u32,
    pub(crate) cone: f32,
    pub(crate) tall: f32,
    pub(crate) slack: f32,
    pub(crate) steps: u32,
}

#[derive(Resource, Default)]
pub(crate) struct SunPlan {
    key: Option<Key>,
    shown: [Option<Window>; LEVELS],
    pub(crate) build: Option<SunBuild>,
    marched: Option<u32>,
    slide: Option<(f64, Vec<(u32, Window)>)>,
    stale: u32,
    dropped: bool,
    tiles: Cursor,
    builds: u32,
    pub(crate) walked: u32,
    pub(crate) short: u32,
    pub(crate) lost: u32,
}

impl Readable for SunPlan {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>) {
        let words = bytemuck::try_pod_read_unaligned::<[u32; 2]>(read.value);
        match (tag, words) {
            (DROPPED, _) => self.dropped = true,
            (MARCHED, Ok([id, walked])) => {
                self.marched = Some(id);
                self.walked = walked;
            }
            (COUNTED, Ok([short, lost])) => {
                self.short = short.saturating_add(lost);
                self.lost = lost;
            }
            _ => {}
        }
    }
}

impl SunPlan {
    fn forget(&mut self) {
        self.shown = [None; LEVELS];
        self.build = None;
        self.marched = None;
        self.slide = None;
        self.dropped = false;
    }

    fn marches(&self, (tiles, atlas): (&TerrainTiles, &Atlas), (terrain, key): TileId) -> u32 {
        let (Some(marched), Some(entry)) = (
            self.key,
            tiles.terrains.iter().find(|entry| entry.terrain == terrain),
        ) else {
            return 0;
        };
        let pyramid = &entry.pyramid;
        let level = u32::from(key.level);
        let size = pyramid.tile_size(level);
        let low = entry.origin.xz() + Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
        let footprint = (low, low + Vec2::splat(size), pyramid.spacing(level));
        let top = pyramid.levels().saturating_sub(1);
        let rooted = atlas
            .slots
            .contains_key(&(terrain, TileKey::new(top as u8, 0, 0)));
        let reads = |window: &Window| {
            let first = (0..top)
                .take_while(|&below| pyramid.spacing(below + 1) <= window.texel)
                .count() as u32;
            !rooted || level >= first
        };
        let shown = self
            .shown
            .iter()
            .enumerate()
            .filter_map(|(at, window)| window.map(|window| (at as u32, window)));
        let built = self.build.iter().flat_map(|build| build.windows.iter());
        let sliding = self.slide.iter().flat_map(|(_, windows)| windows.iter());
        shown
            .chain(built.chain(sliding).copied())
            .filter(|(_, window)| {
                reads(window) && window.reaches(marched.side, marched.cone, footprint)
            })
            .fold(0, |stale, (at, _)| stale | 1 << at)
    }

    fn wanted(&self, key: &Key, eye: Vec2, heading: Vec2) -> Vec<(u32, Window)> {
        let side = key.side as f32;
        (0..key.levels)
            .filter_map(|level| {
                let texel = key.texel * (1_u32 << level) as f32;
                let snap = texel * side / SUN_SNAPS as f32;
                let origin = (eye / snap).round() * snap - Vec2::splat(0.5 * texel * side);
                let shown = self.shown[level as usize];
                let kept = shown.is_some_and(|shown| {
                    self.stale & 1 << level == 0
                        && shown.origin == origin
                        && shown.texel == texel
                        && shown.heading == heading
                });
                let half = shown.map_or(0, |shown| 1 - shown.half);
                (!kept).then_some((
                    level,
                    Window {
                        origin,
                        texel,
                        heading,
                        half,
                    },
                ))
            })
            .collect()
    }
}

fn layer(base: u32, level: u32, window: &Window) -> u32 {
    base + level * SUN_LEVEL_LAYERS + window.half * (SUN_LEVEL_LAYERS / 2)
}

fn record(base: u32, level: u32, side: u32, window: Option<&Window>) -> [f32; 4] {
    window.map_or([0.0; 4], |window| {
        [
            window.origin.x,
            window.origin.y,
            1.0 / (window.texel * side as f32),
            layer(base, level, window) as f32,
        ]
    })
}

fn held(tiles: &TerrainTiles, atlas: &Atlas) -> impl Iterator<Item = usize> {
    tiles
        .terrains
        .iter()
        .enumerate()
        .filter(|(_, entry)| {
            atlas
                .slots
                .keys()
                .any(|(terrain, _)| *terrain == entry.terrain)
        })
        .map(|(index, _)| index)
}

pub(crate) fn plan(
    (settings, tiles, atlas): (Res<TerrainShading>, Res<TerrainTiles>, Res<Atlas>),
    (sun, sky, views, limits, time): (
        Res<Sun>,
        Res<SkyLight>,
        Res<Views>,
        Res<DeviceLimits>,
        Res<FrameTime>,
    ),
    (mut plan, mut piece): (ResMut<SunPlan>, ResMut<HorizonsPiece>),
) {
    let plan = &mut *plan;
    let moves = AtlasMoves::since(&atlas, &mut plan.tiles);
    if moves.all {
        plan.stale = u32::MAX;
    }
    let shaped = moves
        .uploads
        .iter()
        .filter(|&&(_, _, upload)| upload == Upload::Whole)
        .map(|&(id, _, _)| id)
        .chain(
            moves
                .changed
                .iter()
                .copied()
                .filter(|id| !atlas.slots.contains_key(id)),
        );
    for id in shaped {
        plan.stale |= plan.marches((&tiles, &atlas), id);
    }
    let toward = Vec3::from_slice(&sun.direction[..3]);
    let level = Vec2::new(toward.x, toward.z);
    let finest = held(&tiles, &atlas)
        .map(|index| tiles.terrains[index].pyramid.spacing(0))
        .fold(f32::INFINITY, engine_math::min);
    let main = views
        .main()
        .filter(|_| settings.sun_shadows && level.length_squared() > 1.0e-8 && finest.is_finite());
    let mut records = [[0.0; 4]; SUN_STATE as usize + 1];
    let Some(main) = main else {
        plan.forget();
        plan.key = None;
        if piece.sun != records {
            piece.sun = records;
        }
        return;
    };
    let (count, _) = crate::terrain_horizons::room(limits.texture_layers, tiles.terrains.len());
    let slots = crate::terrain_bounce::relit_slots(&crate::terrain_bounce::slots(
        &tiles.terrains,
        &settings,
        limits.texture_layers,
    ));
    let (sine, cosine) = engine_math::sin_cos(sky.sun_size.to_radians());
    let spread = engine_math::max(sine / cosine, 1.0e-6);
    let key = Key {
        side: settings.shadow_resolution,
        base: crate::terrain_horizons::sun_base(count, slots),
        levels: settings.sun_levels.min(SUN_LEVELS),
        texel: finest * settings.sun_texel,
        cone: 2.0 * spread * settings.sun_sample_spread,
        tall: settings.sun_tall,
        slack: spread * settings.sun_march_slack,
        steps: settings.sun_march_steps,
    };
    if plan.key != Some(key) || plan.dropped {
        plan.forget();
        plan.key = Some(key);
    }
    let now = time.real_elapsed;
    if let Some(build) = plan.build.take_if(|build| plan.marched == Some(build.id)) {
        plan.slide = Some((now, build.windows));
    }
    let mut blend = 0.0;
    if let Some((began, windows)) = plan.slide.take() {
        let seconds = f64::from(settings.sun_slide_seconds);
        if seconds > 0.0 && now - began < seconds {
            blend = ((now - began) / seconds) as f32;
            plan.slide = Some((began, windows));
        } else {
            for (level, window) in windows {
                plan.shown[level as usize] = Some(window);
            }
        }
    }
    if plan.build.is_none() && plan.slide.is_none() {
        let eye = Vec2::new(main.detail_eye.x, main.detail_eye.z);
        let windows = plan.wanted(&key, eye, level.normalize());
        if !windows.is_empty() {
            plan.builds = plan.builds.wrapping_add(1);
            plan.stale = 0;
            plan.build = Some(SunBuild {
                id: plan.builds,
                windows,
                base: key.base,
                cone: key.cone,
                tall: key.tall,
                slack: key.slack,
                steps: key.steps,
            });
        }
    }
    for level in 0..key.levels {
        let shown = plan.shown[level as usize];
        let next = plan
            .slide
            .as_ref()
            .and_then(|(_, windows)| windows.iter().find(|(built, _)| *built == level))
            .map(|(_, window)| *window)
            .or(shown);
        records[2 * level as usize] = record(key.base, level, key.side, shown.as_ref());
        records[2 * level as usize + 1] = record(key.base, level, key.side, next.as_ref());
    }
    records[SUN_STATE as usize] = [blend, key.tall, key.levels as f32, spread];
    if piece.sun != records {
        piece.sun = records;
    }
}

pub(crate) struct SunGpu {
    pass: Pass<TerrainGpu>,
    pipeline: ComputePipeline<TerrainGpu>,
    jobs: Buffer<TerrainGpu>,
    terrains: Buffer<TerrainGpu>,
    counts: Buffer<TerrainGpu>,
    build: Option<u32>,
    at: (usize, u32),
    issued: Option<(usize, u32)>,
    told: bool,
    counted: bool,
    recorded: bool,
    groups: [u32; 3],
}

impl SunGpu {
    pub(crate) fn install(
        cx: &mut Install<'_, TerrainGpu>,
        horizons: &Horizons,
        pass: Pass<TerrainGpu>,
    ) -> Result<Self, Error> {
        use crate::shaders::terrain_shading::sun_march;
        let jobs = cx.buffer("sun_jobs", Lifetime::Kept);
        let terrains = cx.buffer("sun_terrains", Lifetime::Kept);
        let counts = cx.buffer("sun_counts", Lifetime::Kept);
        let pipeline = cx.compute_pipeline(
            pass,
            sun_march::SUN_MARCH,
            &[
                (sun_march::SUN_JOBS, Bind::Buffer(jobs)),
                (sun_march::SUN_TERRAINS, Bind::Buffer(terrains)),
                (sun_march::SUN_OUT, Bind::Texture(horizons.texture)),
                (sun_march::SUN_COUNTS, Bind::Buffer(counts)),
            ],
        )?;
        Ok(Self {
            pass,
            pipeline,
            jobs,
            terrains,
            counts,
            build: None,
            at: (0, 0),
            issued: None,
            told: false,
            counted: false,
            recorded: false,
            groups: [0; 3],
        })
    }

    pub(crate) fn rest(&mut self) {
        self.build = None;
        self.issued = None;
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (plan, settings, tiles, atlas): (&SunPlan, &TerrainShading, &TerrainTiles, &Atlas),
        (horizons, peaks_missed): (&Horizons, bool),
    ) {
        if let Some(before) = self.issued.take()
            && (peaks_missed || !(cx.ran(self.pass) && self.recorded))
        {
            self.at = before;
        }
        if horizons.fresh {
            cx.deliver::<SunPlan>(DROPPED, Vec::new());
            self.build = None;
        }
        let Some(build) = &plan.build else {
            self.build = None;
            return;
        };
        if self.build != Some(build.id) {
            self.build = Some(build.id);
            self.at = (0, 0);
            self.told = false;
            self.counted = false;
        }
        let side = settings.shadow_resolution;
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.counts, COUNT_BYTES);
        if self.at.0 >= build.windows.len() {
            if !self.told {
                let walked = (build.windows.len() as u32).saturating_mul(side * side);
                cx.deliver::<SunPlan>(MARCHED, bytemuck::bytes_of(&[build.id, walked]).to_vec());
                self.told = true;
            }
            if !self.counted {
                self.counted =
                    cx.read_back::<SunPlan>(COUNTED, self.counts, 0..8, AfterRead::Keep);
            }
            return;
        }
        if self.at == (0, 0) {
            cx.write(self.counts, 0, &[0_u32; COUNT_BYTES as usize / 4]);
        }
        let terrains: Vec<SunTerrain> = held(tiles, atlas)
            .map(|index| {
                let entry = &tiles.terrains[index];
                let top = entry
                    .tiles
                    .iter()
                    .map(|tile| tile.bounds.1.y)
                    .fold(f32::NEG_INFINITY, engine_math::max);
                SunTerrain {
                    cutout: entry
                        .cutout
                        .map_or([0.0; 4], |(low, high)| [low.x, low.y, high.x, high.y]),
                    shape: [top, 0.0, 0.0, 0.0],
                    table: [index as u32, 0, 0, 0],
                }
            })
            .collect();
        if terrains.is_empty() {
            return;
        }
        let slice = settings.sun_rows.min(side.next_multiple_of(SUN_GROUP));
        let most = (settings.sun_rows / slice).max(1) as usize;
        let before = self.at;
        let mut jobs = Vec::with_capacity(most);
        for _ in 0..most {
            let Some((level, window)) = build.windows.get(self.at.0) else {
                break;
            };
            jobs.push(SunJob {
                place: [window.origin.x, window.origin.y, window.texel, build.tall],
                heading: [window.heading.x, window.heading.y, build.cone, build.slack],
                rows: [
                    self.at.1,
                    layer(build.base, *level, window),
                    build.steps,
                    terrains.len() as u32,
                ],
            });
            self.at.1 += slice;
            if self.at.1 >= side {
                self.at = (self.at.0 + 1, 0);
            }
        }
        let bytes = |count: usize, size: usize| ((count * size) as u64).next_power_of_two();
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.jobs, bytes(most, size_of::<SunJob>()));
        cx.write(self.jobs, 0, &jobs);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.terrains, bytes(terrains.len(), size_of::<SunTerrain>()));
        cx.write(self.terrains, 0, &terrains);
        self.groups = [
            side.div_ceil(SUN_GROUP),
            slice / SUN_GROUP,
            jobs.len() as u32,
        ];
        self.issued = Some(before);
        cx.run(self.pass);
    }

    pub(crate) fn encode(&mut self, cx: &mut Encode<'_, TerrainGpu>) -> u32 {
        let dispatched = cx.dispatch(&self.pipeline, Count::Groups(self.groups));
        self.recorded = dispatched > 0;
        dispatched
    }
}
