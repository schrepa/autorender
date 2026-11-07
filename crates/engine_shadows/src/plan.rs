use std::any::TypeId;

use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, Vec3};
use engine_core::{
    Cursor, Cut, DeviceLimits, FrameBudget, FrameTime, GpuBits, Mobility, PartDraws, PassName,
    PrimaryView, Readable, Readback, SceneView, SceneViews, Since, Spawner, ViewWind, gpu_bits,
};
use engine_places_rendering::shadows::{
    CasterMap, GROUND_TEXELS, Refresh, Region as MapView, ShadowChange, ShadowChanges,
    ShadowMaps,
};
use engine_places_rendering::sun::Sun;
use engine_render::{CameraUniform, DebugShown, pixel_angle};

use crate::atlas::MapsKey;
use crate::constants::LEVEL_WORDS;
use crate::gpu_layouts::{CasterCull, ComposeData, Route, ShadowPiece};
use crate::part::{SHADOW_FORMAT, ShadowsGpu};
use crate::regions::FrameRegions;
use crate::shadow_fit::{ClipPlan, ClipSettings, Clipmap, DrawnTile, LightBasis, region_tiles};
use crate::sky::{RESOLUTION, SkyDraw, SkyMaps};
use crate::{ShadowPass, ShadowSettings, ShadowStats, ShadowsBrick, SkyOcclusion};

const BUDGET: &str = "shadow_tiles";
const TILE_WORDS: usize = 3;
pub(crate) const SKIPPED: u8 = 0;
pub(crate) const LOST: u8 = 1;
pub(crate) const SKY_LOST: u8 = 2;

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MarkKind {
    Map = 0,
    Depth = 1,
    Masks = 2,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MarkTag {
    #[bits(2)]
    pub kind: MarkKind,
    #[bits(8)]
    pub layout: u8,
}

#[derive(Resource, Default)]
pub(crate) struct ShadowFeedback {
    arrived: Vec<(MarkTag, Vec<u32>)>,
}

impl Readable for ShadowFeedback {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>) {
        let Some(tag) = MarkTag::unpack(tag) else {
            return;
        };
        let words = read
            .value
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        self.arrived.push((tag, words));
    }
}

#[derive(Resource, Default)]
pub(crate) struct ShadowDrawn {
    acks: Vec<(u32, u8, Vec<DrawnTile>)>,
}

impl Readable for ShadowDrawn {
    fn arrived(&mut self, layout: u32, read: Readback<&[u8]>) {
        let Some((&status, tiles)) = read.value.split_first() else {
            return;
        };
        let words: Vec<u32> = tiles
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let tiles = words
            .as_chunks::<TILE_WORDS>()
            .0
            .iter()
            .map(|&[level, cell, slot]| (level as usize, cell as usize, slot as u16))
            .collect();
        self.acks.push((layout, status, tiles));
    }
}

#[derive(Default)]
struct Masks {
    marked: Vec<u32>,
    dirty: Vec<u32>,
    moving: Vec<u32>,
}

impl ShadowFeedback {
    fn take(&mut self, layout: u8, words: usize, into: &mut Masks) -> [bool; 3] {
        let mut came = [false; 3];
        for read in [&mut into.marked, &mut into.dirty, &mut into.moving] {
            read.clear();
        }
        for (tag, arrived) in self.arrived.drain(..) {
            if tag.layout != layout {
                continue;
            }
            let (read, at) = match tag.kind {
                MarkKind::Map => (&mut into.dirty, 0),
                MarkKind::Depth => (&mut into.marked, 1),
                MarkKind::Masks => (&mut into.moving, 2),
            };
            came[at] = true;
            read.resize(words, 0);
            for (word, bits) in read.iter_mut().zip(arrived) {
                *word |= bits;
            }
        }
        came
    }
}

#[derive(Resource, Default)]
pub(crate) struct ShadowPlan {
    clipmap: Clipmap,
    plan: ClipPlan,
    changes: Cursor,
    pub(crate) key: Option<MapsKey>,
    pub(crate) layout: u8,
    pub(crate) drawn: Vec<DrawnTile>,
    masks: Masks,
    pub(crate) sky: SkyMaps,
    pub(crate) sky_draw: Option<SkyDraw>,
    sky_view: Option<(Mat4, [f32; 4])>,
    pub(crate) views: Vec<MapView>,
    pub(crate) cameras: Vec<CameraUniform>,
    pub(crate) routes: Vec<Route>,
    pub(crate) levels: Vec<[u32; LEVEL_WORDS as usize]>,
    pub(crate) cull: CasterCull,
    pub(crate) composes: Vec<ComposeData>,
    pub(crate) moving: u32,
    pub(crate) table: Vec<u32>,
    pub(crate) revisions: Vec<u64>,
    pub(crate) mask_words: u32,
}

impl ShadowPlan {
    pub(crate) fn clipmap(&self) -> &Clipmap {
        &self.clipmap
    }

    pub(crate) fn side(&self) -> u32 {
        self.clipmap.side()
    }

    pub(crate) fn mark_bytes(&self) -> u64 {
        u64::from(self.mask_words) * 4
    }

    pub(crate) fn tag(&self, kind: MarkKind) -> u32 {
        MarkTag {
            kind,
            layout: self.layout,
        }
        .pack()
    }

    pub(crate) fn drawn_bytes(&self, status: u8) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(1 + self.drawn.len() * TILE_WORDS * 4);
        bytes.push(status);
        for &(level, cell, slot) in &self.drawn {
            for word in [level as u32, cell as u32, u32::from(slot)] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        bytes
    }

    fn place(
        &mut self,
        settings: &ShadowSettings,
        limit: u32,
        (sun, eye): (Vec3, Vec3),
    ) -> LightBasis {
        let key = MapsKey::new(settings, limit);
        let made = self.key != Some(key);
        self.key = Some(key);
        let (basis, restarted) = self.clipmap.place(
            LightBasis::toward(sun),
            eye,
            ClipSettings {
                levels: key.levels,
                finest_texel: settings.finest_texel,
                tile_texels: settings.tile_texels,
                slots: key.atlas.still,
            },
        );
        if made && !restarted {
            self.clipmap.invalidate(&[]);
        }
        if made || restarted {
            self.layout = self.layout.wrapping_add(1);
        }
        basis
    }

    fn acknowledge(&mut self, drawn: &mut ShadowDrawn) {
        for (layout, status, tiles) in drawn.acks.drain(..) {
            if layout != u32::from(self.layout) {
                continue;
            }
            match status {
                LOST => self.clipmap.invalidate(&tiles),
                SKY_LOST => self.sky.forget(),
                _ => self.clipmap.undraw(&tiles),
            }
        }
    }

    fn idle(&mut self) {
        self.clipmap.idle();
        self.sky_draw = None;
        self.drawn.clear();
        self.views.clear();
        self.cameras.clear();
        self.routes.clear();
        self.composes.clear();
        self.moving = 0;
        self.mask_words = 0;
    }

    fn merge(&mut self, feedback: &mut ShadowFeedback) -> bool {
        let words = self.clipmap.mask_words();
        let [_, marked, moving] = feedback.take(self.layout, words, &mut self.masks);
        let still_changed = self.masks.dirty.iter().any(|&word| word != 0);
        if marked {
            self.clipmap.merge_marks(&self.masks.marked);
        }
        self.clipmap.merge_dirty(&self.masks.dirty);
        if moving {
            self.clipmap.merge_moving(&self.masks.moving);
        }
        self.mask_words = words as u32;
        still_changed
    }

    fn change(&mut self, change: &ShadowChange) -> bool {
        match *change {
            ShadowChange::Box { min, max, spacing } => {
                for level in 0..self.clipmap.len() {
                    if spacing >= GROUND_TEXELS * self.clipmap.texel(level) {
                        self.clipmap.dirty_level_box(level, min, max);
                    }
                }
                false
            }
            ShadowChange::All => {
                self.clipmap.dirty_all();
                false
            }
            ShadowChange::Casters => true,
        }
    }

    fn table(&mut self) {
        let (table, changed) = self.clipmap.take_table();
        self.table.clear();
        self.table.extend_from_slice(table);
        self.revisions.resize(self.clipmap.len(), 0);
        for level in changed {
            if let Some(revision) = self.revisions.get_mut(level) {
                *revision += 1;
            }
        }
    }
}

#[derive(Component)]
pub(crate) struct ShadowViews;

fn scene_views(maps: &ShadowMaps, key: Option<MapsKey>, eye: Vec3) -> SceneViews {
    let views = key.map_or_else(Vec::new, |key| {
        maps.regions
            .iter()
            .map(|region| {
                let (planes, texel) = match region.refresh {
                    Refresh::Kept(kept) => (Some(kept.planes), kept.texel),
                    Refresh::Redraw => (None, 0.0),
                };
                let (side, takes) = match region.map {
                    CasterMap::Region => (key.region_side(), Mobility::Static),
                    CasterMap::Moving => (key.atlas.slot, Mobility::Moving),
                    CasterMap::Sky => (RESOLUTION, Mobility::Static),
                };
                SceneView {
                    view_projection: region.view.to_cols_array_2d(),
                    planes,
                    eye: eye.to_array(),
                    size: [side, side],
                    detail: 0.5 * texel,
                    takes: Some(takes),
                }
            })
            .collect()
    });
    SceneViews {
        pass: ShadowPass::NAME,
        views,
    }
}

fn publish_views(
    views: SceneViews,
    published: &mut Query<'_, '_, &mut SceneViews, With<ShadowViews>>,
    spawner: &mut Spawner<'_, '_, ShadowsBrick>,
) {
    match published.single_mut() {
        Ok(mut held) => {
            held.set_if_neq(views);
        }
        Err(_) => {
            spawner.spawn((ShadowViews, views));
        }
    }
}

pub(crate) fn plan_shadows(
    (settings, sky): (Res<ShadowSettings>, Res<SkyOcclusion>),
    (sun, view, limits, time): (Res<Sun>, Res<PrimaryView>, Res<DeviceLimits>, Res<FrameTime>),
    (changes, wind, shown): (Res<ShadowChanges>, Res<ViewWind>, Res<DebugShown>),
    (mut feedback, mut drawn, mut budget): (
        ResMut<ShadowFeedback>,
        ResMut<ShadowDrawn>,
        ResMut<FrameBudget>,
    ),
    (mut plan, mut piece): (ResMut<ShadowPlan>, ResMut<ShadowPiece>),
    (mut maps, mut published, mut spawner): (
        ResMut<ShadowMaps>,
        Query<&mut SceneViews, With<ShadowViews>>,
        Spawner<ShadowsBrick>,
    ),
) {
    let plan = &mut *plan;
    let eye = view.detail_eye();
    let sun = Vec3::from_slice(&sun.direction[..3]);
    if sun == Vec3::ZERO {
        plan.idle();
        feedback.arrived.clear();
        drawn.acks.clear();
        piece.set_if_neq(ShadowPiece::default());
        if maps.on {
            maps.on = false;
            maps.regions.clear();
        }
        publish_views(scene_views(&maps, None, eye), &mut published, &mut spawner);
        return;
    }
    let basis = plan.place(&settings, limits.max_texture_dimension_2d, (sun, eye));
    plan.acknowledge(&mut drawn);
    let still_changed = plan.merge(&mut feedback);
    let mut casters_changed = false;
    match changes.since(&mut plan.changes) {
        Since::Same => {}
        Since::Listed(listed) => {
            for change in listed {
                casters_changed |= plan.change(change);
            }
        }
        Since::All => {
            plan.clipmap.dirty_all();
            casters_changed = true;
        }
    }
    let Some(key) = plan.key else {
        return;
    };
    let texel_bytes = SHADOW_FORMAT.block_copy_size(None).map_or(0, u64::from);
    let tile_bytes = u64::from(key.atlas.slot).pow(2) * texel_bytes;
    let mut afford = || budget.spend(BUDGET, tile_bytes);
    plan.clipmap.plan(
        (&mut afford, region_tiles(settings.tile_texels)),
        key.atlas.moving,
        &mut plan.plan,
    );
    plan.drawn = plan.clipmap.take_drawing();
    plan.sky_draw = plan
        .sky
        .prepare(&sky, eye, still_changed || casters_changed);
    let gusting = [
        wind.velocity[0],
        wind.velocity[1],
        wind.gusts,
        time.elapsed as f32,
    ];
    if let Some(draw) = plan.sky_draw
        && plan.sky_view.is_none_or(|(clip, _)| clip != draw.clip)
    {
        plan.sky_view = Some((draw.clip, gusting));
    }
    let regions = FrameRegions {
        basis,
        clipmap: &plan.clipmap,
        plan: &plan.plan,
        layout: key.atlas,
    };
    let viewer = eye.extend(1.0 / pixel_angle(view.view_projection(), view.display_size().y));
    regions.views(plan.sky_draw.as_ref(), &mut plan.views);
    regions.cameras(
        (wind.velocity, gusting),
        (viewer, plan.sky_view),
        &mut plan.cameras,
    );
    regions.routes(&mut plan.routes);
    regions.levels(&mut plan.levels);
    regions.composes(settings.tile_texels, &mut plan.composes);
    plan.cull = regions.cull(plan.sky_draw.as_ref());
    let open = [
        sky.strength,
        sky.floor,
        0.0,
        if sky.darkens() { 1.0 } else { 0.0 },
    ];
    piece.set_if_neq(regions.lookup_piece(
        settings.tile_texels,
        (crate::debug::shown(*shown), open),
    ));
    plan.moving = plan.plan.moving.len() as u32;
    plan.table();
    if maps.regions != plan.views || maps.levels != key.levels || !maps.on {
        let maps = &mut *maps;
        maps.regions.clear();
        maps.regions.extend_from_slice(&plan.views);
        (maps.levels, maps.on) = (key.levels, true);
    }
    publish_views(scene_views(&maps, Some(key), eye), &mut published, &mut spawner);
}

pub(crate) fn shadow_stats(
    (settings, plan): (Res<ShadowSettings>, Res<ShadowPlan>),
    draws: Res<PartDraws>,
    mut stats: ResMut<ShadowStats>,
) {
    let tiles = plan.clipmap.stats();
    let (sky_maps, sky_drawn) = plan.sky.counts();
    let (levels, moving) = plan
        .key
        .map_or((0, 0), |key| (key.levels, key.atlas.moving));
    stats.set_if_neq(ShadowStats {
        active: true,
        draws: draws.of(TypeId::of::<ShadowsGpu>()),
        tiles_marked: tiles.marked,
        tiles_resident: tiles.resident,
        tiles_drawn: tiles.drawn,
        tiles_reclaimed: tiles.reclaimed,
        tiles_waiting: tiles.waiting,
        sky_maps,
        sky_drawn,
        tile_cut: tiles.tile_cut,
        moving_cut: tiles.moving_cut,
        level_cut: Cut::of(settings.levels, levels),
        moving_slot_cut: Cut::of(settings.moving_tiles, moving),
    });
}
