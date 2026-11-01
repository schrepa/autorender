use std::any::TypeId;

use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, Vec3};
use engine_core::{Cut, DeviceLimits, FrameTime, PartDraws, Readback, View};
use engine_places::ground::Foliage;
use engine_places::shadows::{GROUND_TEXELS, Region as MapView, ShadowChanges, ShadowMaps};
use engine_places::sun::Sun;
use engine_places::wind::Wind;
use engine_render::{CameraUniform, DebugShown, pixel_angle};

use crate::atlas::MapsKey;
use crate::constants::LEVEL_WORDS;
use crate::gpu_layouts::{CasterCull, ComposeData, Route, ShadowPiece};
use crate::part::ShadowsGpu;
use crate::regions::FrameRegions;
use crate::shadow_fit::{ClipPlan, ClipSettings, Clipmap, LightBasis, region_tiles};
use crate::sky::{SkyDraw, SkyMaps};
use crate::{ShadowSettings, ShadowStats, SkyOcclusion};

pub(crate) const MAP_MARKS: u32 = 0;
pub(crate) const DEPTH_MARKS: u32 = 1;
pub(crate) const MASKS: u32 = 2;

#[derive(Resource, Default)]
pub(crate) struct ShadowFeedback {
    arrived: Vec<(u64, u32, Vec<u32>)>,
}

impl Readback for ShadowFeedback {
    fn arrived(&mut self, frame: u64, tag: u32, bytes: &[u8]) {
        let words = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        self.arrived.push((frame, tag, words));
    }
}

#[derive(Default)]
struct Masks {
    marked: Vec<u32>,
    dirty: Vec<u32>,
    moving: Vec<u32>,
}

impl ShadowFeedback {
    fn take(&mut self, since: u64, words: usize, into: &mut Masks) {
        for read in [&mut into.marked, &mut into.dirty, &mut into.moving] {
            read.clear();
        }
        for (frame, tag, arrived) in self.arrived.drain(..) {
            if frame < since {
                continue;
            }
            let read = match tag {
                MAP_MARKS => &mut into.dirty,
                DEPTH_MARKS => &mut into.marked,
                _ => &mut into.moving,
            };
            read.resize(words, 0);
            for (word, bits) in read.iter_mut().zip(arrived) {
                *word |= bits;
            }
        }
    }
}

#[derive(Resource)]
pub(crate) struct ShadowPlan {
    clipmap: Clipmap,
    plan: ClipPlan,
    pub(crate) key: Option<MapsKey>,
    pub(crate) made: u64,
    since: u64,
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

impl Default for ShadowPlan {
    fn default() -> Self {
        Self {
            clipmap: Clipmap::default(),
            plan: ClipPlan::default(),
            key: None,
            made: 0,
            since: 0,
            masks: Masks::default(),
            sky: SkyMaps::default(),
            sky_draw: None,
            sky_view: None,
            views: Vec::new(),
            cameras: Vec::new(),
            routes: Vec::new(),
            levels: Vec::new(),
            cull: CasterCull::default(),
            composes: Vec::new(),
            moving: 0,
            table: Vec::new(),
            revisions: Vec::new(),
            mask_words: 0,
        }
    }
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

    fn place(
        &mut self,
        settings: &ShadowSettings,
        limit: u32,
        (sun, eye): (Vec3, Vec3),
        frame: u64,
    ) -> LightBasis {
        let key = MapsKey::new(settings, limit);
        let made = self.key != Some(key);
        self.key = Some(key);
        self.made += u64::from(made);
        let basis = LightBasis::toward(sun);
        let restarted = self.clipmap.place(
            basis,
            eye,
            ClipSettings {
                levels: key.levels,
                finest_texel: settings.finest_texel,
                tile_texels: settings.tile_texels,
                slots: key.atlas.still,
            },
        );
        if made && !restarted {
            self.clipmap.invalidate();
        }
        if restarted || self.since == u64::MAX {
            self.since = frame;
        }
        basis
    }

    fn merge(&mut self, feedback: &mut ShadowFeedback) -> bool {
        let words = self.clipmap.mask_words();
        feedback.take(self.since, words, &mut self.masks);
        let still_changed = self.masks.dirty.iter().any(|&word| word != 0);
        self.clipmap.merge_marks(&self.masks.marked);
        self.clipmap.merge_dirty(&self.masks.dirty);
        self.clipmap.merge_moving(&self.masks.moving);
        self.mask_words = words as u32;
        still_changed
    }

    fn change(&mut self, change: &ShadowChanges) {
        if change.all {
            self.clipmap.dirty_all();
        }
        for &(min, max, spacing) in &change.boxes {
            for level in 0..self.clipmap.len() {
                if spacing >= GROUND_TEXELS * self.clipmap.texel(level) {
                    self.clipmap.dirty_level_box(level, min, max);
                }
            }
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

    pub(crate) fn forget(&mut self) {
        self.key = None;
        self.since = u64::MAX;
        self.sky.release();
        self.sky_draw = None;
        self.sky_view = None;
        self.views.clear();
    }
}

pub(crate) fn plan_shadows(
    (settings, sky): (Res<ShadowSettings>, Res<SkyOcclusion>),
    (sun, view, limits, time): (Res<Sun>, Res<View>, Res<DeviceLimits>, Res<FrameTime>),
    (foliage, wind, shown): (Res<Foliage>, Res<Wind>, Res<DebugShown>),
    mut feedback: ResMut<ShadowFeedback>,
    changes: Query<&ShadowChanges, Changed<ShadowChanges>>,
    (mut plan, mut piece): (ResMut<ShadowPlan>, ResMut<ShadowPiece>),
    mut maps: ResMut<ShadowMaps>,
) {
    let plan = &mut *plan;
    let eye = view.detail_eye();
    let published = Vec3::from_slice(&sun.direction[..3]);
    let sun = if published == Vec3::ZERO {
        Vec3::from_slice(&Sun::NONE.direction[..3])
    } else {
        published
    };
    let basis = plan.place(&settings, limits.max_texture_dimension_2d, (sun, eye), time.frame);
    let still_changed = plan.merge(&mut feedback);
    let mut casters_changed = false;
    for change in &changes {
        casters_changed |= change.casters;
        plan.change(change);
    }
    let Some(key) = plan.key else {
        return;
    };
    plan.clipmap.plan(
        (settings.redraw_budget, region_tiles(settings.tile_texels)),
        key.atlas.moving,
        &mut plan.plan,
    );
    plan.sky_draw = plan.sky.prepare(&sky, eye, still_changed || casters_changed);
    let gusting = [wind.velocity[0], wind.velocity[1], wind.gusts, time.elapsed_seconds as f32];
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
    regions.cameras((wind.velocity, gusting), (viewer, plan.sky_view), &mut plan.cameras);
    regions.routes(&mut plan.routes);
    regions.levels(&mut plan.levels);
    regions.composes(settings.tile_texels, &mut plan.composes);
    plan.cull = regions.cull(plan.sky_draw.as_ref());
    piece.set_if_neq(regions.lookup_piece(
        (settings.tile_texels, foliage.crown_depth),
        crate::debug::shown(*shown),
    ));
    plan.moving = plan.plan.moving.len() as u32;
    plan.table();
    let darkens = sky.darkens();
    if maps.regions != plan.views
        || maps.sky_darkens != darkens
        || maps.levels != key.levels
        || !maps.on
    {
        let maps = &mut *maps;
        maps.regions.clear();
        maps.regions.extend_from_slice(&plan.views);
        (maps.levels, maps.sky_darkens, maps.on) = (key.levels, darkens, true);
    }
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
        sky_maps,
        sky_drawn,
        tile_cut: tiles.tile_cut,
        moving_cut: tiles.moving_cut,
        level_cut: Cut::of(settings.levels, levels),
        moving_slot_cut: Cut::of(settings.moving_tiles, moving),
    });
}

pub(crate) fn forget_shadows(
    mut plan: ResMut<ShadowPlan>,
    mut feedback: ResMut<ShadowFeedback>,
    mut stats: ResMut<ShadowStats>,
) {
    plan.forget();
    feedback.arrived.clear();
    stats.set_if_neq(ShadowStats::default());
}
