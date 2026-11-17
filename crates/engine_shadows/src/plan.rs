use std::any::TypeId;
use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::math;
use engine_core::{
    Camera, Cursor, Cut, DeviceLimits, FrameTime, Mobility, Mover, Moves, Node, OwnedView,
    OwnedViews, Pace, PartDraws, PassName, Pause, PauseReason, Readable, Readback, Readiness,
    Settling, Since, Spawner, Switched, Transform, ViewDetail, ViewFacts, Views, mover_of,
};
use engine_places_rendering::shadows::{
    CasterMap, Refresh, Region as MapView, SHADOW_REGIONS, ShadowMaps,
};
use engine_places_rendering::sky::SkyLight;
use engine_places_rendering::sun::Sun;
use engine_render::DebugShown;

use crate::changes::{ShadowChange, ShadowChanges};
use crate::constants::LEVEL_SNAP;
use crate::gpu_layouts::{ComposeData, ShadowLevel, ShadowPiece};
use crate::levels::{LevelKey, LightBasis, Levels};
use crate::part::{SHADOW_FORMAT, ShadowsGpu};
use crate::regions::Placed;
use crate::sky::{SkyDraw, SkyMaps};
use crate::{ShadowPass, ShadowSettings, ShadowStats, ShadowsBrick, SkyOcclusion};

pub(crate) const SKIPPED: u8 = 0;
pub(crate) const LOST: u8 = 1;
pub(crate) const SKY_LOST: u8 = 2;
pub(crate) const SKY_STALE: u8 = 3;
pub(crate) const WAITED: u8 = 4;

#[derive(Resource, Default)]
pub(crate) struct ShadowDrawn {
    acks: Vec<(u32, u8, Vec<u32>)>,
}

impl Readable for ShadowDrawn {
    fn arrived(&mut self, layout: u32, read: Readback<&[u8]>) {
        let Some((&status, levels)) = read.value.split_first() else {
            return;
        };
        let levels = levels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        self.acks.push((layout, status, levels));
    }
}

#[derive(Resource, Default)]
pub(crate) struct ShadowPlan {
    levels: Levels,
    changes: Cursor,
    pub(crate) key: Option<LevelKey>,
    pub(crate) layout: u8,
    pub(crate) restarted: bool,
    pub(crate) preparing: bool,
    pub(crate) sky: SkyMaps,
    pub(crate) sky_draws: Vec<SkyDraw>,
    pub(crate) views: Vec<MapView>,
    placed: Vec<Placed>,
    pub(crate) level_data: Vec<ShadowLevel>,
    pub(crate) composes: Vec<ComposeData>,
    pub(crate) composed: Vec<(u32, Range<u32>)>,
    moving: u32,
    settling: Finishing,
    cameras: Vec<Entity>,
}

struct Finishing(Settling);

impl Default for Finishing {
    fn default() -> Self {
        Self(Settling::on_finished())
    }
}

impl ShadowPlan {
    pub(crate) fn levels(&self) -> &Levels {
        &self.levels
    }

    pub(crate) fn drawn_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![SKIPPED];
        for (level, _) in &self.composed {
            bytes.extend_from_slice(&level.to_le_bytes());
        }
        bytes
    }

    fn acknowledge(&mut self, drawn: &mut ShadowDrawn) {
        for (layout, status, levels) in drawn.acks.drain(..) {
            if layout != u32::from(self.layout) {
                continue;
            }
            match status {
                LOST => self.levels.forget(None),
                SKY_LOST => self.sky.forget(),
                SKY_STALE => self.sky.stale(&levels),
                WAITED => self.levels.wait(&levels),
                _ => self.levels.forget(Some(&levels)),
            }
        }
    }

    fn idle(&mut self) {
        self.levels.idle();
        self.sky.idle();
        self.sky_draws.clear();
        self.views.clear();
        self.placed.clear();
        self.composes.clear();
        self.composed.clear();
        self.moving = 0;
    }
}

fn level_key(settings: &ShadowSettings, main: &ViewFacts, limits: &DeviceLimits) -> LevelKey {
    let measure = ViewDetail::perspective(
        main.detail_eye,
        main.view_projection,
        main.size().y,
        settings.pixel_error,
    );
    let snaps = (measure.reach(4.0) / LEVEL_SNAP as f32).ceil() as u32 + 1;
    let most = limits.max_texture_dimension_2d / LEVEL_SNAP;
    let levels = settings.levels.min(limits.texture_layers).max(1);
    LevelKey {
        levels,
        moving: settings
            .moving_levels
            .min(levels)
            .min(limits.texture_layers.saturating_sub(levels)),
        side: snaps.min(most).max(1) * LEVEL_SNAP,
        finest: settings.finest_texel.to_bits(),
    }
}

const DETAIL_PIXELS: f32 = 0.5;

#[derive(Component)]
pub(crate) struct ShadowViews;

#[derive(Component)]
pub(crate) struct ShadowCamera;

type Lenses<'w, 's> = Query<'w, 's, &'static mut Camera, With<ShadowCamera>>;

fn live(node: Entity, (moves, lenses): (&Moves<'_, '_, ShadowsBrick>, &Lenses<'_, '_>)) -> bool {
    lenses.contains(node) && moves.local(node).is_some()
}

fn live_cameras(
    pool: &[Entity],
    found: (&Moves<'_, '_, ShadowsBrick>, &Lenses<'_, '_>),
) -> usize {
    pool.iter()
        .position(|&node| !live(node, found))
        .unwrap_or(usize::MAX)
}

fn place_cameras(
    placed: &[Placed],
    eye: Vec3,
    pool: &mut Vec<Entity>,
    (spawner, moves, lenses): (
        &mut Spawner<'_, '_, ShadowsBrick>,
        &mut Moves<'_, '_, ShadowsBrick>,
        &mut Lenses<'_, '_>,
    ),
) {
    for (at, placed) in placed.iter().enumerate() {
        let held = pool.get(at).copied();
        let Some(node) = held.filter(|&node| live(node, (moves, lenses))) else {
            let made = spawner.spawn((
                ShadowCamera,
                Node::default(),
                Transform::from_translation(eye),
                Camera::fitted(placed.clip, eye),
                Mover {
                    by: mover_of::<ShadowsBrick>(),
                    pace: Pace::Frame,
                },
            ));
            match pool.get_mut(at) {
                Some(held) => *held = made,
                None => pool.push(made),
            }
            continue;
        };
        if let Some(mut transform) = moves.get_mut(node)
            && transform.translation != eye
        {
            transform.translation = eye;
        }
        let origin = moves
            .local(node)
            .map_or(eye, |transform| transform.translation);
        if let Ok(mut lens) = lenses.get_mut(node) {
            lens.set_if_neq(Camera::fitted(placed.clip, origin));
        }
    }
}

fn scene_views(
    maps: &ShadowMaps,
    placed: &[Placed],
    (cameras, eye): (&[Entity], Vec3),
) -> OwnedViews {
    let views = maps
        .regions
        .iter()
        .zip(placed)
        .zip(cameras)
        .map(|((region, placed), &camera)| {
            let (planes, texel) = match region.refresh {
                Refresh::Kept(kept) => (Some(kept.planes), kept.texel),
                Refresh::Redraw => (None, 0.0),
            };
            let takes = match region.map {
                CasterMap::Region | CasterMap::Sky => Mobility::Static,
                CasterMap::Moving => Mobility::Moving,
            };
            OwnedView {
                camera,
                target: placed.target,
                clear: placed.clear,
                capture: false,
                passes: None,
                draws: true,
                planes,
                eye: eye.to_array(),
                detail: if texel > 0.0 { DETAIL_PIXELS } else { 0.0 },
                takes: Some(takes),
            }
        })
        .collect();
    OwnedViews {
        pass: ShadowPass::NAME,
        views,
    }
}

fn publish_views(
    views: OwnedViews,
    published: &mut Query<'_, '_, &mut OwnedViews, With<ShadowViews>>,
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
    (settings, sky, occluding): (
        Res<ShadowSettings>,
        Res<SkyOcclusion>,
        Switched<SkyOcclusion>,
    ),
    (sun, light, views, limits): (Res<Sun>, Res<SkyLight>, Res<Views>, Res<DeviceLimits>),
    (changes, shown, time): (Res<ShadowChanges>, Res<DebugShown>, Res<FrameTime>),
    (mut drawn, readiness, pause): (ResMut<ShadowDrawn>, Res<Readiness>, Res<Pause>),
    (mut plan, mut piece): (ResMut<ShadowPlan>, ResMut<ShadowPiece>),
    (mut maps, mut published, mut spawner): (
        ResMut<ShadowMaps>,
        Query<&mut OwnedViews, With<ShadowViews>>,
        Spawner<ShadowsBrick>,
    ),
    (mut moves, mut lenses): (Moves<ShadowsBrick>, Lenses),
) {
    let plan = &mut *plan;
    let main = views.main();
    let eye = main.map_or(Vec3::ZERO, |main| main.detail_eye);
    let sun = Vec3::from_slice(&sun.direction[..3]);
    let key = main
        .filter(|_| sun != Vec3::ZERO)
        .map(|main| level_key(&settings, main, &limits));
    let Some(key) = key else {
        plan.idle();
        drawn.acks.clear();
        piece.set_if_neq(ShadowPiece::default());
        if maps.on {
            maps.on = false;
            maps.regions.clear();
        }
        publish_views(
            scene_views(&maps, &[], (&[], eye)),
            &mut published,
            &mut spawner,
        );
        return;
    };
    plan.restarted = plan
        .levels
        .place(key, settings.finest_texel, (LightBasis::toward(sun), eye));
    if plan.restarted {
        plan.layout = plan.layout.wrapping_add(1);
    }
    plan.key = Some(key);
    plan.acknowledge(&mut drawn);
    let mut casters_changed = false;
    match changes.since(&mut plan.changes) {
        Since::Same => {}
        Since::Listed(listed) => {
            for change in listed {
                casters_changed |= *change == ShadowChange::Casters;
                plan.levels.change(change, DETAIL_PIXELS);
            }
        }
        Since::All => {
            plan.levels.change(&ShadowChange::All, DETAIL_PIXELS);
            casters_changed = true;
        }
    }
    let quiet = readiness.loading.total == 0 && readiness.compiling == 0;
    plan.preparing = pause.holds(PauseReason::Preparing);
    let held = plan.preparing && !readiness.only_finishing();
    let live = live_cameras(&plan.cameras, (&moves, &lenses)).min(SHADOW_REGIONS as usize);
    let room = if held { 0 } else { live };
    let holds = (!plan.preparing && !quiet, readiness.compiling > 0);
    plan.levels.plan(room, holds);
    let free = room.saturating_sub(plan.levels.regions.len());
    let moving = changes.moving > 0 && !held;
    plan.moving = if moving { key.moving } else { 0 };
    let occludes = occluding.on();
    plan.sky.prepare(
        (&sky, occludes),
        (eye, time.real_elapsed),
        (casters_changed, quiet, plan.preparing),
        (free, &mut plan.sky_draws),
    );
    crate::regions::views(
        &plan.levels,
        (plan.moving, key.side, settings.sample_texels),
        &plan.sky_draws,
        (&mut plan.views, &mut plan.placed),
    );
    crate::regions::composes(
        &plan.levels,
        key.side,
        (&mut plan.composes, &mut plan.composed),
    );
    crate::regions::level_data(&plan.levels, plan.moving, &mut plan.level_data);
    let (sine, cosine) = math::sin_cos(light.sun_size.to_radians());
    let spread = sine / cosine;
    let open = [
        sky.strength,
        sky.floor,
        0.0,
        if occludes { 1.0 } else { 0.0 },
    ];
    piece.set_if_neq(ShadowPiece {
        state: [
            1.0,
            settings.pixel_error,
            crate::debug::shown(*shown) as f32,
            spread,
        ],
        level_state: [
            key.levels as f32,
            settings.finest_texel,
            key.side as f32,
            0.0,
        ],
        soft: [
            settings.soft_texels,
            (0.5 * (settings.soft_texels + 1.0)).ceil(),
            0.0,
            0.0,
        ],
        sky: open,
    });
    if maps.regions != plan.views || maps.levels != key.levels || !maps.on {
        let maps = &mut *maps;
        maps.regions.clear();
        maps.regions.extend_from_slice(&plan.views);
        (maps.levels, maps.on) = (key.levels, true);
    }
    place_cameras(
        &plan.placed,
        eye,
        &mut plan.cameras,
        (&mut spawner, &mut moves, &mut lenses),
    );
    publish_views(
        scene_views(&maps, &plan.placed, (&plan.cameras, eye)),
        &mut published,
        &mut spawner,
    );
}

pub(crate) fn shadow_stats(
    (settings, plan): (Res<ShadowSettings>, Res<ShadowPlan>),
    draws: Res<PartDraws>,
    mut stats: ResMut<ShadowStats>,
) {
    let levels = &plan.levels;
    let drawn = levels.regions.len() as u32;
    plan.settling
        .0
        .left((levels.wanted - drawn).saturating_add(plan.sky.left()));
    let (sky_maps, sky_drawn) = plan.sky.counts();
    let (held, side, bytes) = plan.key.map_or((0, 0, 0), |key| {
        let texel = SHADOW_FORMAT.block_copy_size(None).map_or(0, u64::from);
        let (wide, regions) = crate::regions::region_shape(key.side);
        let levels = u64::from(key.side).pow(2) * u64::from(key.layers());
        let staged = u64::from(wide).pow(2) * u64::from(regions);
        (key.levels, key.side, (levels + staged) * texel)
    });
    stats.set_if_neq(ShadowStats {
        active: true,
        draws: draws.of(TypeId::of::<ShadowsGpu>()),
        levels: held,
        side,
        bytes,
        regions: Cut::of(levels.wanted, drawn),
        redraws: levels.redraws,
        moving: plan.moving,
        sky_maps,
        sky_drawn,
        level_cut: Cut::of(settings.levels, held),
    });
}
