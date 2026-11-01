use bevy_ecs::prelude::*;
use engine_core::{
    Cut, Expected, GlobalTransforms, InvalidSetting, Nearest, Poses, Rule, Setting, View, fields,
    settings, stats,
};
use engine_places::lights::{
    LightRegion, LocalLight, MAX_LIGHT_RANGE, MAX_SPOT_DEGREES, MIN_SPOT_DEGREES, Spot,
};
use engine_places::meshes::Instances;
use glam::{Mat4, Vec3};

use crate::part::LightTotals;

settings! {
    #[section("local_lights")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneLocalLights")]
    pub struct LocalLightSettings {
        pub enabled: bool = true => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_lights: Option<u32> = None => budget,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub lights_per_cluster: Option<u32> = None => budget,
        pub intensity: f32 = 1.0 => range(0.0, 100.0),
        pub distance: f32 = 250.0 => range(1.0, 100_000.0),
    }
}

impl LocalLightSettings {
    pub fn active(&self) -> bool {
        self.enabled && self.intensity > 0.0
    }
}

stats! {
    #[section("local_light_stats")]
    pub struct LightStats {
        pub shaded: u32,
        pub invalid: u32,
        pub visible: Option<u32>,
        pub light_cut: Cut,
        pub cluster_cut: Cut,
    }
}

pub(crate) fn light_rule(field: &str) -> Option<Rule> {
    match field {
        "color" => Some(fields::at_least(0.0)),
        "intensity" => Some(fields::above(0.0)),
        "range" => Some(fields::above_to(0.0, MAX_LIGHT_RANGE)),
        "offset" => Some(fields::range(-1.0e6, 1.0e6)),
        "radius" | "length" => Some(fields::range(0.0, MAX_LIGHT_RANGE)),
        _ => None,
    }
}

pub(crate) fn spot_rule(field: &str) -> Option<Rule> {
    match field {
        "direction" => Some(fields::finite()),
        "inner_angle" => Some(fields::range(0.0, MAX_SPOT_DEGREES.to_radians())),
        "outer_angle" => Some(fields::range(
            MIN_SPOT_DEGREES.to_radians(),
            MAX_SPOT_DEGREES.to_radians(),
        )),
        _ => None,
    }
}

pub(crate) fn spot_order(
    direction: Vec3,
    inner_angle: f32,
    outer_angle: f32,
) -> Result<(), InvalidSetting> {
    if direction.length_squared() == 0.0 {
        return Err(InvalidSetting::new("direction", Expected::Nonzero).got(direction.to_array()));
    }
    if outer_angle < inner_angle {
        return Err(InvalidSetting::new("outer_angle", Expected::Sorted)
            .got(outer_angle)
            .nearest(inner_angle));
    }
    Ok(())
}

fn ruled<T: Setting>(value: &T, field: &'static str, rule: Option<Rule>) -> bool {
    Setting::check(value, field, fields::same(rule)).is_ok()
}

fn valid(light: &LocalLight) -> bool {
    ruled(&light.color, "color", light_rule("color"))
        && ruled(&light.intensity, "intensity", light_rule("intensity"))
        && ruled(&light.range, "range", light_rule("range"))
        && ruled(&light.offset, "offset", light_rule("offset"))
        && light.spot.is_none_or(|spot| valid_spot(&spot))
        && ruled(&light.radius, "radius", light_rule("radius"))
        && ruled(&light.length, "length", light_rule("length"))
}

fn valid_spot(spot: &Spot) -> bool {
    ruled(&spot.direction, "direction", spot_rule("direction"))
        && ruled(&spot.inner_angle, "inner_angle", spot_rule("inner_angle"))
        && ruled(&spot.outer_angle, "outer_angle", spot_rule("outer_angle"))
        && spot_order(spot.direction, spot.inner_angle, spot.outer_angle).is_ok()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RenderLight {
    pub(crate) position: Vec3,
    pub(crate) range: f32,
    pub(crate) color: Vec3,
    pub(crate) direction: Vec3,
    pub(crate) cone: [f32; 2],
    pub(crate) shape: [f32; 2],
    pub(crate) region: LightRegion,
    pub(crate) slot: Option<u32>,
}

impl RenderLight {
    pub(crate) const POINT: [f32; 2] = [-2.0, -2.0];

    fn new(light: &LocalLight, scale: f32, model: Option<Mat4>, slot: Option<u32>) -> Self {
        let (position, direction) = match light.spot {
            Some(spot) => (light.offset, spot.direction),
            None if light.length > 0.0 => (light.offset, Vec3::Y),
            None => (light.offset, Vec3::ZERO),
        };
        let (position, direction, size) = match model {
            Some(model) => (
                model.transform_point3(position),
                model.transform_vector3(direction),
                model.transform_vector3(Vec3::Y).length(),
            ),
            None => (position, direction, 1.0),
        };
        let cone = light.spot.map_or(Self::POINT, |spot| {
            let outer = engine_math::clamp(spot.outer_angle, 1.0e-3, 1.55);
            let inner = engine_math::clamp(spot.inner_angle, 0.0, outer);
            [
                engine_math::max(engine_math::cos(inner), engine_math::cos(outer) + 1.0e-4),
                engine_math::cos(outer),
            ]
        });
        let length = if light.spot.is_some() {
            0.0
        } else {
            light.length
        };
        Self {
            position,
            range: light.range,
            color: light.color * (light.intensity * scale),
            direction: direction.normalize_or_zero(),
            cone,
            shape: [light.radius * size, length * size],
            region: light.region,
            slot,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct LightList {
    pub(crate) lights: Vec<RenderLight>,
}

#[derive(Resource, Default)]
pub(crate) struct LightGather {
    scanned: bool,
    count: usize,
    settings: Option<LocalLightSettings>,
    lights: Vec<(Entity, LocalLight, Option<u32>)>,
    world_lights: Vec<usize>,
    invalid: usize,
    ranked: Vec<(f32, usize)>,
    nearest: Nearest,
    cut: usize,
    shown_invalid: usize,
}

impl LightGather {
    fn rescan(
        &mut self,
        lights: &Query<'_, '_, (Entity, Ref<'static, LocalLight>)>,
        instances: &Instances,
    ) {
        self.lights.clear();
        self.world_lights.clear();
        self.invalid = 0;
        for (entity, light) in lights {
            if !valid(&light) {
                self.invalid += 1;
                continue;
            }
            let slot = instances.drawn(entity);
            if slot.is_none() {
                self.world_lights.push(self.lights.len());
            }
            self.lights.push((entity, *light, slot));
        }
    }

    fn publish(
        &mut self,
        settings: &LocalLightSettings,
        (globals, poses): (&GlobalTransforms, &Poses),
        eye: Vec3,
        list: &mut Vec<RenderLight>,
    ) {
        let budget = settings
            .max_lights
            .map_or(usize::MAX, |budget| budget as usize);
        list.clear();
        if self.lights.len() > budget {
            self.publish_nearest(settings, (globals, poses), eye, budget, list);
            return;
        }
        let Self {
            lights,
            shown_invalid,
            ..
        } = self;
        for (entity, light, slot) in lights.iter() {
            let model = match slot {
                Some(_) => None,
                None => match poses.of(*entity) {
                    Some(model) => Some(Mat4::from(model)),
                    None => {
                        *shown_invalid += 1;
                        continue;
                    }
                },
            };
            list.push(RenderLight::new(light, settings.intensity, model, *slot));
        }
    }

    fn publish_nearest(
        &mut self,
        settings: &LocalLightSettings,
        (globals, poses): (&GlobalTransforms, &Poses),
        eye: Vec3,
        budget: usize,
        list: &mut Vec<RenderLight>,
    ) {
        let Self {
            lights,
            ranked,
            nearest,
            cut,
            shown_invalid,
            ..
        } = self;
        ranked.clear();
        for (index, (entity, light, _)) in lights.iter().enumerate() {
            let Some(model) = globals.get(*entity).and_then(|pose| pose.matrix()) else {
                continue;
            };
            let distance = model.transform_point3(light.offset).distance_squared(eye)
                - light.range * light.range;
            ranked.push((distance, index));
        }
        let kept = nearest.kept_in_order(ranked.iter().map(|&(distance, _)| distance), budget);
        *cut = ranked.len() - kept.len();
        *shown_invalid += lights.len() - ranked.len();
        for position in kept {
            let (entity, light, slot) = &lights[ranked[position].1];
            let model = match slot {
                Some(_) => None,
                None => match poses.of(*entity) {
                    Some(model) => Some(Mat4::from(model)),
                    None => {
                        *shown_invalid += 1;
                        continue;
                    }
                },
            };
            list.push(RenderLight::new(light, settings.intensity, model, *slot));
        }
    }
}

pub(crate) fn gather_lights(
    settings: Res<LocalLightSettings>,
    lights: Query<(Entity, Ref<'static, LocalLight>)>,
    instances: Res<Instances>,
    (globals, poses): (Res<GlobalTransforms>, Res<Poses>),
    view: Res<View>,
    mut gather: ResMut<LightGather>,
    mut list: ResMut<LightList>,
) {
    let gather = &mut *gather;
    gather.cut = 0;
    gather.shown_invalid = 0;
    let mut count = 0;
    let mut changed = !gather.scanned
        || instances.is_changed()
        || gather.settings != Some(*settings);
    for (_, light) in &lights {
        count += 1;
        changed |= light.is_changed();
    }
    changed |= count != gather.count;
    if changed {
        gather.rescan(&lights, &instances);
        gather.count = count;
        gather.settings = Some(*settings);
        gather.scanned = true;
    } else if gather.world_lights.is_empty()
        && settings
            .max_lights
            .is_none_or(|budget| gather.lights.len() <= budget as usize)
    {
        gather.shown_invalid = gather.invalid;
        return;
    }
    gather.shown_invalid = gather.invalid;
    gather.publish(&settings, (&globals, &poses), view.detail_eye(), &mut list.lights);
}

pub(crate) fn light_stats(
    list: Res<LightList>,
    gather: Res<LightGather>,
    view: Res<View>,
    mut totals: ResMut<LightTotals>,
    mut stats: ResMut<LightStats>,
) {
    let listed = !list.lights.is_empty();
    if !listed && !view.debug() {
        totals.forget();
    }
    let lit = listed && !view.debug();
    let found = totals.found().filter(|_| lit);
    let kept = list.lights.len().min(u32::MAX as usize) as u32;
    let cut = gather.cut.min(u32::MAX as usize) as u32;
    stats.set_if_neq(LightStats {
        shaded: if lit { list.lights.len() as u32 } else { 0 },
        invalid: gather.shown_invalid.min(u32::MAX as usize) as u32,
        visible: found.map(|(reached, _)| reached),
        light_cut: Cut::of(kept.saturating_add(cut), kept),
        cluster_cut: found.map_or_else(Cut::default, |(_, cut)| cut),
    });
}

pub(crate) fn forget_lights(
    mut gather: ResMut<LightGather>,
    mut list: ResMut<LightList>,
    mut totals: ResMut<LightTotals>,
    mut stats: ResMut<LightStats>,
) {
    *gather = LightGather::default();
    list.lights.clear();
    totals.forget();
    *stats = LightStats::default();
}
