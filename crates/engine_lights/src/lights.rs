use bevy_ecs::prelude::*;
use engine_core::{
    Cut, Expected, GlobalTransforms, InvalidSetting, Nearest, Rule, View, fields, settings, stats,
};
use engine_places::lights::{
    LightRegion, LocalLight, MAX_LIGHT_RANGE, MAX_SPOT_DEGREES, MIN_SPOT_DEGREES,
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
    ranked: Vec<(f32, Entity)>,
    nearest: Nearest,
    cut: usize,
}

fn render_light(
    (entity, light): (Entity, &LocalLight),
    settings: &LocalLightSettings,
    (instances, globals): (&Instances, &GlobalTransforms),
) -> Option<RenderLight> {
    let slot = instances.drawn(entity);
    let model = match slot {
        Some(_) => None,
        None => Some(Mat4::from(globals.pose(entity)?)),
    };
    Some(RenderLight::new(light, settings.intensity, model, slot))
}

pub(crate) fn gather_lights(
    settings: Res<LocalLightSettings>,
    lights: Query<(Entity, &'static LocalLight)>,
    (instances, globals): (Res<Instances>, Res<GlobalTransforms>),
    view: Res<View>,
    mut gather: ResMut<LightGather>,
    mut list: ResMut<LightList>,
) {
    let LightGather {
        ranked,
        nearest,
        cut,
    } = &mut *gather;
    *cut = 0;
    list.lights.clear();
    let held = (&*instances, &*globals);
    let budget = settings
        .max_lights
        .map_or(usize::MAX, |budget| budget as usize);
    if lights.iter().len() <= budget {
        let shown = lights
            .iter()
            .filter_map(|light| render_light(light, &settings, held));
        list.lights.extend(shown);
        return;
    }
    let eye = view.detail_eye();
    ranked.clear();
    for (entity, light) in &lights {
        let Some(model) = globals.pose(entity) else {
            continue;
        };
        let distance =
            model.transform_point3(light.offset).distance_squared(eye) - light.range * light.range;
        ranked.push((distance, entity));
    }
    let kept = nearest.kept_in_order(ranked.iter().map(|&(distance, _)| distance), budget);
    *cut = ranked.len() - kept.len();
    for position in kept {
        let entity = ranked[position].1;
        if let Ok(light) = lights.get(entity)
            && let Some(light) = render_light(light, &settings, held)
        {
            list.lights.push(light);
        }
    }
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
