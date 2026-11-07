use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Query, Res, Resource, World};
use bevy_ecs::system::{SystemParam, SystemParamItem};
use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    AssetLibrary, Camera, Clear, ColorSpace, DRAWN_UNBOUNDED, Detail, Drawn, DrawnPoses, DrawnTable, Error,
    ErrorCode, FrameFacts, FrameId, FrameTime, GlobalTransforms, Jitter, Layers, Mobility, Operation,
    OwnedViews, PrimaryView, Target, View, ViewCut, ViewHistory, take_asks,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(&self, distance: f32) -> Vec3 {
        self.origin + self.direction * distance
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewFacts {
    pub view: Entity,
    pub camera: Entity,
    pub target: Target,
    pub pixels: [u32; 4],
    pub render: UVec2,
    pub layers: Layers,
    pub clear: Clear,
    pub order: i32,
    pub lens: Camera,
    pub eye: Vec3,
    pub forward: Vec3,
    pub view_projection: Mat4,
    pub previous_view_projection: Mat4,
    pub history: ViewHistory,
    pub debug: bool,
    pub jitter: Option<Vec2>,
    pub capture: bool,
    pub frame: FrameId,
}

impl ViewFacts {
    pub fn size(&self) -> UVec2 {
        UVec2::new(self.pixels[2], self.pixels[3])
    }

    pub fn whole(&self, target: UVec2) -> bool {
        self.pixels == [0, 0, target.x, target.y]
    }

    pub fn aspect(&self) -> f32 {
        self.pixels[2].max(1) as f32 / self.pixels[3].max(1) as f32
    }

    pub fn covers(&self, pixel: Vec2) -> bool {
        let [x, y, width, height] = self.pixels.map(|value| value as f32);
        pixel.x >= x && pixel.y >= y && pixel.x < x + width && pixel.y < y + height
    }

    fn device(&self, pixel: Vec2) -> Vec2 {
        let [x, y, width, height] = self.pixels.map(|value| value as f32);
        Vec2::new(
            (pixel.x - x) / crate::math::max(width, 1.0) * 2.0 - 1.0,
            1.0 - (pixel.y - y) / crate::math::max(height, 1.0) * 2.0,
        )
    }

    pub fn ray(&self, pixel: Vec2) -> Option<Ray> {
        let inverse = self.view_projection.inverse();
        if !inverse.is_finite() || !pixel.is_finite() {
            return None;
        }
        let device = self.device(pixel);
        let point = |depth: f32| {
            let clip = inverse * Vec4::new(device.x, device.y, depth, 1.0);
            (clip.w.abs() > f32::EPSILON).then(|| clip.truncate() / clip.w)
        };
        let near = point(1.0)?;
        let far = point(0.0)?;
        let direction = (far - near).try_normalize()?;
        Some(Ray {
            origin: near,
            direction,
        })
    }

    pub fn pixel_of(&self, point: Vec3) -> Option<Vec2> {
        let clip = self.view_projection * point.extend(1.0);
        if clip.w <= f32::EPSILON || !clip.is_finite() {
            return None;
        }
        let device = clip.truncate() / clip.w;
        let [x, y, width, height] = self.pixels.map(|value| value as f32);
        Some(Vec2::new(
            x + (device.x * 0.5 + 0.5) * width,
            y + (0.5 - device.y * 0.5) * height,
        ))
    }

    fn json(&self) -> Value {
        let target = match self.target {
            Target::Surface => json!("surface"),
            Target::Texture(texture) => json!({ "texture": format!("{texture:?}") }),
        };
        json!({
            "view": self.view.to_bits(),
            "camera": self.camera.to_bits(),
            "target": target,
            "pixels": self.pixels,
            "render": self.render.to_array(),
            "layers": self.layers.0,
            "clear": self.clear,
            "order": self.order,
            "eye": self.eye.to_array(),
            "forward": self.forward.to_array(),
            "history": self.history,
            "debug": self.debug,
            "jitter": self.jitter.map(|jitter| jitter.to_array()),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawnView {
    pub owner: Option<&'static str>,
    pub view_projection: Mat4,
    pub planes: Option<[[f32; 4]; 6]>,
    pub eye: Vec3,
    pub size: UVec2,
    pub detail: f32,
    pub takes: Option<Mobility>,
    pub layers: Layers,
}

#[derive(Clone, Copy, PartialEq)]
struct Key {
    target: Target,
    pixels: [u32; 4],
    render: UVec2,
    camera: Entity,
    cut: Option<u64>,
    scene: u64,
    debug: bool,
    jittered: bool,
}

#[derive(Clone, Copy)]
struct Held {
    view: Entity,
    key: Key,
    started: FrameId,
    view_projection: Mat4,
}

#[derive(Resource, Default)]
pub struct Views {
    listed: Vec<ViewFacts>,
    refused: Vec<(Entity, Error)>,
    held: Vec<Held>,
    drawn: Vec<DrawnView>,
    owned: Vec<(&'static str, u32, u32)>,
}

impl Views {
    pub fn listed(&self) -> &[ViewFacts] {
        &self.listed
    }

    pub fn drawn(&self) -> &[DrawnView] {
        &self.drawn
    }

    pub fn owned(&self, pass: &str) -> Option<std::ops::Range<u32>> {
        self.owned
            .iter()
            .find(|(known, _, _)| *known == pass)
            .map(|&(_, base, count)| base..base + count)
    }

    pub fn main(&self) -> Option<&ViewFacts> {
        self.listed.first()
    }

    pub fn get(&self, view: Entity) -> Option<&ViewFacts> {
        self.listed.iter().find(|facts| facts.view == view)
    }

    pub fn at(&self, pixel: Vec2) -> Option<&ViewFacts> {
        self.listed
            .iter()
            .filter(|facts| facts.target.surface() && facts.covers(pixel))
            .max_by_key(|facts| facts.order)
    }

    pub fn refused(&self) -> &[(Entity, Error)] {
        &self.refused
    }

    fn json(&self) -> Value {
        let listed: Vec<Value> = self.listed.iter().map(ViewFacts::json).collect();
        let refused: Vec<Value> = self
            .refused
            .iter()
            .map(|(view, error)| json!({ "view": view.to_bits(), "error": error }))
            .collect();
        json!({ "views": listed, "refused": refused })
    }
}

pub fn main_camera(world: &World) -> Option<Entity> {
    let mut views = world.try_query::<(Entity, &View)>()?;
    views
        .iter(world)
        .min_by_key(|(entity, view)| (!view.target.surface(), view.order, *entity))
        .map(|(_, view)| view.camera)
}

fn refuse(view: Entity, field: &str, code: ErrorCode) -> Error {
    Error::new(code, format!("$.views.{}.{field}", view.to_bits()))
}

fn target_size(assets: Option<&AssetLibrary>, target: Target, surface: UVec2) -> Option<UVec2> {
    match target {
        Target::Surface => Some(surface.max(UVec2::ONE)),
        Target::Texture(texture) => assets?
            .texture(texture)
            .filter(|texture| texture.is_target() && texture.color_space() == ColorSpace::Srgb)
            .map(|texture| UVec2::new(texture.width(), texture.height()).max(UVec2::ONE)),
    }
}

fn rendered(pixels: [u32; 4], target: Target, facts: FrameFacts) -> UVec2 {
    let size = UVec2::new(pixels[2], pixels[3]);
    if !target.surface() || facts.render_size == facts.display_size {
        return size;
    }
    let share = facts.render_size.as_vec2() / facts.display_size.max(UVec2::ONE).as_vec2();
    (size.as_vec2() * share)
        .round()
        .as_uvec2()
        .clamp(UVec2::ONE, size.max(UVec2::ONE))
}

fn eye_of(pose: Mat4) -> Option<(Vec3, Vec3, Vec3)> {
    let eye = pose.w_axis.truncate();
    let forward = pose.transform_vector3(Vec3::NEG_Z).try_normalize()?;
    let up = pose.transform_vector3(Vec3::Y).try_normalize()?;
    eye.is_finite().then_some((eye, forward, up))
}

pub(crate) fn settle(world: &mut World, facts: FrameFacts, scene: u64) {
    let cuts: Vec<Entity> = take_asks::<ViewCut>(world)
        .into_iter()
        .map(|cut| cut.view)
        .collect();
    let frame = FrameId(world.get_resource::<FrameTime>().map_or(0, |time| time.frame));
    let mut found: Vec<(Entity, View)> = world
        .try_query::<(Entity, &View)>()
        .map(|mut views| views.iter(world).map(|(entity, view)| (entity, *view)).collect())
        .unwrap_or_default();
    found.sort_by_key(|(entity, view)| (!view.target.surface(), view.order, *entity));
    let held = world
        .get_resource::<Views>()
        .map(|views| views.held.clone())
        .unwrap_or_default();
    let (globals, poses) = (
        world.get_resource::<GlobalTransforms>(),
        world.get_resource::<DrawnPoses>(),
    );
    let assets = world.get_resource::<AssetLibrary>();
    let offset = world.get_resource::<Jitter>().and_then(|jitter| jitter.offset);
    let mut listed = Vec::with_capacity(found.len());
    let mut refused = Vec::new();
    let mut kept = Vec::with_capacity(found.len());
    for (entity, view) in found {
        let Some(lens) = world.get::<Camera>(view.camera).copied() else {
            refused.push((entity, refuse(entity, "camera", ErrorCode::MissingComponent)));
            continue;
        };
        if let Err(error) = lens.check() {
            refused.push((entity, error.within(&format!("$.views.{}.lens", entity.to_bits()))));
            continue;
        }
        let path = format!("$.views.{}.viewport", entity.to_bits());
        if let Err(error) = view.viewport.check(&path) {
            refused.push((entity, error));
            continue;
        }
        let pose = globals
            .zip(poses)
            .and_then(|(globals, poses)| poses.drawn_pose(globals, view.camera));
        let Some((eye, forward, up)) = pose.and_then(eye_of) else {
            refused.push((entity, refuse(entity, "camera", ErrorCode::InvalidCamera)));
            continue;
        };
        let Some(size) = target_size(assets, view.target, facts.display_size) else {
            refused.push((entity, refuse(entity, "target", ErrorCode::InvalidTexture)));
            continue;
        };
        let pixels = view.viewport.pixels((size.x, size.y));
        let render = rendered(pixels, view.target, facts);
        let aspect = pixels[2].max(1) as f32 / pixels[3].max(1) as f32;
        let view_projection = lens.clip(aspect) * Mat4::look_to_rh(eye, forward, up);
        if !view_projection.is_finite() {
            refused.push((entity, refuse(entity, "camera", ErrorCode::InvalidCamera)));
            continue;
        }
        let main = listed.is_empty() && view.target.surface();
        let debug = main && facts.debug;
        let jitter = offset
            .filter(|_| view.target.surface() && !debug)
            .map(Vec2::from);
        let cut = globals
            .zip(poses)
            .and_then(|(globals, poses)| poses.fresh_at(globals, view.camera));
        let key = Key {
            target: view.target,
            pixels,
            render,
            camera: view.camera,
            cut,
            scene,
            debug,
            jittered: jitter.is_some(),
        };
        let before = held.iter().find(|held| held.view == entity);
        let reset = cuts.contains(&entity) || before.is_none_or(|held| held.key != key);
        let (started, previous, history) = match before {
            Some(held) if !reset => (
                held.started,
                held.view_projection,
                ViewHistory::Valid {
                    since: held.started,
                },
            ),
            Some(_) | None => (frame, view_projection, ViewHistory::Fresh),
        };
        kept.push(Held {
            view: entity,
            key,
            started,
            view_projection,
        });
        listed.push(ViewFacts {
            view: entity,
            camera: view.camera,
            target: view.target,
            pixels,
            render,
            layers: view.layers,
            clear: view.clear,
            order: view.order,
            lens,
            eye,
            forward,
            view_projection,
            previous_view_projection: previous,
            history,
            debug,
            jitter,
            capture: main && facts.capture,
            frame,
        });
    }
    world.insert_resource(Views {
        listed,
        refused,
        held: kept,
        drawn: Vec::new(),
        owned: Vec::new(),
    });
}

pub(crate) fn settle_owned(world: &mut World) {
    let detail = world
        .get_resource::<Detail>()
        .filter(|detail| detail.enabled)
        .map_or(0.0, |detail| detail.pixel_error);
    let detail_eye = world.get_resource::<PrimaryView>().map(PrimaryView::detail_eye);
    let mut published: Vec<OwnedViews> = world
        .try_query::<&OwnedViews>()
        .map(|mut query| query.iter(world).cloned().collect())
        .unwrap_or_default();
    published.sort_by_key(|views| views.pass);
    let Some(mut views) = world.get_resource_mut::<Views>() else {
        return;
    };
    let mut drawn: Vec<DrawnView> = views
        .listed
        .iter()
        .enumerate()
        .map(|(index, facts)| DrawnView {
            owner: None,
            view_projection: facts.view_projection,
            planes: None,
            eye: detail_eye.filter(|_| index == 0).unwrap_or(facts.eye),
            size: facts.render,
            detail,
            takes: None,
            layers: facts.layers,
        })
        .collect();
    let mut owned = Vec::with_capacity(published.len());
    for listed in &published {
        let base = drawn.len() as u32;
        drawn.extend(listed.views.iter().map(|view| DrawnView {
            owner: Some(listed.pass),
            view_projection: Mat4::from_cols_array_2d(&view.view_projection),
            planes: view.planes,
            eye: Vec3::from_array(view.eye),
            size: UVec2::from_array(view.size),
            detail: view.detail,
            takes: view.takes,
            layers: Layers::ALL,
        }));
        owned.push((listed.pass, base, listed.views.len() as u32));
    }
    if views.drawn != drawn || views.owned != owned {
        views.drawn = drawn;
        views.owned = owned;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub drawn: Entity,
    pub node: Entity,
    pub distance: f32,
}

#[derive(SystemParam)]
pub struct Picking<'w, 's> {
    table: Res<'w, DrawnTable>,
    drawn: Query<'w, 's, &'static Drawn>,
    poses: Res<'w, DrawnPoses>,
    globals: Res<'w, GlobalTransforms>,
}

fn slab(origin: Vec3, direction: Vec3, low: Vec3, high: Vec3) -> Option<f32> {
    let inverse = direction.recip();
    let first = (low - origin) * inverse;
    let second = (high - origin) * inverse;
    let near = first.min(second).max_element();
    let far = first.max(second).min_element();
    let near = crate::math::max(near, 0.0);
    (far >= near && far.is_finite()).then_some(near)
}

impl Picking<'_, '_> {
    pub fn ray(&self, ray: Ray, layers: Layers) -> Vec<Hit> {
        let mut hits = Vec::new();
        for (slot, record) in self.table.records().iter().enumerate() {
            let Some(owner) = self.table.owner(slot as u32) else {
                continue;
            };
            if record.flags & DRAWN_UNBOUNDED != 0 {
                continue;
            }
            let Ok(drawn) = self.drawn.get(owner) else {
                continue;
            };
            if !drawn.layers.overlaps(layers) {
                continue;
            }
            let Some(pose) = self.poses.drawn_pose(&self.globals, drawn.node) else {
                continue;
            };
            let inverse = pose.inverse();
            if !inverse.is_finite() {
                continue;
            }
            let origin = inverse.transform_point3(ray.origin);
            let direction = inverse.transform_vector3(ray.direction);
            let low = Vec4::from_array(record.bounds_min).truncate();
            let high = Vec4::from_array(record.bounds_max).truncate();
            if let Some(distance) = slab(origin, direction, low, high) {
                hits.push(Hit {
                    drawn: owner,
                    node: drawn.node,
                    distance,
                });
            }
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        hits
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ViewsRequest {}

pub(crate) struct ViewsState;

impl Operation for ViewsState {
    const NAME: &'static str = "views";
    const CONTROL: bool = false;
    type Params = Option<Res<'static, Views>>;
    type Request = ViewsRequest;

    fn run(views: SystemParamItem<'_, '_, Self::Params>, _: ViewsRequest) -> Result<Value, Error> {
        Ok(views.map_or_else(|| json!({ "views": [], "refused": [] }), |views| views.json()))
    }
}
