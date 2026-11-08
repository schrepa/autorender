use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Query, Res, Resource, World};
use bevy_ecs::system::{Commands, SystemParam, SystemParamItem};
use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    AssetLibrary, Camera, Clear, ColorSpace, DRAWN_UNBOUNDED, Detail, Drawn, DrawnPoses, DrawnTable, Error,
    ErrorCode, FrameFacts, FrameId, FrameTime, GlobalTransforms, Interest, Jitter, Layers, List,
    Mobility, Operation, OwnedView, OwnedViews, PassBit, PassMask, PassViews, Requests,
    ScenePasses, Target, View, ViewCut, ViewHistory, ViewJitter, ViewLayer, take_asks,
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
    pub clear_color: [f32; 4],
    pub order: i32,
    pub lens: Camera,
    pub eye: Vec3,
    pub detail_eye: Vec3,
    pub forward: Vec3,
    pub view_projection: Mat4,
    pub previous_view_projection: Mat4,
    pub previous_eye: Vec3,
    pub history: ViewHistory,
    pub jitter: Option<Vec2>,
    pub noise: f32,
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
            "clear_color": self.clear_color,
            "order": self.order,
            "eye": self.eye.to_array(),
            "detail_eye": self.detail_eye.to_array(),
            "forward": self.forward.to_array(),
            "history": self.history,
            "jitter": self.jitter.map(|jitter| jitter.to_array()),
            "noise": self.noise,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawnView {
    pub owner: Option<&'static str>,
    pub local: u32,
    pub view_projection: Mat4,
    pub planes: Option<[[f32; 4]; 6]>,
    pub position: Vec3,
    pub eye: Vec3,
    pub size: UVec2,
    pub detail: f32,
    pub takes: Option<Mobility>,
    pub layers: Layers,
    pub target: ViewLayer,
    pub clear: Clear,
    pub capture: bool,
    pub passes: PassMask,
    pub draws: bool,
}

#[derive(Clone, Copy, PartialEq)]
struct Key {
    target: Target,
    pixels: [u32; 4],
    render: UVec2,
    camera: Entity,
    cut: Option<u64>,
    jittered: bool,
}

#[derive(Clone, Copy)]
struct Held {
    view: Entity,
    key: Key,
    started: FrameId,
    view_projection: Mat4,
    eye: Vec3,
}

#[derive(Resource, Default)]
pub struct Views {
    listed: Vec<ViewFacts>,
    refused: Vec<(Entity, Error)>,
    held: Vec<Held>,
    drawn: Vec<DrawnView>,
    owned: Vec<(&'static str, u32, u32)>,
    frame: FrameFacts,
    frozen: Option<(Entity, Vec3)>,
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

    pub fn frame(&self) -> FrameFacts {
        self.frame
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

fn history_of(
    before: Option<&Held>,
    reset: bool,
    (frame, view_projection, eye): (FrameId, Mat4, Vec3),
) -> (FrameId, Mat4, Vec3, ViewHistory) {
    match before {
        Some(held) if !reset => (
            held.started,
            held.view_projection,
            held.eye,
            ViewHistory::Valid {
                since: held.started,
            },
        ),
        Some(_) | None => (frame, view_projection, eye, ViewHistory::Fresh),
    }
}

pub(crate) fn settle(world: &mut World, facts: FrameFacts) {
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
    let (held, frozen) = world
        .get_resource::<Views>()
        .map(|views| (views.held.clone(), views.frozen))
        .unwrap_or_default();
    let (globals, poses) = (
        world.get_resource::<GlobalTransforms>(),
        world.get_resource::<DrawnPoses>(),
    );
    let assets = world.get_resource::<AssetLibrary>();
    let given: Vec<ViewJitter> = world
        .get_resource::<Jitter>()
        .map(|jitter| jitter.views.clone())
        .unwrap_or_default();
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
        let offset = given.iter().find(|given| given.view == entity);
        let jitter = offset
            .filter(|_| !view.steady)
            .map(|given| Vec2::from(given.offset));
        let noise = offset.map_or(0.0, |given| given.noise);
        let cut = globals
            .zip(poses)
            .and_then(|(globals, poses)| poses.fresh_at(globals, view.camera));
        let key = Key {
            target: view.target,
            pixels,
            render,
            camera: view.camera,
            cut,
            jittered: jitter.is_some(),
        };
        let before = held.iter().find(|held| held.view == entity);
        let reset = cuts.contains(&entity) || before.is_none_or(|held| held.key != key);
        let (started, previous, previous_eye, history) =
            history_of(before, reset, (frame, view_projection, eye));
        kept.push(Held {
            view: entity,
            key,
            started,
            view_projection,
            eye,
        });
        let detail_eye = frozen
            .filter(|(view, _)| *view == entity)
            .map_or(eye, |(_, at)| at);
        listed.push(ViewFacts {
            view: entity,
            camera: view.camera,
            target: view.target,
            pixels,
            render,
            layers: view.layers,
            clear: view.clear,
            clear_color: view.clear_color,
            order: view.order,
            lens,
            eye,
            detail_eye,
            forward,
            view_projection,
            previous_view_projection: previous,
            previous_eye,
            history,
            jitter,
            noise,
            frame,
        });
    }
    world.insert_resource(Views {
        listed,
        refused,
        held: kept,
        drawn: Vec::new(),
        owned: Vec::new(),
        frame: facts,
        frozen,
    });
}

pub(crate) fn settle_owned(world: &mut World) {
    let detail = world
        .get_resource::<Detail>()
        .filter(|detail| detail.enabled)
        .map_or(0.0, |detail| detail.pixel_error);
    let mut published: Vec<OwnedViews> = world
        .try_query::<&OwnedViews>()
        .map(|mut query| query.iter(world).cloned().collect())
        .unwrap_or_default();
    published.sort_by_key(|views| views.pass);
    let passes = world.get_resource::<ScenePasses>();
    let every = passes.map_or(PassMask::NONE, |passes| {
        passes
            .listed()
            .iter()
            .filter(|listed| listed.pass.views != PassViews::Owned)
            .filter_map(|listed| listed.bit)
            .fold(PassMask::NONE, |mask, bit| mask.set(bit, true))
    });
    let placed: Vec<Vec<DrawnView>> = published
        .iter()
        .map(|listed| {
            let owner = passes
                .and_then(|passes| passes.bit(listed.pass))
                .map_or(PassMask::NONE, PassBit::mask);
            (0_u32..)
                .zip(&listed.views)
                .map(|(local, view)| owned_view(world, (listed.pass, local, owner), view))
                .collect()
        })
        .collect();
    let Some(mut views) = world.get_resource_mut::<Views>() else {
        return;
    };
    let mut drawn: Vec<DrawnView> = views
        .listed
        .iter()
        .enumerate()
        .map(|(index, facts)| DrawnView {
            owner: None,
            local: index as u32,
            view_projection: facts.view_projection,
            planes: None,
            position: facts.eye,
            eye: facts.detail_eye,
            size: facts.render,
            detail,
            takes: None,
            layers: facts.layers,
            target: ViewLayer {
                layer: 0,
                rect: facts.pixels,
            },
            clear: facts.clear,
            capture: false,
            passes: every,
            draws: true,
        })
        .collect();
    let mut owned = Vec::with_capacity(published.len());
    for (listed, placed) in published.iter().zip(placed) {
        owned.push((listed.pass, drawn.len() as u32, placed.len() as u32));
        drawn.extend(placed);
    }
    if views.drawn != drawn || views.owned != owned {
        views.drawn = drawn;
        views.owned = owned;
    }
}

fn owned_view(
    world: &World,
    (owner, local, pass): (&'static str, u32, PassMask),
    view: &OwnedView,
) -> DrawnView {
    let [_, _, width, height] = view.target.rect;
    let aspect = width.max(1) as f32 / height.max(1) as f32;
    let lens = world.get::<Camera>(view.camera).copied();
    let pose = world
        .get::<crate::Transform>(view.camera)
        .map(crate::Transform::matrix)
        .and_then(eye_of);
    let placed = lens
        .zip(pose)
        .map(|(lens, (position, forward, up))| {
            (
                lens.clip(aspect) * Mat4::look_to_rh(position, forward, up),
                position,
            )
        })
        .filter(|(view_projection, _)| view_projection.is_finite());
    let (view_projection, position) = placed.unwrap_or((Mat4::ZERO, Vec3::ZERO));
    DrawnView {
        owner: Some(owner),
        local,
        view_projection,
        planes: view.planes,
        position,
        eye: Vec3::from_array(view.eye),
        size: UVec2::new(width, height),
        detail: view.detail,
        takes: view.takes,
        layers: Layers::ALL,
        target: view.target,
        clear: view.clear,
        capture: view.capture,
        passes: view.passes.unwrap_or(pass),
        draws: view.draws && placed.is_some(),
    }
}

#[derive(Resource, Debug, Default)]
pub struct WholeFrames {
    requests: Requests<&'static str, ()>,
}

impl List for WholeFrames {
    const NAME: &'static str = "whole_frames";
    type Member = &'static str;
}

impl WholeFrames {
    pub fn ask(&mut self, by: &'static str) -> Interest<&'static str> {
        let interest = self.requests.ask(by);
        self.end();
        interest
    }

    pub fn asked(&mut self) -> bool {
        self.requests.settle(0.0);
        self.end();
        !self.requests.is_empty()
    }

    fn end(&mut self) {
        let asked: Vec<(&'static str, crate::Ticket)> = self
            .requests
            .asked()
            .map(|(&by, ticket)| (by, ticket))
            .collect();
        for (by, ticket) in asked {
            self.requests.ready(&by, ticket, ());
        }
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ViewControl {
    #[serde(default)]
    freeze: Option<bool>,
}

impl Operation for ViewControl {
    const NAME: &'static str = "view";
    type Params = (Res<'static, Views>, Commands<'static, 'static>);
    type Request = Self;

    fn run(
        (views, mut commands): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let main = views.main();
        let held = views
            .frozen
            .filter(|(view, _)| main.is_some_and(|facts| facts.view == *view));
        let frozen = match request.freeze {
            Some(true) => held.or_else(|| main.map(|facts| (facts.view, facts.eye))),
            Some(false) => None,
            None => held,
        };
        if frozen != views.frozen {
            commands.queue(move |world: &mut World| {
                if let Some(mut views) = world.get_resource_mut::<Views>() {
                    views.frozen = frozen;
                }
            });
        }
        Ok(json!({ "frozen_at": frozen.map(|(_, eye)| eye.to_array()) }))
    }
}
