//! The metro: the wild city's plan (a downtown core and secondary centers,
//! midtown, residential streets, industry, parks, plazas, markets,
//! construction sites, stadiums, a river and an elevated rail ring) built
//! with every engine feature: glTF materials with normal, metal-roughness,
//! emissive and occlusion maps over procedural textures,
//! optionally BC7-compressed; grime through second texture
//! coordinates and trees and people in vertex colors;
//! alpha-masked foliage, fences and lattices with masked shadows; glass that
//! refracts and blended panes; per-object tints
//! and billboards whose materials are replaced in place;
//! keyframed props and skinned pedestrians;
//! street lamps, headlights, neon and floodlights, fountains,
//! steam, sparks and a few smoking stacks.
//!
//! Static renderables fill 20-unit blocks between 4-unit streets outward from
//! the center, as in the wild city. With `hierarchy`, each block's parts are
//! children of a block node and code-driven assemblies hang from moving
//! pivots; without it those are roots. Animated models (keyframed props and
//! people) always keep their own node trees, as imported glTF models do.

mod animated;
mod bc7;
mod materials;
mod models;
mod people;
mod props;
mod textures;
mod world;

use engine::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use engine::prelude::*;

use crate::kit::StressConfig;
use crate::kit::field::position_for_index;
use crate::kit::motion::{Animation, Motion, Orbit, Spin, Travel, Wheel};
use crate::kit::plan::{HALF_BLOCK, PERIOD, budget, compose, part_at, river_row, street_loop};
use crate::kit::random::{Rng, hash};
use animated::animated_prop;
use materials::{AD_SLOTS, Clip, Library, Mat, NEON_COLORS};
use models::{Car, MID_FLOORS, Mid, Model, Style, TOWER_FLOORS};
pub(crate) use world::MetroWorld;

const ROAD_TOP: f32 = 0.15;
/// Top of sidewalks and lots.
const LOT: f32 = 0.3;
const RAIL_BASE: f32 = 9.5;
const DECK_TOP: f32 = 10.4;
const WATER_TOP: f32 = -0.2;
/// Static renderables of an average block, for sizing districts.
const PER_BLOCK: f32 = 42.0;
/// Reflection probes: the height they capture from, and the
/// world distance inside their boxes' faces over which they fade out.
const PROBE_HEIGHT: f32 = 2.5;
const PROBE_BLEND: f32 = 2.0;
/// Scales of people (built in meters) and small street furniture.
const PERSON: f32 = 0.47;
const FURNITURE: f32 = 0.5;
/// Seconds each billboard shows an advertisement.
pub(crate) const AD_PERIOD: f64 = 6.0;

const WHITE: [f32; 4] = [1.0; 4];
const GLASS_TINTS: [[f32; 4]; 6] = [
    [0.7, 0.85, 1.0, 1.0],
    [0.6, 0.95, 0.9, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.95, 0.8, 0.6, 1.0],
    [0.8, 0.75, 1.0, 1.0],
    [0.55, 0.7, 0.9, 1.0],
];
const STONE_TINTS: [[f32; 4]; 6] = [
    [1.0, 0.95, 0.88, 1.0],
    [0.95, 0.8, 0.7, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.85, 0.85, 0.82, 1.0],
    [1.0, 0.88, 0.75, 1.0],
    [0.9, 0.92, 1.0, 1.0],
];
const PASTELS: [[f32; 4]; 8] = [
    [1.0, 0.75, 0.7, 1.0],
    [0.7, 0.85, 1.0, 1.0],
    [0.75, 1.0, 0.75, 1.0],
    [1.0, 0.95, 0.65, 1.0],
    [0.85, 0.75, 1.0, 1.0],
    [1.0, 0.85, 0.6, 1.0],
    [0.95, 0.95, 0.95, 1.0],
    [0.7, 0.95, 0.95, 1.0],
];
const ROOF_TINTS: [[f32; 4]; 5] = [
    [1.0, 0.8, 0.75, 1.0],
    [0.45, 0.5, 0.65, 1.0],
    [0.8, 0.55, 0.5, 1.0],
    [0.5, 0.7, 0.55, 1.0],
    [0.35, 0.35, 0.38, 1.0],
];
const PAINTS: [[f32; 4]; 12] = [
    [0.85, 0.08, 0.08, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.08, 0.08, 0.1, 1.0],
    [0.1, 0.3, 0.85, 1.0],
    [1.0, 0.75, 0.08, 1.0],
    [0.15, 0.6, 0.3, 1.0],
    [0.6, 0.62, 0.66, 1.0],
    [1.0, 0.4, 0.08, 1.0],
    [0.4, 0.15, 0.6, 1.0],
    [0.1, 0.7, 0.8, 1.0],
    [0.95, 0.9, 0.8, 1.0],
    [0.3, 0.1, 0.12, 1.0],
];
const CANVAS: [[f32; 4]; 6] = [
    [1.0, 0.25, 0.25, 1.0],
    [0.25, 0.55, 1.0, 1.0],
    [0.25, 0.8, 0.4, 1.0],
    [1.0, 0.75, 0.2, 1.0],
    [0.7, 0.35, 0.9, 1.0],
    [1.0, 1.0, 1.0, 1.0],
];
const CRANE: [f32; 4] = [1.0, 0.72, 0.1, 1.0];

/// A renderable's model, material and tint.
#[derive(Clone, Copy, Debug)]
struct Look {
    model: Model,
    mat: Mat,
    tint: [f32; 4],
}

fn look(model: Model, mat: Mat, tint: [f32; 4]) -> Look {
    Look { model, mat, tint }
}

#[derive(Debug)]
struct Node {
    parent: Option<u32>,
    transform: Transform,
    look: Option<Look>,
    motion: Option<Motion>,
    moving: bool,
}

/// Components beyond a node's transform, motion and mesh.
#[derive(Clone, Debug)]
enum Extra {
    Light(LocalLight),
    Emitter(ParticleEmitter),
    /// Plays `clip` on the `targets` nodes.
    Player {
        clip: Clip,
        targets: Vec<u32>,
        time: f32,
        speed: f32,
    },
    /// Deforms this node's mesh by the `joints` nodes.
    Skin(Vec<u32>),
    /// A node players move: presented smoothly with interpolation.
    Animated,
    /// A billboard face whose material cycles.
    Advert,
}

/// Where parts attach: a transform-only node (with `hierarchy`) or the
/// world, through `frame`, which is never scaled.
#[derive(Clone, Copy)]
struct Anchor {
    node: Option<u32>,
    frame: Transform,
}

#[derive(Clone, Copy)]
struct Site {
    anchor: Anchor,
    at: Vec3,
    size: f32,
    yaw: f32,
}

/// Animated models blocks offer to the movers.
#[derive(Clone, Copy)]
enum Prop {
    Flag,
    Dish,
    Trivision,
    Door,
    Turbine,
    Excavator,
    Carousel,
    Drawbridge,
    Ring,
    /// A window washer on a tower face of height class `n`.
    Gondola(u8),
}

#[derive(Clone, Copy)]
struct Rooftop {
    anchor: Anchor,
    base: Vec3,
    height: f32,
}

struct Metro {
    seed: u32,
    hierarchy: bool,
    nodes: Vec<Node>,
    extras: Vec<(u32, Extra)>,
    static_left: u32,
    moving_left: u32,
    /// Build-space size and center of every model placed.
    shapes: HashMap<Model, (Vec3, Vec3)>,
    blocks: Vec<(i32, i32, f32)>,
    rooftops: Vec<Rooftop>,
    cranes: Vec<Site>,
    wheels: Vec<Site>,
    props: Vec<(Prop, Site)>,
    /// Places people stand about at.
    spots: Vec<(Anchor, Vec3)>,
    river: Option<(i32, f32)>,
    rail: Option<i32>,
    radius: f32,
    /// Skyline centers in block units: position, reach and peak intensity.
    centers: Vec<(Vec2, f32, f32)>,
}

impl Metro {
    fn shape(&mut self, model: Model) -> (Vec3, Vec3) {
        *self.shapes.entry(model).or_insert_with(|| {
            let (mesh, center) = model.assemble().expect("every metro model builds");
            let bounds = mesh.bounds();
            (bounds.max() - bounds.min(), center)
        })
    }

    fn size(&mut self, model: Model) -> Vec3 {
        self.shape(model).0
    }

    /// How tall the block at (`bx`, `bz`) builds, in `[0, 1]`.
    fn skyline(&self, bx: i32, bz: i32) -> f32 {
        let at = Vec2::new(bx as f32, bz as f32);
        self.centers
            .iter()
            .fold(0.0, |tallest, &(center, reach, peak)| {
                math::max(tallest, peak * (1.0 - at.distance(center) / reach))
            })
    }

    fn world(&self) -> Anchor {
        Anchor {
            node: None,
            frame: Transform::default(),
        }
    }

    fn anchor(&mut self, frame: Transform) -> Anchor {
        if self.hierarchy {
            self.nodes.push(Node {
                parent: None,
                transform: frame,
                look: None,
                motion: None,
                moving: false,
            });
            Anchor {
                node: Some(self.nodes.len() as u32 - 1),
                frame,
            }
        } else {
            Anchor { node: None, frame }
        }
    }

    /// A node in `anchor`'s frame (local, or composed into world space).
    fn push(
        &mut self,
        anchor: &Anchor,
        local: Transform,
        look: Option<Look>,
        motion: Option<Motion>,
        moving: bool,
    ) -> u32 {
        self.nodes.push(Node {
            parent: anchor.node,
            transform: if anchor.node.is_some() {
                local
            } else {
                compose(&anchor.frame, &local)
            },
            look,
            motion,
            moving,
        });
        self.nodes.len() as u32 - 1
    }

    /// A static part; `None` once the static budget is spent.
    fn part(&mut self, anchor: &Anchor, local: Transform, look: Look) -> Option<u32> {
        if self.static_left == 0 {
            return None;
        }
        self.static_left -= 1;
        Some(self.push(anchor, local, Some(look), None, false))
    }

    /// A static part standing on `base`, turned by `yaw` and scaled.
    fn put(
        &mut self,
        anchor: &Anchor,
        look: Look,
        base: Vec3,
        yaw: f32,
        scale: Vec3,
    ) -> Option<u32> {
        let height = self.size(look.model).y * scale.y;
        self.part(
            anchor,
            Transform {
                translation: base + Vec3::Y * (height * 0.5),
                rotation: Quat::from_rotation_y(yaw),
                scale,
            },
            look,
        )
    }

    /// A moving part with its motion as a child of `anchor` (`nested`) and
    /// as a root (`flat`, in world space).
    fn mover(
        &mut self,
        anchor: &Anchor,
        local: Transform,
        look: Look,
        nested: Option<Motion>,
        flat: Option<Motion>,
    ) -> Option<u32> {
        if self.moving_left == 0 {
            return None;
        }
        self.moving_left -= 1;
        let motion = if anchor.node.is_some() { nested } else { flat };
        Some(self.push(anchor, local, Some(look), motion, true))
    }

    /// A transform-only pivot for a code-driven assembly, with hierarchies.
    fn pivot(&mut self, frame: Transform, motion: Motion) -> Option<Anchor> {
        let anchor = self.anchor(frame);
        let node = anchor.node?;
        self.nodes[node as usize].motion = Some(motion);
        Some(anchor)
    }

    /// The root node of an animated model, kept in every layout.
    fn model_root(&mut self, anchor: &Anchor, local: Transform, motion: Option<Motion>) -> Anchor {
        let node = self.push(anchor, local, None, motion, false);
        Anchor {
            node: Some(node),
            frame: local,
        }
    }

    /// A moving renderable (or, without a look, a node) under a model root.
    fn animated(&mut self, root: &Anchor, local: Transform, look: Option<Look>) -> Option<u32> {
        if look.is_some() {
            if self.moving_left == 0 {
                return None;
            }
            self.moving_left -= 1;
        }
        let node = self.push(root, local, look, None, look.is_some());
        self.extras.push((node, Extra::Animated));
        Some(node)
    }

    fn extra(&mut self, node: Option<u32>, extra: Extra) {
        if let Some(node) = node {
            self.extras.push((node, extra));
        }
    }

    /// `point` of `model` (in its build space) relative to its center.
    fn offset(&mut self, model: Model, point: Vec3) -> Vec3 {
        point - self.shape(model).1
    }
}

fn plan(config: &StressConfig) -> Metro {
    let (moving, static_total, radius) = budget(config, PER_BLOCK);
    let mut centers = vec![(Vec2::ZERO, radius * 0.34, 1.0)];
    let mut rng = Rng::new(config.seed ^ 0x6e7, 0, 0);
    if radius >= 6.0 {
        for _ in 0..3 + rng.below(3) {
            let (sin, cos) = math::sin_cos(rng.range(0.0, TAU));
            let distance = radius * rng.range(0.45, 0.75);
            centers.push((
                Vec2::new(cos, sin) * distance,
                radius * rng.range(0.1, 0.18),
                rng.range(0.55, 0.9),
            ));
        }
    }
    Metro {
        seed: config.seed,
        hierarchy: config.hierarchy,
        nodes: Vec::with_capacity(config.renderables as usize + 8192),
        extras: Vec::new(),
        static_left: static_total,
        moving_left: moving,
        shapes: HashMap::default(),
        blocks: Vec::new(),
        rooftops: Vec::new(),
        cranes: Vec::new(),
        wheels: Vec::new(),
        props: Vec::new(),
        spots: Vec::new(),
        river: river_row(radius).map(|row| (row, 0.0)),
        rail: (radius >= 6.0 && moving >= 400)
            .then(|| math::max((radius * 0.32).round(), 2.0) as i32),
        radius,
        centers,
    }
}

fn build(config: &StressConfig) -> Metro {
    let mut city = plan(config);
    if config.dense_colliders {
        let world = city.world();
        let crates = config.colliders.min(config.renderables - city.moving_left);
        let size = city.size(Model::Container);
        for index in 0..crates {
            let random = hash(index ^ config.seed);
            let scale = Vec3::new(
                (0.7 + (random % 5) as f32 * 0.08) / size.x,
                1.0 / size.y,
                (0.7 + ((random >> 4) % 5) as f32 * 0.08) / size.z,
            );
            city.nodes.push(Node {
                parent: world.node,
                transform: Transform {
                    translation: position_for_index(index, config),
                    rotation: Quat::from_rotation_y((random % 8) as f32 * 0.16),
                    scale,
                },
                look: Some(look(
                    Model::Container,
                    Mat::Container,
                    PAINTS[index as usize % PAINTS.len()],
                )),
                motion: None,
                moving: false,
            });
        }
    }
    let reach = (city.radius * 1.8).ceil() as i32 + 2;
    let mut order = Vec::with_capacity(((2 * reach + 1) * (2 * reach + 1)) as usize);
    for bz in -reach..=reach {
        for bx in -reach..=reach {
            let jitter = Rng::new(config.seed ^ 0x51ed, bx, bz).range(0.0, 0.9);
            order.push((((bx * bx + bz * bz) as f32).sqrt() + jitter, bx, bz));
        }
    }
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (distance, bx, bz) in order {
        if city.static_left == 0 {
            break;
        }
        let rho = distance / city.radius;
        block(&mut city, bx, bz, rho);
    }
    movers(&mut city, config);
    city
}

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

fn at(translation: Vec3) -> Transform {
    part_at(translation, Vec3::ONE)
}

fn turned(translation: Vec3, rotation: Quat, scale: Vec3) -> Transform {
    Transform {
        translation,
        rotation,
        scale,
    }
}

fn tint(color: [f32; 4], shade: f32) -> [f32; 4] {
    [color[0] * shade, color[1] * shade, color[2] * shade, 1.0]
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn block(city: &mut Metro, bx: i32, bz: i32, rho: f32) {
    let mut rng = Rng::new(city.seed, bx, bz);
    let center = v(
        bx as f32 * PERIOD + PERIOD * 0.5,
        0.0,
        bz as f32 * PERIOD + PERIOD * 0.5,
    );
    let anchor = city.anchor(at(center));
    let river = city.river.is_some_and(|(row, _)| row == bz);
    let west = v(-HALF_BLOCK - 2.0, 0.0, 0.0);
    let north = v(0.0, 0.0, -HALF_BLOCK - 2.0);
    // Streets on the west and north sides and their crossing; the west
    // street bridges the river.
    if river {
        let deck = look(Model::BridgeDeck, Mat::Road, WHITE);
        city.put(
            &anchor,
            deck,
            west + Vec3::Y * (ROAD_TOP - 0.75),
            FRAC_PI_2,
            Vec3::ONE,
        );
        if rng.chance(45) {
            city.put(
                &anchor,
                look(Model::BridgeArch, Mat::Chrome, WHITE),
                west + Vec3::Y * ROAD_TOP,
                FRAC_PI_2,
                Vec3::ONE,
            );
        }
    } else {
        city.put(
            &anchor,
            look(Model::Road, Mat::Road, WHITE),
            west,
            FRAC_PI_2,
            Vec3::ONE,
        );
    }
    city.put(
        &anchor,
        look(Model::Road, Mat::Road, WHITE),
        north,
        0.0,
        Vec3::ONE,
    );
    city.put(
        &anchor,
        look(Model::RoadCross, Mat::Crossing, WHITE),
        west + north,
        0.0,
        Vec3::ONE,
    );
    if let Some(ring) = city.rail {
        rail(city, &anchor, bx, bz, ring);
    }
    if river {
        waterfront(city, &anchor, &mut rng, bx);
        return;
    }
    city.blocks.push((bx, bz, rho));
    let skyline = city.skyline(bx, bz);
    city.put(
        &anchor,
        look(Model::Sidewalk, Mat::Pavers, WHITE),
        Vec3::ZERO,
        0.0,
        Vec3::ONE,
    );
    let ground = if rho < 0.55 || skyline > 0.0 {
        Mat::Plaza
    } else if rho < 0.85 {
        Mat::Grass
    } else {
        Mat::Asphalt
    };
    let ground_look = look(Model::Ground, ground, WHITE);
    if bx == 0 && bz == 0 {
        city.put(
            &anchor,
            look(Model::Ground, Mat::Plaza, WHITE),
            Vec3::Y * 0.1,
            0.0,
            Vec3::ONE,
        );
        landmark(city, &anchor, &mut rng);
        street(city, &anchor, &mut rng, true);
        return;
    }
    // Chances in thousandths of a park, plaza, market, construction site
    // and stadium; buildings otherwise.
    let chances = [
        if rho > 0.6 && skyline == 0.0 { 130 } else { 55 },
        if rho < 0.5 || skyline > 0.0 { 45 } else { 0 },
        if (0.2..0.85).contains(&rho) { 35 } else { 0 },
        if (0.15..0.9).contains(&rho) { 35 } else { 0 },
        if (0.3..0.8).contains(&rho) { 12 } else { 0 },
    ];
    let mut roll = rng.below(1000);
    let mut kind = chances.len();
    for (index, chance) in chances.iter().enumerate() {
        if roll < *chance {
            kind = index;
            break;
        }
        roll -= chance;
    }
    match kind {
        0 => {
            city.put(
                &anchor,
                look(Model::Ground, Mat::Grass, WHITE),
                Vec3::Y * 0.1,
                0.0,
                Vec3::ONE,
            );
            park_block(city, &anchor, &mut rng);
        }
        1 => {
            city.put(
                &anchor,
                look(Model::Ground, Mat::Plaza, WHITE),
                Vec3::Y * 0.1,
                0.0,
                Vec3::ONE,
            );
            plaza_block(city, &anchor, &mut rng);
        }
        2 => {
            city.put(
                &anchor,
                look(Model::Ground, Mat::Pavers, WHITE),
                Vec3::Y * 0.1,
                0.0,
                Vec3::ONE,
            );
            market_block(city, &anchor, &mut rng);
        }
        3 => {
            city.put(
                &anchor,
                look(Model::Ground, Mat::Dirt, WHITE),
                Vec3::Y * 0.1,
                0.0,
                Vec3::ONE,
            );
            construction_block(city, &anchor, &mut rng);
        }
        4 => {
            city.put(
                &anchor,
                look(Model::Ground, Mat::Plaza, WHITE),
                Vec3::Y * 0.1,
                0.0,
                Vec3::ONE,
            );
            stadium_block(city, &anchor, &mut rng);
        }
        _ => {
            city.put(&anchor, ground_look, Vec3::Y * 0.1, 0.0, Vec3::ONE);
            if skyline >= 0.3 {
                downtown(city, &anchor, &mut rng, skyline);
            } else if skyline > 0.0 || rho < 0.55 {
                let scale =
                    math::clamp(math::max(skyline / 0.3, 1.0 - (rho - 0.3) / 0.3), 0.15, 1.0);
                midtown(city, &anchor, &mut rng, scale);
            } else if rho < 0.85 {
                residential(city, &anchor, &mut rng);
            } else {
                industrial(city, &anchor, &mut rng);
            }
        }
    }
    street(city, &anchor, &mut rng, skyline > 0.0 || rho < 0.55);
}

/// Street furniture around a block: lamps with lights, a traffic light,
/// trees along busy streets and a few fixtures.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn street(city: &mut Metro, anchor: &Anchor, rng: &mut Rng, busy: bool) {
    let edge = HALF_BLOCK - 0.6;
    // Lamps at two corners, and halfway along busy streets, their arms
    // (along the model's +X) reaching out over the street.
    let mut lamps = vec![
        (-edge, -edge, v(-1.0, 0.0, -1.0)),
        (edge, edge, v(1.0, 0.0, 1.0)),
    ];
    if busy {
        lamps.extend([
            (0.0, -edge, v(0.0, 0.0, -1.0)),
            (-edge, 0.0, v(-1.0, 0.0, 0.0)),
        ]);
    }
    for (x, z, out) in lamps {
        let yaw = (-out.z).atan2(out.x);
        let lamp = city.put(
            anchor,
            look(Model::Lamp, Mat::Fixture, WHITE),
            v(x, LOT, z),
            yaw,
            Vec3::ONE,
        );
        let head = city.offset(Model::Lamp, v(0.9, 3.54, 0.0));
        city.extra(
            lamp,
            Extra::Light(
                LocalLight::spot(
                    Vec3::new(1.0, 0.74, 0.46),
                    26.0,
                    9.0,
                    Vec3::new(0.15, -1.0, 0.0),
                    35.0_f32.to_radians(),
                    70.0_f32.to_radians(),
                )
                .with_offset(head),
            ),
        );
    }
    city.put(
        anchor,
        look(Model::TrafficLight, Mat::Fixture, WHITE),
        v(-edge + 0.1, LOT, -edge + 1.2),
        PI * 0.75,
        Vec3::ONE,
    );
    if busy {
        for index in 0..4 {
            let along = -6.0 + index as f32 * 4.0;
            let (x, z) = if index % 2 == 0 {
                (along, edge + 0.1)
            } else {
                (edge + 0.1, along)
            };
            let tree = if rng.chance(20) {
                Model::Palm
            } else {
                Model::Tree(rng.below(3) as u8)
            };
            let mat = if tree == Model::Palm {
                Mat::Fronds
            } else {
                Mat::Leaves
            };
            let scale = rng.range(0.7, 0.9);
            city.put(
                anchor,
                look(tree, mat, WHITE),
                v(x, LOT, z),
                rng.range(0.0, TAU),
                Vec3::splat(scale),
            );
        }
    }
    // A few fixtures along the curbs.
    let fixtures = [
        Model::Bench,
        Model::Bin,
        Model::Hydrant,
        Model::Bollard,
        Model::StreetSign,
        Model::Planter,
    ];
    for _ in 0..2 + rng.below(3) {
        let model = rng.pick(&fixtures);
        let along = rng.range(-7.5, 7.5);
        let (x, z, yaw) = match rng.below(4) {
            0 => (along, -edge + 0.4, 0.0),
            1 => (along, edge - 0.4, PI),
            2 => (-edge + 0.4, along, FRAC_PI_2),
            _ => (edge - 0.4, along, -FRAC_PI_2),
        };
        let scale = if model == Model::StreetSign {
            0.8
        } else {
            FURNITURE
        };
        let base = v(x, LOT, z);
        city.put(
            anchor,
            look(model, Mat::Fixture, WHITE),
            base,
            yaw,
            Vec3::splat(scale),
        );
        if model == Model::Planter {
            city.put(
                anchor,
                look(Model::Flowers, Mat::Flora, WHITE),
                base + Vec3::Y * (0.7 * FURNITURE),
                yaw,
                Vec3::splat(FURNITURE),
            );
        }
    }
    if busy && rng.chance(30) {
        // A bus shelter with glass sides and someone waiting.
        let (x, z) = (rng.range(-5.0, 5.0), edge - 0.5);
        city.put(
            anchor,
            look(Model::ShelterFrame, Mat::Fixture, WHITE),
            v(x, LOT, z),
            PI,
            Vec3::splat(FURNITURE * 1.3),
        );
        city.put(
            anchor,
            look(Model::ShelterGlass, Mat::ClearGlass, WHITE),
            v(x, LOT + 0.13, z),
            PI,
            Vec3::splat(FURNITURE * 1.3),
        );
        city.spots.push((*anchor, v(x + 0.6, LOT, z - 0.3)));
    }
    if busy && rng.chance(12) {
        // Steam rising from a manhole.
        let manhole = city.put(
            anchor,
            look(Model::Manhole, Mat::Fixture, WHITE),
            v(rng.range(-6.0, 6.0), ROAD_TOP, -HALF_BLOCK - 1.0),
            0.0,
            Vec3::ONE,
        );
        let seed = rng.next();
        city.extra(manhole, Extra::Emitter(steam(seed)));
    }
}

fn rail(city: &mut Metro, anchor: &Anchor, bx: i32, bz: i32, ring: i32) {
    if bx.abs() > ring || bz.abs() > ring || (bx.abs() != ring && bz.abs() != ring) {
        return;
    }
    let corner = v(-HALF_BLOCK - 2.0, 0.0, -HALF_BLOCK - 2.0);
    city.put(
        anchor,
        look(Model::RailPillar, Mat::Concrete, WHITE),
        corner,
        0.0,
        Vec3::ONE,
    );
    let station = (bx + bz).rem_euclid(5) == 0;
    let deck = |city: &mut Metro, offset: Vec3, yaw: f32| {
        city.put(
            anchor,
            look(Model::RailDeck, Mat::Rails, WHITE),
            corner + offset + Vec3::Y * RAIL_BASE,
            yaw,
            Vec3::ONE,
        );
        if station {
            city.put(
                anchor,
                look(Model::StationFrame, Mat::Fixture, WHITE),
                corner + offset + Vec3::Y * (DECK_TOP - 0.3),
                yaw,
                Vec3::new(1.0, 1.0, 0.75),
            );
            city.put(
                anchor,
                look(Model::StationRoof, Mat::ClearGlass, WHITE),
                corner + offset + Vec3::Y * (DECK_TOP + 3.4),
                yaw,
                Vec3::new(1.0, 1.0, 0.75),
            );
        }
    };
    if bz.abs() == ring && bx < ring {
        deck(city, v(PERIOD * 0.5, 0.0, 0.0), 0.0);
    }
    if bx.abs() == ring && bz < ring {
        deck(city, v(0.0, 0.0, PERIOD * 0.5), FRAC_PI_2);
    }
}

/// The river's water, docks and moored boats, and now and then a
/// footbridge whose leaves lift.
fn waterfront(city: &mut Metro, anchor: &Anchor, rng: &mut Rng, bx: i32) {
    let water = look(Model::Ground, Mat::Water, WHITE);
    city.put(
        anchor,
        water,
        Vec3::Y * (WATER_TOP - 0.2),
        0.0,
        v(20.0 / 16.8, 1.0, 20.0 / 16.8),
    );
    if let Some((_, span)) = &mut city.river {
        *span = math::max(*span, (bx.abs() as f32 + 1.0) * PERIOD);
    }
    for side in [-1.0, 1.0] {
        if rng.chance(55) {
            let z = side * 8.4;
            city.put(
                anchor,
                look(Model::Dock, Mat::Wood, WHITE),
                v(rng.range(-3.0, 3.0), WATER_TOP + 0.1, z),
                0.0,
                Vec3::ONE,
            );
            if rng.chance(70) {
                let boat = rng.pick(&[Model::Speedboat, Model::Sailboat, Model::Speedboat]);
                let at = v(rng.range(-4.0, 4.0), WATER_TOP, z - side * 2.4);
                city.put(
                    anchor,
                    look(boat, Mat::Vehicle, rng.pick(&PAINTS)),
                    at,
                    rng.range(-0.2, 0.2),
                    Vec3::ONE,
                );
            }
        }
    }
    if bx.rem_euclid(7) == 3 {
        // A footbridge's piers; its leaves are animated.
        for side in [-1.0, 1.0] {
            city.put(
                anchor,
                look(Model::RailPillar, Mat::Concrete, WHITE),
                v(0.0, WATER_TOP - 6.5, side * 9.6),
                FRAC_PI_2,
                v(0.8, 0.8, 0.8),
            );
        }
        city.props.push((
            Prop::Drawbridge,
            Site {
                anchor: *anchor,
                at: v(0.0, 1.5, 0.0),
                size: 0.0,
                yaw: FRAC_PI_2,
            },
        ));
    }
}

/// A skyscraper's material and tint.
fn tower_paint(rng: &mut Rng) -> (Mat, [f32; 4]) {
    match rng.below(10) {
        0..=3 => (Mat::Glass, rng.pick(&GLASS_TINTS)),
        4..=5 => (Mat::Panel, rng.pick(&GLASS_TINTS)),
        6..=7 => (Mat::Office, rng.pick(&STONE_TINTS)),
        _ => (Mat::Deco, rng.pick(&STONE_TINTS)),
    }
}

/// A tower of one of the styles with its crown and rooftop parts, a lobby,
/// neon and window washers.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn skyscraper(city: &mut Metro, anchor: &Anchor, rng: &mut Rng, base: Vec3, height: f32) {
    let style = rng.pick(&[
        Style::Box,
        Style::Box,
        Style::Setback,
        Style::Round,
        Style::Twisted,
        Style::Tapered,
        Style::Hex,
        Style::Deco,
        Style::Fins,
        Style::Slant,
    ]);
    let class = TOWER_FLOORS
        .iter()
        .position(|floors| *floors as f32 * models::FLOOR >= height)
        .unwrap_or(TOWER_FLOORS.len() - 1) as u8;
    let (mat, color) = tower_paint(rng);
    let model = Model::Tower(style, class);
    let yaw = if matches!(
        style,
        Style::Round | Style::Twisted | Style::Hex | Style::Deco
    ) {
        rng.range(0.0, TAU)
    } else {
        rng.below(4) as f32 * FRAC_PI_2
    };
    let widen = rng.range(0.9, 1.15);
    if city
        .put(
            anchor,
            look(model, mat, color),
            base,
            yaw,
            v(widen, 1.0, widen),
        )
        .is_none()
    {
        return;
    }
    let tall = city.size(model).y;
    let top = base + Vec3::Y * tall;
    let chrome = if rng.chance(25) {
        Mat::Gold
    } else {
        Mat::Chrome
    };
    match style {
        Style::Box | Style::Fins => {
            city.put(
                anchor,
                look(Model::RoofSlab(0), Mat::Roof, WHITE),
                top,
                yaw,
                v(widen, 1.0, widen),
            );
            city.put(
                anchor,
                look(Model::Railing(0), Mat::Fence, WHITE),
                top + Vec3::Y * 0.35,
                yaw,
                v(widen, 1.0, widen),
            );
            match rng.below(4) {
                0 => {
                    city.put(
                        anchor,
                        look(Model::Helipad, Mat::Crossing, WHITE),
                        top + Vec3::Y * 0.35,
                        0.0,
                        Vec3::ONE,
                    );
                }
                1 => {
                    city.put(
                        anchor,
                        look(Model::Antenna, Mat::Fixture, WHITE),
                        top + v(1.2, 0.35, -1.2),
                        0.0,
                        Vec3::ONE,
                    );
                }
                2 => {
                    for index in 0..3 {
                        city.put(
                            anchor,
                            look(Model::Solar, Mat::Solar, WHITE),
                            top + v(-1.8 + index as f32 * 1.8, 0.35, 0.0),
                            yaw,
                            v(0.55, 0.8, 0.8),
                        );
                    }
                }
                _ => {
                    city.put(
                        anchor,
                        look(Model::WaterTank, Mat::Wood, WHITE),
                        top + v(-1.3, 0.35, 1.3),
                        0.0,
                        Vec3::ONE,
                    );
                }
            }
            for index in 0..rng.below(3) {
                city.put(
                    anchor,
                    look(Model::AcUnit, Mat::Fixture, WHITE),
                    top + v(1.4, 0.35, 0.6 - index as f32 * 1.2),
                    yaw,
                    Vec3::ONE,
                );
            }
            city.rooftops.push(Rooftop {
                anchor: *anchor,
                base,
                height: tall,
            });
        }
        Style::Round => {
            let glass = if rng.chance(50) {
                Mat::ClearGlass
            } else {
                chrome
            };
            city.put(
                anchor,
                look(Model::Dome, glass, WHITE),
                top,
                0.0,
                Vec3::splat(1.2),
            );
        }
        Style::Hex | Style::Deco => {
            city.put(
                anchor,
                look(Model::Crown, chrome, WHITE),
                top,
                yaw,
                Vec3::splat(if style == Style::Hex { 1.1 } else { 0.62 }),
            );
        }
        Style::Setback | Style::Twisted | Style::Tapered => {
            city.put(
                anchor,
                look(Model::Spire, chrome, WHITE),
                top,
                0.0,
                Vec3::splat(rng.range(0.7, 1.3)),
            );
        }
        Style::Slant => {}
    }
    // Street level: a glass lobby and revolving door, or neon on a corner.
    let face = Quat::from_rotation_y(yaw);
    if rng.chance(45) {
        let front = base + face * v(0.0, 0.0, models::TOWER_WIDTH * 0.5 * widen + 1.3);
        city.put(
            anchor,
            look(Model::GlassBox(1), Mat::FrostedGlass, WHITE),
            front - face * v(0.0, 0.0, 0.9),
            yaw,
            v(0.6, 1.0, 0.6),
        );
        let canopy = city.put(
            anchor,
            look(Model::DoorCanopy, Mat::Fixture, WHITE),
            front,
            yaw,
            Vec3::ONE,
        );
        city.extra(
            canopy,
            Extra::Light(LocalLight::point(Vec3::new(1.0, 0.85, 0.6), 10.0, 5.0)),
        );
        city.props.push((
            Prop::Door,
            Site {
                anchor: *anchor,
                at: front + Vec3::Y * 0.05,
                size: 0.0,
                yaw,
            },
        ));
    } else if rng.chance(50) {
        let corner = base
            + face
                * v(
                    models::TOWER_WIDTH * 0.5 * widen + 0.2,
                    3.0,
                    models::TOWER_WIDTH * 0.5 * widen - 0.8,
                );
        let color = rng.below(NEON_COLORS.len() as u32) as u8;
        let neon = city.put(
            anchor,
            look(Model::NeonBlade, Mat::Neon(color), WHITE),
            corner,
            yaw,
            Vec3::ONE,
        );
        let light = NEON_COLORS[color as usize];
        city.extra(
            neon,
            Extra::Light(LocalLight::point(Vec3::from(light), 6.0, 6.0)),
        );
    }
    if matches!(style, Style::Box | Style::Fins | Style::Setback) && rng.chance(35) {
        let face_point = base + face * v(0.0, 0.0, models::TOWER_WIDTH * 0.5 * widen + 0.05);
        city.props.push((
            Prop::Gondola(class),
            Site {
                anchor: *anchor,
                at: face_point,
                size: 0.0,
                yaw,
            },
        ));
    }
}

/// Four towers per block, `scale` (0.3 to 1) setting their heights, round a
/// small plaza.
fn downtown(city: &mut Metro, anchor: &Anchor, rng: &mut Rng, scale: f32) {
    for (x, z) in [(-4.4, -4.4), (4.4, -4.4), (-4.4, 4.4), (4.4, 4.4)] {
        let height = (40.0 + 95.0 * scale) * rng.range(0.55, 1.1);
        skyscraper(city, anchor, rng, v(x, LOT, z), height);
    }
    // The cross between the towers.
    match rng.below(4) {
        0 => {
            city.put(
                anchor,
                look(Model::Sculpture, Mat::Chrome, WHITE),
                v(0.0, LOT, 0.0),
                rng.range(0.0, TAU),
                Vec3::splat(0.55),
            );
        }
        1 => {
            fountain(city, anchor, rng, v(0.0, LOT, 0.0), 0.4);
        }
        2 => {
            city.put(
                anchor,
                look(Model::Planter, Mat::Fixture, WHITE),
                v(0.0, LOT, 0.0),
                0.0,
                Vec3::splat(0.9),
            );
            city.put(
                anchor,
                look(Model::Flowers, Mat::Flora, WHITE),
                v(0.0, LOT + 0.63, 0.0),
                0.0,
                Vec3::splat(0.9),
            );
        }
        _ => {
            city.put(
                anchor,
                look(Model::Plinth, Mat::Plaza, WHITE),
                v(0.0, LOT, 0.0),
                0.0,
                Vec3::splat(0.5),
            );
            city.put(
                anchor,
                look(Model::Figure, Mat::Gold, WHITE),
                v(0.0, LOT + 1.1, 0.0),
                rng.range(0.0, TAU),
                Vec3::splat(0.4),
            );
        }
    }
    city.spots.push((*anchor, v(1.2, LOT, 0.6)));
}

/// A midrise lot: body, roof and rooftop clutter, shop fronts, balconies,
/// fire escapes and ads.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn midrise(city: &mut Metro, anchor: &Anchor, rng: &mut Rng, base: Vec3, height: f32) {
    let style = rng.pick(&[
        Mid::Block,
        Mid::Block,
        Mid::Ell,
        Mid::Court,
        Mid::Walkup,
        Mid::Terrace,
    ]);
    let class = MID_FLOORS
        .iter()
        .position(|floors| *floors as f32 * models::FLOOR >= height)
        .unwrap_or(MID_FLOORS.len() - 1) as u8;
    let (mat, color) = match rng.below(6) {
        0 => (Mat::Office, rng.pick(&STONE_TINTS)),
        1 | 2 => (Mat::Brick, rng.pick(&STONE_TINTS)),
        3 => (Mat::Panel, rng.pick(&GLASS_TINTS)),
        4 => (Mat::Deco, rng.pick(&STONE_TINTS)),
        _ => (Mat::Glass, rng.pick(&GLASS_TINTS)),
    };
    let yaw = rng.below(4) as f32 * FRAC_PI_2;
    let model = Model::Midrise(style, class);
    if city
        .put(anchor, look(model, mat, color), base, yaw, Vec3::ONE)
        .is_none()
    {
        return;
    }
    let tall = city.size(model).y;
    let top = base + Vec3::Y * tall;
    let face = Quat::from_rotation_y(yaw);
    let half = models::MID_WIDTH * 0.5;
    if matches!(style, Mid::Block | Mid::Walkup) {
        city.put(
            anchor,
            look(Model::RoofSlab(1), Mat::Roof, WHITE),
            top,
            yaw,
            Vec3::ONE,
        );
        match rng.below(8) {
            0 | 1 => {
                city.put(
                    anchor,
                    look(Model::WaterTank, Mat::Wood, WHITE),
                    top + face * v(0.8, 0.35, -0.8),
                    0.0,
                    Vec3::splat(0.8),
                );
            }
            2 => {
                city.put(
                    anchor,
                    look(Model::Solar, Mat::Solar, WHITE),
                    top + Vec3::Y * 0.35,
                    yaw,
                    v(0.9, 1.0, 1.0),
                );
            }
            3 => {
                let post = top + face * v(-0.9, 0.35, 0.9);
                city.put(
                    anchor,
                    look(Model::DishPost, Mat::Fixture, WHITE),
                    post,
                    0.0,
                    Vec3::ONE,
                );
                city.props.push((
                    Prop::Dish,
                    Site {
                        anchor: *anchor,
                        at: post + Vec3::Y * 1.2,
                        size: 0.0,
                        yaw: rng.range(0.0, TAU),
                    },
                ));
            }
            4 => {
                let board = top + Vec3::Y * 0.35;
                let face_board = city.put(
                    anchor,
                    look(
                        Model::Billboard,
                        Mat::Ad(rng.below(u32::from(AD_SLOTS)) as u8),
                        WHITE,
                    ),
                    board,
                    yaw + rng.pick(&[0.0, FRAC_PI_2]),
                    Vec3::splat(0.9),
                );
                city.extra(face_board, Extra::Advert);
            }
            5 => {
                city.props.push((
                    Prop::Turbine,
                    Site {
                        anchor: *anchor,
                        at: top + Vec3::Y * 0.35,
                        size: 0.35,
                        yaw: rng.range(0.0, TAU),
                    },
                ));
            }
            _ => {}
        }
        for index in 0..rng.below(3) {
            city.put(
                anchor,
                look(Model::AcUnit, Mat::Fixture, WHITE),
                top + face * v(-0.9 + index as f32 * 1.1, 0.35, -0.6),
                yaw,
                Vec3::ONE,
            );
        }
        if tall > 16.0 {
            city.rooftops.push(Rooftop {
                anchor: *anchor,
                base,
                height: tall,
            });
        }
    }
    // Street level on the front face (+Z turned by the yaw).
    let front = |x: f32, y: f32, out: f32| base + face * v(x, y, half + out);
    if mat == Mat::Brick && rng.chance(60) {
        city.put(
            anchor,
            look(Model::FireEscape, Mat::Lattice, [0.2, 0.2, 0.22, 1.0]),
            front(0.8, 0.0, 0.0),
            yaw,
            v(0.8, 1.0, 1.0),
        );
    }
    if matches!(mat, Mat::Brick | Mat::Office) && tall > 9.0 && rng.chance(35) {
        for floor in 1..((tall / models::FLOOR) as u32 / 2).min(5) {
            let y = floor as f32 * models::FLOOR * 2.0;
            city.put(
                anchor,
                look(Model::Balcony, Mat::Concrete, WHITE),
                front(-1.0, y, 0.0),
                yaw,
                v(0.6, 1.0, 1.0),
            );
            city.put(
                anchor,
                look(Model::BalconyRail, Mat::Pane, WHITE),
                front(-1.0, y + 0.15, 0.0),
                yaw,
                v(0.6, 1.0, 1.0),
            );
        }
    }
    if rng.chance(55) {
        city.put(
            anchor,
            look(Model::Awning, Mat::Canvas, rng.pick(&CANVAS)),
            front(0.0, models::FLOOR * 0.95, 0.0),
            yaw,
            v(0.9, 1.0, 0.9),
        );
    } else if rng.chance(40) {
        let color = rng.below(NEON_COLORS.len() as u32) as u8;
        let neon = city.put(
            anchor,
            look(Model::NeonBlade, Mat::Neon(color), WHITE),
            front(half - 0.3, 2.2, 0.6),
            yaw + FRAC_PI_2,
            v(0.8, 0.8, 0.8),
        );
        city.extra(
            neon,
            Extra::Light(LocalLight::point(
                Vec3::from(NEON_COLORS[color as usize]),
                5.0,
                5.0,
            )),
        );
    }
}

/// Nine lots per block, `scale` (0 to 1) setting their heights.
fn midtown(city: &mut Metro, anchor: &Anchor, rng: &mut Rng, scale: f32) {
    for row in 0..3 {
        for column in 0..3 {
            let base = v(-5.6 + column as f32 * 5.6, LOT, -5.6 + row as f32 * 5.6);
            let height = (7.0 + 30.0 * scale) * rng.range(0.5, 1.2);
            if height > 30.0 && rng.chance(30) {
                skyscraper(city, anchor, rng, base, height);
                continue;
            }
            midrise(city, anchor, rng, base, height);
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn residential(city: &mut Metro, anchor: &Anchor, rng: &mut Rng) {
    for row in 0..4 {
        for column in 0..4 {
            let base = v(-6.3 + column as f32 * 4.2, LOT, -6.3 + row as f32 * 4.2);
            if rng.chance(12) {
                let tree = rng.pick(&[Model::Tree(0), Model::Tree(1), Model::Pine, Model::Poplar]);
                city.put(
                    anchor,
                    look(tree, Mat::Leaves, WHITE),
                    base,
                    rng.range(0.0, TAU),
                    Vec3::splat(rng.range(0.7, 0.95)),
                );
                continue;
            }
            let yaw = rng.below(4) as f32 * FRAC_PI_2;
            let face = Quat::from_rotation_y(yaw);
            if rng.chance(15) {
                city.put(
                    anchor,
                    look(Model::Townhouse, Mat::Brick, rng.pick(&PASTELS)),
                    base,
                    yaw,
                    v(0.62, 1.0, 0.9),
                );
                continue;
            }
            if rng.chance(10) {
                // A corner shop under an awning.
                city.put(
                    anchor,
                    look(Model::Shop, Mat::Office, rng.pick(&PASTELS)),
                    base,
                    yaw,
                    v(0.75, 1.0, 0.9),
                );
                city.put(
                    anchor,
                    look(Model::Awning, Mat::Canvas, rng.pick(&CANVAS)),
                    base + face * v(0.0, models::FLOOR * 0.9, 1.45),
                    yaw,
                    v(1.1, 1.0, 0.8),
                );
                continue;
            }
            let floors = 1 + rng.below(3) as u8;
            let mat = if rng.chance(55) {
                Mat::Office
            } else {
                Mat::Brick
            };
            if city
                .put(
                    anchor,
                    look(Model::House(floors), mat, rng.pick(&PASTELS)),
                    base,
                    yaw,
                    Vec3::ONE,
                )
                .is_none()
            {
                return;
            }
            let top = base + Vec3::Y * (floors as f32 * models::FLOOR);
            let roof = if rng.chance(65) {
                Model::Gable
            } else {
                Model::Hip
            };
            let roof_scale = if roof == Model::Gable {
                v(0.95, 1.0, 1.0)
            } else {
                v(0.97, 1.0, 0.97)
            };
            city.put(
                anchor,
                look(roof, Mat::Tiles, rng.pick(&ROOF_TINTS)),
                top,
                yaw,
                roof_scale,
            );
            if rng.chance(20) {
                city.put(
                    anchor,
                    look(Model::Solar, Mat::Solar, WHITE),
                    top + face * v(0.0, 0.3, 0.4),
                    yaw,
                    Vec3::splat(0.5),
                );
            }
            if rng.chance(35) {
                // A car in the drive.
                let car = rng.pick(&[Car::Sedan, Car::Hatch, Car::Suv, Car::Van, Car::Pickup]);
                city.put(
                    anchor,
                    look(Model::Car(car), Mat::Vehicle, rng.pick(&PAINTS)),
                    base + face * v(-2.1, 0.0, 0.2),
                    yaw + FRAC_PI_2,
                    Vec3::ONE,
                );
            }
            if rng.chance(30) {
                let tree = rng.pick(&[Model::Tree(2), Model::Bush, Model::Poplar]);
                city.put(
                    anchor,
                    look(tree, Mat::Leaves, WHITE),
                    base + face * v(1.2, 0.0, -1.9),
                    rng.range(0.0, TAU),
                    Vec3::splat(0.55),
                );
            }
            match rng.below(5) {
                0 => {
                    city.put(
                        anchor,
                        look(Model::Hedge, Mat::Hedge, WHITE),
                        base + face * v(0.0, 0.0, 1.9),
                        yaw,
                        v(0.9, 0.6, 0.6),
                    );
                }
                1 => {
                    city.put(
                        anchor,
                        look(Model::Bush, Mat::Leaves, WHITE),
                        base + face * v(1.4, 0.0, 1.8),
                        0.0,
                        Vec3::splat(0.6),
                    );
                }
                2 => {
                    city.put(
                        anchor,
                        look(Model::Flowers, Mat::Flora, WHITE),
                        base + face * v(-1.0, 0.0, 1.9),
                        yaw,
                        v(1.0, 1.0, 0.5),
                    );
                }
                _ => {}
            }
        }
    }
    city.spots.push((*anchor, v(0.0, LOT, 0.0)));
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn industrial(city: &mut Metro, anchor: &Anchor, rng: &mut Rng) {
    for row in 0..2 {
        let z = -4.2 + row as f32 * 8.4;
        match rng.below(4) {
            0 => {
                city.put(
                    anchor,
                    look(Model::Warehouse, Mat::Siding, rng.pick(&STONE_TINTS)),
                    v(-0.8, LOT, z),
                    0.0,
                    v(1.0, 1.0, 0.9),
                );
                for index in 0..2 {
                    city.put(
                        anchor,
                        look(Model::AcUnit, Mat::Fixture, WHITE),
                        v(-4.0 + index as f32 * 5.0, LOT + 6.7, z),
                        0.0,
                        Vec3::ONE,
                    );
                }
                city.put(
                    anchor,
                    look(Model::Container, Mat::Container, rng.pick(&PAINTS)),
                    v(7.2, LOT, z),
                    FRAC_PI_2,
                    Vec3::ONE,
                );
            }
            1 => {
                city.put(
                    anchor,
                    look(Model::Shed, Mat::Siding, rng.pick(&PASTELS)),
                    v(-2.8, LOT, z),
                    0.0,
                    Vec3::ONE,
                );
                for index in 0..2 {
                    city.put(
                        anchor,
                        look(Model::Tank, Mat::Brushed, WHITE),
                        v(4.3, LOT, z - 1.8 + index as f32 * 3.6),
                        0.0,
                        Vec3::splat(0.75),
                    );
                }
                city.put(
                    anchor,
                    look(Model::Truck, Mat::Vehicle, rng.pick(&PAINTS)),
                    v(-2.8, LOT, z + 3.4 - row as f32 * 6.8),
                    0.0,
                    Vec3::ONE,
                );
            }
            2 => {
                // Stacked containers and a truck to haul them.
                for index in 0..6 {
                    let (column, level) = (index % 3, index / 3);
                    let at = v(-4.0 + column as f32 * 2.7, LOT + level as f32 * 2.6, z);
                    city.put(
                        anchor,
                        look(Model::Container, Mat::Container, rng.pick(&PAINTS)),
                        at,
                        FRAC_PI_2,
                        Vec3::ONE,
                    );
                }
                city.put(
                    anchor,
                    look(Model::Truck, Mat::Vehicle, rng.pick(&PAINTS)),
                    v(4.5, LOT, z),
                    FRAC_PI_2,
                    Vec3::ONE,
                );
            }
            _ => {
                // A tank farm.
                for index in 0..3 {
                    city.put(
                        anchor,
                        look(Model::Tank, Mat::Brushed, rng.pick(&STONE_TINTS)),
                        v(-5.5 + index as f32 * 4.6, LOT, z),
                        0.0,
                        Vec3::splat(0.85),
                    );
                }
                city.put(
                    anchor,
                    look(Model::Silo, Mat::Concrete, WHITE),
                    v(6.5, LOT, z),
                    0.0,
                    Vec3::splat(0.8),
                );
            }
        }
    }
    // Parked cars and pallets of crates along the south fence.
    for index in 0..rng.below(4) + 2 {
        let car = rng.pick(&[Car::Sedan, Car::Hatch, Car::Van, Car::Pickup]);
        city.put(
            anchor,
            look(Model::Car(car), Mat::Vehicle, rng.pick(&PAINTS)),
            v(-6.8 + index as f32 * 1.35, LOT, 7.6),
            FRAC_PI_2,
            Vec3::ONE,
        );
    }
    for index in 0..rng.below(4) + 1 {
        let at = v(
            3.0 + index as f32 * 1.1,
            LOT,
            7.4 - (index % 2) as f32 * 0.9,
        );
        city.put(
            anchor,
            look(Model::Container, Mat::Wood, WHITE),
            at,
            0.0,
            v(0.14, 0.3, 0.35),
        );
    }
    // A chain-link fence round the yard and a floodlight over it.
    city.put(
        anchor,
        look(Model::Railing(0), Mat::Fence, WHITE),
        v(0.0, LOT, 0.0),
        0.0,
        v(2.7, 1.6, 2.7),
    );
    let flood = city.put(
        anchor,
        look(Model::Floodlight, Mat::Fixture, WHITE),
        v(7.8, LOT, 0.0),
        -FRAC_PI_2,
        Vec3::splat(0.6),
    );
    if rng.chance(50) {
        let head = city.offset(Model::Floodlight, v(0.0, 12.2, -0.3));
        city.extra(
            flood,
            Extra::Light(
                LocalLight::spot(
                    Vec3::new(1.0, 0.9, 0.75),
                    40.0,
                    16.0,
                    Vec3::new(0.0, -0.8, -1.0),
                    25.0_f32.to_radians(),
                    55.0_f32.to_radians(),
                )
                .with_offset(head),
            ),
        );
    }
    if rng.chance(20) {
        // A tall stack; only some of them smoke.
        let base = v(-7.4, LOT, rng.range(-6.0, 6.0));
        let stack = city.put(
            anchor,
            look(Model::Stack, Mat::Concrete, WHITE),
            base,
            0.0,
            Vec3::ONE,
        );
        if rng.chance(30) {
            let seed = rng.next();
            let top = city.offset(Model::Stack, v(0.0, 27.0, 0.0));
            city.extra(
                stack,
                Extra::Emitter(ParticleEmitter {
                    offset: top,
                    ..smoke(seed)
                }),
            );
        }
    }
    if rng.chance(30) {
        city.props.push((
            Prop::Turbine,
            Site {
                anchor: *anchor,
                at: v(6.5, LOT, -7.0),
                size: 1.0,
                yaw: rng.range(0.0, TAU),
            },
        ));
    }
}

fn tree(city: &mut Metro, anchor: &Anchor, rng: &mut Rng, base: Vec3, scale: f32) {
    let model = rng.pick(&[
        Model::Tree(0),
        Model::Tree(1),
        Model::Tree(2),
        Model::Pine,
        Model::Poplar,
        Model::Palm,
        Model::Bush,
    ]);
    let mat = if model == Model::Palm {
        Mat::Fronds
    } else {
        Mat::Leaves
    };
    let size = scale * rng.range(0.7, 1.15);
    city.put(
        anchor,
        look(model, mat, WHITE),
        base,
        rng.range(0.0, TAU),
        Vec3::splat(size),
    );
}

/// A basin with water and a spray of particles.
fn fountain(city: &mut Metro, anchor: &Anchor, rng: &mut Rng, base: Vec3, scale: f32) {
    let basin = city.put(
        anchor,
        look(Model::Basin, Mat::Plaza, WHITE),
        base,
        0.0,
        Vec3::splat(scale),
    );
    city.put(
        anchor,
        look(Model::Pool, Mat::Water, WHITE),
        base + Vec3::Y * (0.32 * scale),
        0.0,
        Vec3::splat(scale),
    );
    let seed = rng.next();
    let spout = city.offset(Model::Basin, v(0.0, 2.4, 0.0));
    city.extra(
        basin,
        Extra::Emitter(ParticleEmitter {
            offset: spout,
            ..spray(seed)
        }),
    );
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn park_block(city: &mut Metro, anchor: &Anchor, rng: &mut Rng) {
    let centerpiece = rng.below(4);
    for _ in 0..rng.below(8) + 12 {
        let at = v(rng.range(-7.8, 7.8), LOT, rng.range(-7.8, 7.8));
        if centerpiece != 3 && at.length() < 5.5 {
            continue;
        }
        tree(city, anchor, rng, at, 1.0);
    }
    for _ in 0..rng.below(3) + 1 {
        let at = v(rng.range(-6.0, 6.0), LOT, rng.range(-6.0, 6.0));
        city.put(
            anchor,
            look(Model::Rock, Mat::Concrete, WHITE),
            at,
            rng.range(0.0, TAU),
            Vec3::splat(rng.range(0.4, 0.8)),
        );
    }
    for index in 0..3 {
        let (sin, cos) = math::sin_cos(TAU * index as f32 / 3.0 + 0.4);
        city.put(
            anchor,
            look(Model::Bench, Mat::Fixture, WHITE),
            v(cos * 6.0, LOT, sin * 6.0),
            -TAU * index as f32 / 3.0,
            Vec3::splat(FURNITURE),
        );
    }
    // Crossing paths, lit at night.
    for yaw in [0.0, FRAC_PI_2] {
        city.put(
            anchor,
            look(Model::Ground, Mat::Pavers, WHITE),
            Vec3::Y * 0.12,
            yaw,
            v(1.0, 1.0, 0.14),
        );
    }
    for (x, z) in [(2.2, 2.2), (-2.2, -2.2)] {
        let lamp = city.put(
            anchor,
            look(Model::Lamp, Mat::Fixture, WHITE),
            v(x, LOT, z),
            rng.range(0.0, TAU),
            Vec3::splat(0.8),
        );
        let head = city.offset(Model::Lamp, v(0.9, 3.54, 0.0));
        city.extra(
            lamp,
            Extra::Light(
                LocalLight::spot(
                    Vec3::new(1.0, 0.8, 0.55),
                    14.0,
                    7.0,
                    Vec3::NEG_Y,
                    40.0_f32.to_radians(),
                    75.0_f32.to_radians(),
                )
                .with_offset(head),
            ),
        );
    }
    for _ in 0..rng.below(3) + 2 {
        let at = v(rng.range(-7.0, 7.0), LOT, rng.range(-7.0, 7.0));
        city.put(
            anchor,
            look(Model::Bush, Mat::Leaves, WHITE),
            at,
            rng.range(0.0, TAU),
            Vec3::splat(rng.range(0.5, 0.8)),
        );
    }
    let picnic = v(rng.range(-5.0, 5.0), LOT, rng.range(4.0, 6.5));
    city.put(
        anchor,
        look(Model::CafeTable, Mat::Fixture, WHITE),
        picnic,
        0.0,
        Vec3::splat(FURNITURE),
    );
    city.put(
        anchor,
        look(Model::Umbrella, Mat::Canvas, rng.pick(&CANVAS)),
        picnic,
        0.0,
        Vec3::splat(FURNITURE * 1.2),
    );
    match centerpiece {
        0 => {
            // A Ferris wheel's legs; the wheel turns.
            let (size, yaw) = (rng.range(0.85, 1.1), rng.range(0.0, PI));
            let face = Quat::from_rotation_y(yaw);
            for side in [-1.0, 1.0] {
                let leg = turned(
                    face * v(0.0, 4.0 * size, side * 1.1),
                    face * Quat::from_rotation_x(side * 0.14),
                    Vec3::splat(size),
                );
                city.part(anchor, leg, look(Model::FerrisLeg, Mat::Fixture, WHITE));
            }
            city.wheels.push(Site {
                anchor: *anchor,
                at: v(0.0, 8.0 * size + LOT, 0.0),
                size,
                yaw,
            });
        }
        1 => {
            city.put(
                anchor,
                look(Model::Pool, Mat::Water, WHITE),
                v(0.0, LOT, 0.0),
                0.0,
                v(1.6, 1.0, 1.1),
            );
            fountain(city, anchor, rng, v(0.0, LOT, 0.0), 0.5);
        }
        2 => {
            city.put(
                anchor,
                look(Model::CarouselBase, Mat::Fixture, WHITE),
                v(0.0, LOT, 0.0),
                0.0,
                Vec3::splat(0.8),
            );
            city.props.push((
                Prop::Carousel,
                Site {
                    anchor: *anchor,
                    at: v(0.0, LOT, 0.0),
                    size: 0.8,
                    yaw: 0.0,
                },
            ));
        }
        _ => {
            city.put(
                anchor,
                look(Model::Sculpture, Mat::Chrome, WHITE),
                v(0.0, LOT, 0.0),
                rng.range(0.0, TAU),
                Vec3::splat(0.8),
            );
        }
    }
    city.put(
        anchor,
        look(Model::Flowers, Mat::Flora, WHITE),
        v(rng.range(-6.0, 6.0), LOT, rng.range(-6.0, 6.0)),
        0.0,
        Vec3::splat(1.5),
    );
    city.spots.push((*anchor, v(3.0, LOT, -3.0)));
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn plaza_block(city: &mut Metro, anchor: &Anchor, rng: &mut Rng) {
    match rng.below(4) {
        0 => {
            city.put(
                anchor,
                look(Model::Sculpture, Mat::Chrome, WHITE),
                v(0.0, LOT, 0.0),
                rng.range(0.0, TAU),
                Vec3::ONE,
            );
        }
        1 => {
            city.put(
                anchor,
                look(Model::GlassBox(0), Mat::ClearGlass, WHITE),
                v(0.0, LOT, 0.0),
                0.0,
                v(1.3, 1.4, 1.3),
            );
            city.put(
                anchor,
                look(Model::Dome, Mat::Chrome, WHITE),
                v(0.0, LOT + 4.48, 0.0),
                0.0,
                v(1.3, 0.8, 1.3),
            );
        }
        2 => {
            city.put(
                anchor,
                look(Model::Plinth, Mat::Plaza, WHITE),
                v(0.0, LOT, 0.0),
                0.0,
                Vec3::ONE,
            );
            city.put(
                anchor,
                look(Model::Figure, Mat::Gold, WHITE),
                v(0.0, LOT + 2.2, 0.0),
                rng.range(0.0, TAU),
                Vec3::splat(0.8),
            );
        }
        _ => fountain(city, anchor, rng, v(0.0, LOT, 0.0), 0.9),
    }
    // Palms or trees around, flags, kiosks and cafés.
    let palms = rng.chance(50);
    for index in 0..8 {
        let (sin, cos) = math::sin_cos(TAU * index as f32 / 8.0);
        let at = v(cos * 6.8, LOT, sin * 6.8);
        if palms {
            city.put(
                anchor,
                look(Model::Palm, Mat::Fronds, WHITE),
                at,
                rng.range(0.0, TAU),
                Vec3::splat(0.9),
            );
        } else {
            tree(city, anchor, rng, at, 0.8);
        }
    }
    for index in 0..3 {
        let at = v(-3.0 + index as f32 * 3.0, LOT, -7.6);
        city.put(
            anchor,
            look(Model::Flagpole, Mat::Fixture, WHITE),
            at,
            0.0,
            Vec3::ONE,
        );
        city.props.push((
            Prop::Flag,
            Site {
                anchor: *anchor,
                at: at + Vec3::Y * 6.9,
                size: 0.0,
                yaw: rng.range(-0.3, 0.3),
            },
        ));
    }
    for index in 0..2 {
        let at = v(4.5 - index as f32 * 9.0, LOT, 4.5);
        city.put(
            anchor,
            look(Model::KioskBody, Mat::Fixture, WHITE),
            at,
            PI,
            Vec3::splat(FURNITURE * 1.4),
        );
        city.put(
            anchor,
            look(Model::KioskRoof, Mat::Canvas, rng.pick(&CANVAS)),
            at + Vec3::Y * (2.0 * FURNITURE * 1.4),
            0.0,
            Vec3::splat(FURNITURE * 1.4),
        );
    }
    for index in 0..3 {
        let at = v(-2.0 + index as f32 * 2.0, LOT, 3.5);
        city.put(
            anchor,
            look(Model::CafeTable, Mat::Fixture, WHITE),
            at,
            0.0,
            Vec3::splat(FURNITURE),
        );
        city.put(
            anchor,
            look(Model::Umbrella, Mat::Canvas, rng.pick(&CANVAS)),
            at,
            0.0,
            Vec3::splat(FURNITURE * 1.2),
        );
    }
    if rng.chance(40) {
        let board = v(-6.8, LOT, -2.0);
        city.put(
            anchor,
            look(Model::Flagpole, Mat::Fixture, WHITE),
            board,
            0.0,
            v(1.2, 0.45, 1.2),
        );
        city.props.push((
            Prop::Trivision,
            Site {
                anchor: *anchor,
                at: board + Vec3::Y * 3.9,
                size: 0.0,
                yaw: FRAC_PI_2,
            },
        ));
    }
    city.spots.push((*anchor, v(-1.5, LOT, 2.0)));
    city.spots.push((*anchor, v(2.5, LOT, -2.5)));
}

fn market_block(city: &mut Metro, anchor: &Anchor, rng: &mut Rng) {
    for row in 0..3 {
        for column in 0..4 {
            let at = v(-6.0 + column as f32 * 4.0, LOT, -5.0 + row as f32 * 5.0);
            match rng.below(4) {
                0 | 1 => {
                    city.put(
                        anchor,
                        look(Model::KioskBody, Mat::Fixture, WHITE),
                        at,
                        rng.pick(&[0.0, PI]),
                        Vec3::splat(FURNITURE * 1.3),
                    );
                    city.put(
                        anchor,
                        look(Model::KioskRoof, Mat::Canvas, rng.pick(&CANVAS)),
                        at + Vec3::Y * (2.0 * FURNITURE * 1.3),
                        0.0,
                        Vec3::splat(FURNITURE * 1.3),
                    );
                }
                2 => {
                    city.put(
                        anchor,
                        look(Model::FoodCart, Mat::Vehicle, rng.pick(&PAINTS)),
                        at,
                        rng.range(0.0, TAU),
                        Vec3::splat(FURNITURE),
                    );
                    city.put(
                        anchor,
                        look(Model::Umbrella, Mat::Canvas, rng.pick(&CANVAS)),
                        at + v(0.6, 0.0, 0.0),
                        0.0,
                        Vec3::splat(FURNITURE * 1.1),
                    );
                }
                _ => {
                    city.put(
                        anchor,
                        look(Model::Planter, Mat::Fixture, WHITE),
                        at,
                        0.0,
                        Vec3::splat(FURNITURE),
                    );
                    city.put(
                        anchor,
                        look(Model::Flowers, Mat::Flora, WHITE),
                        at + Vec3::Y * (0.7 * FURNITURE),
                        0.0,
                        Vec3::splat(FURNITURE),
                    );
                }
            }
        }
    }
    city.put(
        anchor,
        look(Model::BoothFrame, Mat::Fixture, WHITE),
        v(7.2, LOT, 7.2),
        0.0,
        Vec3::splat(FURNITURE),
    );
    city.put(
        anchor,
        look(Model::BoothGlass, Mat::ClearGlass, WHITE),
        v(7.2, LOT + 0.05, 7.2),
        0.0,
        Vec3::splat(FURNITURE),
    );
    city.spots.push((*anchor, v(-2.0, LOT, -2.5)));
    city.spots.push((*anchor, v(2.0, LOT, 2.5)));
}

fn construction_block(city: &mut Metro, anchor: &Anchor, rng: &mut Rng) {
    let class = rng.below(3) as u8;
    let frame = Model::Midrise(Mid::Block, class);
    city.put(
        anchor,
        look(frame, Mat::Concrete, WHITE),
        v(-3.0, LOT, -3.0),
        0.0,
        v(1.4, 1.0, 1.4),
    );
    let tall = city.size(frame).y;
    for side in [0.0, FRAC_PI_2] {
        city.put(
            anchor,
            look(Model::FireEscape, Mat::Lattice, [0.9, 0.55, 0.1, 1.0]),
            v(-3.0, LOT, -3.0) + Quat::from_rotation_y(side) * v(0.0, 0.0, 3.4),
            side,
            v(1.9, tall / 12.6, 1.0),
        );
    }
    // The crane mast; its jib turns.
    let mast = v(5.2, LOT, 5.2);
    let height = tall + rng.range(10.0, 18.0);
    let mast_height = city.size(Model::CraneMast).y;
    if city
        .put(
            anchor,
            look(Model::CraneMast, Mat::Lattice, CRANE),
            mast,
            0.0,
            v(1.0, height / mast_height, 1.0),
        )
        .is_some()
    {
        city.cranes.push(Site {
            anchor: *anchor,
            at: mast + Vec3::Y * height,
            size: rng.range(0.7, 1.0),
            yaw: rng.range(0.0, TAU),
        });
    }
    city.props.push((
        Prop::Excavator,
        Site {
            anchor: *anchor,
            at: v(3.0, LOT, -5.0),
            size: 0.0,
            yaw: rng.range(0.0, TAU),
        },
    ));
    for index in 0..3 {
        city.put(
            anchor,
            look(Model::Container, Mat::Container, rng.pick(&PAINTS)),
            v(-6.5 + index as f32 * 2.7, LOT, 7.0),
            FRAC_PI_2,
            Vec3::ONE,
        );
    }
    city.put(
        anchor,
        look(Model::Mixer, Mat::Vehicle, WHITE),
        v(2.0, LOT, 7.0),
        0.0,
        Vec3::ONE,
    );
    city.put(
        anchor,
        look(Model::MixerDrum, Mat::Vehicle, rng.pick(&PAINTS)),
        v(1.3, LOT + 0.9, 7.0),
        0.0,
        Vec3::ONE,
    );
    if rng.chance(50) {
        // A welder's sparks high on the frame.
        let sparks = city.put(
            anchor,
            look(Model::Gondola, Mat::Fixture, WHITE),
            v(-3.0, LOT + tall * 0.6, 0.5),
            0.0,
            Vec3::splat(0.6),
        );
        let seed = rng.next();
        city.extra(sparks, Extra::Emitter(welding(seed)));
    }
}

fn stadium_block(city: &mut Metro, anchor: &Anchor, rng: &mut Rng) {
    let team = rng.pick(&PAINTS);
    city.put(
        anchor,
        look(Model::Stadium, Mat::Seats, tint(team, 1.0)),
        v(0.0, LOT, 0.0),
        0.0,
        Vec3::ONE,
    );
    city.put(
        anchor,
        look(Model::Pitch, Mat::Pitch, WHITE),
        v(0.0, LOT, 0.0),
        0.0,
        Vec3::ONE,
    );
    city.put(
        anchor,
        look(Model::StadiumRoof, Mat::Chrome, WHITE),
        v(0.0, LOT + 7.2, 0.0),
        0.0,
        Vec3::ONE,
    );
    for (x, z) in [(-8.8f32, -8.8f32), (8.8, -8.8), (-8.8, 8.8), (8.8, 8.8)] {
        // The lamp faces -Z: turned toward the pitch.
        let yaw = x.atan2(z);
        let light = city.put(
            anchor,
            look(Model::Floodlight, Mat::Fixture, WHITE),
            v(x, LOT, z),
            yaw,
            Vec3::ONE,
        );
        let head = city.offset(Model::Floodlight, v(0.0, 12.2, -0.3));
        city.extra(
            light,
            Extra::Light(
                LocalLight::spot(
                    Vec3::new(0.9, 0.95, 1.0),
                    120.0,
                    30.0,
                    Vec3::new(0.0, -0.7, -1.0),
                    20.0_f32.to_radians(),
                    45.0_f32.to_radians(),
                )
                .with_offset(head),
            ),
        );
    }
}

fn landmark(city: &mut Metro, anchor: &Anchor, rng: &mut Rng) {
    city.put(
        anchor,
        look(Model::TvTower, Mat::Chrome, WHITE),
        v(0.0, LOT, 0.0),
        0.0,
        Vec3::ONE,
    );
    city.put(
        anchor,
        look(Model::ObsPod, Mat::ClearGlass, WHITE),
        v(0.0, LOT + 118.5, 0.0),
        0.0,
        Vec3::ONE,
    );
    city.props.push((
        Prop::Ring,
        Site {
            anchor: *anchor,
            at: v(0.0, LOT + 126.5, 0.0),
            size: 1.0,
            yaw: 0.0,
        },
    ));
    for index in 0..4 {
        let (sin, cos) = math::sin_cos(TAU * (index as f32 + 0.5) / 4.0);
        fountain(city, anchor, rng, v(cos * 6.2, LOT, sin * 6.2), 0.45);
    }
    for index in 0..12 {
        let (sin, cos) = math::sin_cos(TAU * index as f32 / 12.0);
        city.put(
            anchor,
            look(Model::Palm, Mat::Fronds, WHITE),
            v(cos * 8.6, LOT, sin * 8.6),
            rng.range(0.0, TAU),
            Vec3::splat(0.85),
        );
    }
    for index in 0..4 {
        let (sin, cos) = math::sin_cos(TAU * index as f32 / 4.0);
        let at = v(cos * 4.0, LOT, sin * 4.0);
        city.put(
            anchor,
            look(Model::Flagpole, Mat::Fixture, WHITE),
            at,
            0.0,
            Vec3::ONE,
        );
        city.props.push((
            Prop::Flag,
            Site {
                anchor: *anchor,
                at: at + Vec3::Y * 6.9,
                size: 0.0,
                yaw: TAU * index as f32 / 4.0,
            },
        ));
    }
    city.spots.push((*anchor, v(3.0, LOT, 3.0)));
}

// Movers.

/// Every moving renderable: assemblies at the sites the blocks offered,
/// then trains, boats, aircraft, rooftop parts, people, buses and cars.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn movers(city: &mut Metro, config: &StressConfig) {
    let total = city.moving_left;
    let mut rng = Rng::new(config.seed ^ 0x7a11, 0, 0);
    let share = |percent: u32| total / 100 * percent + (total % 100) * percent / 100;
    let cranes = std::mem::take(&mut city.cranes);
    for site in cranes.iter().take((share(3) / 2) as usize) {
        crane(city, &mut rng, site);
    }
    let wheels = std::mem::take(&mut city.wheels);
    for site in wheels.iter().take((share(3) / 9) as usize) {
        ferris(city, &mut rng, site);
    }
    // Keyframed props: shuffled so every kind appears, within their share.
    let mut sites = std::mem::take(&mut city.props);
    for index in (1..sites.len()).rev() {
        sites.swap(index, rng.below(index as u32 + 1) as usize);
    }
    let prop_budget = share(12);
    let start = city.moving_left;
    for (prop, site) in &sites {
        if start - city.moving_left >= prop_budget {
            break;
        }
        animated_prop(city, &mut rng, *prop, site);
    }
    if let Some(ring) = city.rail {
        let half = ring as f32 * PERIOD;
        let perimeter = 8.0 * half;
        let trains = math::min((share(7) / 5) as f32, perimeter / 60.0) as u32;
        let speed = rng.range(12.0, 16.0);
        let world = city.world();
        let length = city.size(Model::TrainCar).y;
        for train in 0..trains {
            let phase = perimeter * train as f32 / trains as f32;
            let color = rng.pick(&[WHITE, [0.95, 0.35, 0.25, 1.0], [0.3, 0.55, 0.95, 1.0]]);
            for wagon in 0..5 {
                let motion = Motion::Travel(Travel {
                    center: v(0.0, DECK_TOP + length * 0.5, 0.0),
                    half: Vec2::splat(half),
                    speed,
                    phase: phase - wagon as f32 * 5.3,
                });
                let model = if wagon == 0 {
                    Model::Locomotive
                } else {
                    Model::TrainCar
                };
                city.mover(
                    &world,
                    at(Vec3::ZERO),
                    look(model, Mat::Vehicle, color),
                    Some(motion),
                    Some(motion),
                );
            }
        }
    }
    if let Some((row, span)) = city.river {
        let world = city.world();
        let boats = share(4).min((span * 0.3) as u32);
        for _ in 0..boats {
            let model = rng.pick(&[
                Model::Speedboat,
                Model::Speedboat,
                Model::Ferry,
                Model::Barge,
                Model::Sailboat,
            ]);
            let size = city.size(model);
            let half = Vec2::new(
                rng.range(12.0, math::max(span * 0.6, 14.0)),
                if model == Model::Speedboat { 3.0 } else { 2.2 },
            );
            let center = v(
                rng.range(-span, span) * 0.5,
                WATER_TOP + size.y * 0.5 - 0.3,
                row as f32 * PERIOD + PERIOD * 0.5,
            );
            let motion = Motion::Travel(Travel {
                center,
                half,
                speed: rng.range(2.5, 6.0) * rng.sign(),
                phase: rng.range(0.0, 8.0 * (half.x + half.y)),
            });
            let paint = rng.pick(&PAINTS);
            let boat = city.mover(
                &world,
                at(Vec3::ZERO),
                look(model, Mat::Vehicle, paint),
                Some(motion),
                Some(motion),
            );
            if boat.is_some() && model == Model::Sailboat {
                // The sail rides on the same loop.
                let sail = Motion::Travel(Travel {
                    center: center + Vec3::Y * 2.2,
                    ..travel(motion)
                });
                city.mover(
                    &world,
                    at(Vec3::ZERO),
                    look(Model::Sail, Mat::Canvas, WHITE),
                    Some(sail),
                    Some(sail),
                );
            }
        }
    }
    for _ in 0..share(2) / 2 {
        helicopter(city, &mut rng);
    }
    for _ in 0..share(1).min(6) {
        blimp(city, &mut rng);
    }
    for _ in 0..share(7) / 2 {
        drone(city, &mut rng);
    }
    let rooftops = std::mem::take(&mut city.rooftops);
    if !rooftops.is_empty() {
        for index in 0..share(3) {
            let roof = rooftops[(hash(index ^ config.seed) as usize) % rooftops.len()];
            let at = roof.base + Vec3::Y * (roof.height + 0.35);
            let spin = Spin {
                base: Quat::IDENTITY,
                axis_z: false,
                speed: rng.range(1.5, 4.0) * rng.sign(),
                phase: rng.range(0.0, TAU),
            };
            let flat = Spin {
                base: roof.anchor.frame.rotation,
                ..spin
            };
            let size = city.size(Model::Beacon);
            let beacon = city.mover(
                &roof.anchor,
                part_at(at + Vec3::Y * (size.y * 0.5), Vec3::ONE),
                look(Model::Beacon, Mat::Fixture, WHITE),
                Some(Motion::Spin(spin)),
                Some(Motion::Spin(flat)),
            );
            let lamp = city.offset(Model::Beacon, v(0.0, 1.9, 0.0));
            city.extra(
                beacon,
                Extra::Light(
                    LocalLight::point(Vec3::new(1.0, 0.3, 0.2), 12.0, 10.0).with_offset(lamp),
                ),
            );
        }
    }
    // People walking the sidewalks, and some standing about.
    let spots = std::mem::take(&mut city.spots);
    for index in 0..share(3) as usize {
        if spots.is_empty() {
            break;
        }
        let (anchor, at) = spots[index % spots.len()];
        let jitter = v(rng.range(-0.8, 0.8), 0.0, rng.range(-0.8, 0.8));
        person(city, &mut rng, &anchor, at + jitter, None);
    }
    for _ in 0..share(12) {
        // Busier toward the center: blocks were laid out from the center.
        let (bx, bz) = if city.blocks.is_empty() {
            (0, 0)
        } else {
            let pick = rng.unit() * rng.unit() * city.blocks.len() as f32;
            let (bx, bz, _) = city.blocks[(pick as usize).min(city.blocks.len() - 1)];
            (bx, bz)
        };
        let center = v(
            bx as f32 * PERIOD + PERIOD * 0.5,
            LOT,
            bz as f32 * PERIOD + PERIOD * 0.5,
        );
        let side = if rng.chance(50) { 9.0 } else { 9.45 };
        let speed = rng.range(0.45, 0.8) * rng.sign();
        let motion = Motion::Travel(Travel {
            center,
            half: Vec2::splat(side),
            speed,
            phase: rng.range(0.0, 8.0 * side),
        });
        let world = city.world();
        person(city, &mut rng, &world, Vec3::ZERO, Some(motion));
    }
    let buses = share(5);
    let trucks = share(3);
    let cars = city.moving_left.saturating_sub(buses + trucks);
    for index in 0..buses + trucks + cars {
        let kind = if index < buses {
            0
        } else if index < buses + trucks {
            1
        } else {
            2
        };
        vehicle(city, &mut rng, kind);
    }
}

fn travel(motion: Motion) -> Travel {
    match motion {
        Motion::Travel(travel) => travel,
        _ => unreachable!("boats travel"),
    }
}

/// A person with a skeleton of joint nodes, walking along `motion` or
/// standing at `base` in `anchor`'s frame.
fn person(city: &mut Metro, rng: &mut Rng, anchor: &Anchor, base: Vec3, motion: Option<Motion>) {
    if city.moving_left == 0 {
        return;
    }
    let scale = PERSON * rng.range(0.9, 1.1);
    let root = city.model_root(
        anchor,
        Transform {
            translation: base,
            rotation: Quat::from_rotation_y(rng.range(0.0, TAU)),
            scale: Vec3::splat(scale),
        },
        motion,
    );
    let mut joints: Vec<u32> = Vec::with_capacity(people::JOINTS);
    for (parent, offset) in people::SKELETON {
        let parent_node = parent.map_or(root.node, |parent| Some(joints[parent]));
        let node = city.push(
            &Anchor {
                node: parent_node,
                frame: Transform::default(),
            },
            at(Vec3::from(offset)),
            None,
            None,
            false,
        );
        city.extras.push((node, Extra::Animated));
        joints.push(node);
    }
    let variant = rng.below(u32::from(people::VARIANTS)) as u8;
    let shade = rng.range(0.85, 1.0);
    let body = city.animated(
        &root,
        Transform::default(),
        Some(look(
            Model::Person(variant),
            Mat::Person,
            tint(WHITE, shade),
        )),
    );
    city.extra(body, Extra::Skin(joints.clone()));
    let walking = motion.is_some();
    let speed = match motion {
        Some(Motion::Travel(travel)) => travel.speed.abs() / 0.62,
        _ => rng.range(0.8, 1.2),
    };
    city.extra(
        root.node,
        Extra::Player {
            clip: if walking { Clip::Walk } else { Clip::Idle },
            targets: joints,
            time: rng.range(0.0, 4.0),
            speed,
        },
    );
}

/// One animated part of a prop: the pivot its clip turns (or moves) under
/// the prop's root, and the part's rest pose on it.
struct Part {
    pivot: Vec3,
    rotation: Quat,
    scale: Vec3,
    shift: Vec3,
    look: Look,
}

fn piece(pivot: Vec3, look: Look) -> Part {
    Part {
        pivot,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        shift: Vec3::ZERO,
        look,
    }
}

/// Moves `child` under `parent`, both direct children of one root.
fn reparent(city: &mut Metro, child: u32, parent: u32) {
    let parent_local = city.nodes[parent as usize].transform;
    let child_local = city.nodes[child as usize].transform;
    city.nodes[child as usize].parent = Some(parent);
    city.nodes[child as usize].transform = Transform {
        translation: parent_local.rotation.inverse()
            * (child_local.translation - parent_local.translation),
        rotation: parent_local.rotation.inverse() * child_local.rotation,
        scale: child_local.scale,
    };
}

/// A bus (kind 0), a truck (1) or a car (2) with headlights on a loop.
fn vehicle(city: &mut Metro, rng: &mut Rng, kind: u32) {
    let (center, half) = street_loop(&city.blocks, city.radius, rng);
    let direction = rng.sign();
    let lane = direction * 1.1;
    let half = half + Vec2::splat(lane);
    let model = match kind {
        0 => rng.pick(&[Model::Bus, Model::Bus, Model::DoubleDecker]),
        1 => rng.pick(&[Model::Truck, Model::Mixer]),
        _ => Model::Car(rng.pick(&[
            Car::Sedan,
            Car::Sedan,
            Car::Hatch,
            Car::Suv,
            Car::Taxi,
            Car::Van,
            Car::Pickup,
            Car::Sports,
            Car::Police,
        ])),
    };
    let paint = match model {
        Model::Car(Car::Taxi) => [1.0, 0.78, 0.08, 1.0],
        Model::Car(Car::Police) => [0.95, 0.95, 1.0, 1.0],
        Model::Car(Car::Sports) => rng.pick(&[
            [0.95, 0.95, 0.97, 1.0],
            [0.85, 0.08, 0.08, 1.0],
            [1.0, 0.75, 0.08, 1.0],
        ]),
        _ => rng.pick(&PAINTS),
    };
    let size = city.size(model);
    let motion = Motion::Travel(Travel {
        center: center + Vec3::Y * (ROAD_TOP + size.y * 0.5),
        half,
        speed: rng.range(5.0, 13.0) * direction,
        phase: rng.range(0.0, 4.0 * (half.x + half.y)),
    });
    let world = city.world();
    let body = city.mover(
        &world,
        at(Vec3::ZERO),
        look(model, Mat::Vehicle, paint),
        Some(motion),
        Some(motion),
    );
    // Headlights from the front, following the drawn pose.
    let front = v(size.x * 0.5 + 0.05, -size.y * 0.15, 0.0);
    city.extra(
        body,
        Extra::Light(
            LocalLight::spot(
                Vec3::new(1.0, 0.93, 0.8),
                14.0,
                9.0,
                Vec3::new(1.0, -0.18, 0.0),
                18.0_f32.to_radians(),
                38.0_f32.to_radians(),
            )
            .with_offset(front),
        ),
    );
}

fn drone(city: &mut Metro, rng: &mut Rng) {
    let (center, _) = street_loop(&city.blocks, city.radius, rng);
    let downtown = center.length() < city.radius * PERIOD * 0.3;
    let height = if downtown {
        rng.range(40.0, 130.0)
    } else {
        rng.range(15.0, 50.0)
    };
    let direction = rng.sign();
    let speed = rng.range(0.25, 0.9) * direction;
    let phase = rng.range(0.0, TAU);
    let orbit = Orbit {
        center: center + Vec3::Y * height,
        radius: rng.range(4.0, 26.0),
        speed,
        phase,
        bob: rng.range(0.3, 1.2),
        facing_speed: speed,
        facing_phase: phase + direction * FRAC_PI_2,
    };
    let body = look(Model::Drone, Mat::Vehicle, rng.pick(&PAINTS));
    let cargo = look(Model::Container, Mat::Container, rng.pick(&PAINTS));
    let cargo_scale = Vec3::splat(0.12);
    let drop = 0.7;
    let world = city.world();
    match city.pivot(Transform::default(), Motion::Orbit(orbit)) {
        Some(pivot) => {
            city.mover(&pivot, at(Vec3::ZERO), body, None, None);
            city.mover(
                &pivot,
                part_at(Vec3::NEG_Y * drop, cargo_scale),
                cargo,
                None,
                None,
            );
        }
        None => {
            city.mover(
                &world,
                at(Vec3::ZERO),
                body,
                None,
                Some(Motion::Orbit(orbit)),
            );
            let lowered = Orbit {
                center: orbit.center - Vec3::Y * drop,
                ..orbit
            };
            city.mover(
                &world,
                part_at(Vec3::ZERO, cargo_scale),
                cargo,
                None,
                Some(Motion::Orbit(lowered)),
            );
        }
    }
}

fn helicopter(city: &mut Metro, rng: &mut Rng) {
    let direction = rng.sign();
    let speed = rng.range(0.04, 0.12) * direction;
    let phase = rng.range(0.0, TAU);
    let orbit = Orbit {
        center: v(
            rng.range(-60.0, 60.0),
            rng.range(70.0, 150.0),
            rng.range(-60.0, 60.0),
        ),
        radius: rng.range(30.0, math::max(city.radius * PERIOD * 0.7, 40.0)),
        speed,
        phase,
        bob: rng.range(0.5, 2.0),
        facing_speed: speed,
        facing_phase: phase + direction * FRAC_PI_2,
    };
    let (rotor_speed, rotor_phase) = (rng.range(16.0, 24.0), rng.range(0.0, TAU));
    let body = look(
        Model::Heli,
        Mat::Vehicle,
        rng.pick(&[[0.9, 0.2, 0.15, 1.0], WHITE, [0.15, 0.2, 0.3, 1.0]]),
    );
    let rotor = look(Model::Rotor, Mat::Vehicle, WHITE);
    let lift = 1.0;
    let world = city.world();
    match city.pivot(Transform::default(), Motion::Orbit(orbit)) {
        Some(pivot) => {
            city.mover(&pivot, at(Vec3::ZERO), body, None, None);
            let spin = Spin {
                base: Quat::IDENTITY,
                axis_z: false,
                speed: rotor_speed,
                phase: rotor_phase,
            };
            city.mover(
                &pivot,
                at(Vec3::Y * lift),
                rotor,
                Some(Motion::Spin(spin)),
                None,
            );
        }
        None => {
            city.mover(
                &world,
                at(Vec3::ZERO),
                body,
                None,
                Some(Motion::Orbit(orbit)),
            );
            let turning = Orbit {
                center: orbit.center + Vec3::Y * lift,
                facing_speed: orbit.facing_speed + rotor_speed,
                facing_phase: orbit.facing_phase + rotor_phase,
                ..orbit
            };
            city.mover(
                &world,
                at(Vec3::ZERO),
                rotor,
                None,
                Some(Motion::Orbit(turning)),
            );
        }
    }
}

fn blimp(city: &mut Metro, rng: &mut Rng) {
    let direction = rng.sign();
    let speed = rng.range(0.01, 0.025) * direction;
    let phase = rng.range(0.0, TAU);
    let orbit = Orbit {
        center: v(
            rng.range(-40.0, 40.0),
            rng.range(160.0, 210.0),
            rng.range(-40.0, 40.0),
        ),
        radius: rng.range(60.0, math::max(city.radius * PERIOD * 0.8, 80.0)),
        speed,
        phase,
        bob: rng.range(1.0, 3.0),
        facing_speed: speed,
        facing_phase: phase + direction * FRAC_PI_2,
    };
    let world = city.world();
    let body = city.mover(
        &world,
        at(Vec3::ZERO),
        look(Model::Blimp, Mat::Vehicle, rng.pick(&PAINTS)),
        Some(Motion::Orbit(orbit)),
        Some(Motion::Orbit(orbit)),
    );
    city.extra(
        body,
        Extra::Light(LocalLight::point(Vec3::new(1.0, 0.9, 0.7), 40.0, 20.0)),
    );
}

fn crane(city: &mut Metro, rng: &mut Rng, site: &Site) {
    let top = compose(&site.anchor.frame, &at(site.at)).translation;
    let (speed, phase) = (rng.range(0.1, 0.35) * rng.sign(), rng.range(0.0, TAU));
    let base = Quat::from_rotation_y(site.yaw);
    let spin = Spin {
        base,
        axis_z: false,
        speed,
        phase,
    };
    let jib = look(Model::CraneJib, Mat::Lattice, CRANE);
    let load = look(Model::Container, Mat::Container, rng.pick(&PAINTS));
    let jib_center = city.offset(Model::CraneJib, Vec3::ZERO) * -1.0;
    let scale = Vec3::splat(site.size);
    let (reach, drop) = (14.0 * site.size, rng.range(4.0, 9.0));
    let world = city.world();
    match city.pivot(
        Transform {
            translation: top,
            rotation: base,
            scale: Vec3::ONE,
        },
        Motion::Spin(spin),
    ) {
        Some(pivot) => {
            city.mover(
                &pivot,
                part_at(jib_center * site.size, scale),
                jib,
                None,
                None,
            );
            city.mover(
                &pivot,
                part_at(v(reach, -drop, 0.0), Vec3::splat(0.35)),
                load,
                None,
                None,
            );
        }
        None => {
            // The jib's center circles the mast as the jib turns.
            let orbit_jib = Orbit {
                center: top,
                radius: (jib_center * site.size).x,
                speed,
                phase: phase + site.yaw,
                bob: 0.0,
                facing_speed: speed,
                facing_phase: phase + site.yaw,
            };
            city.mover(
                &world,
                part_at(Vec3::ZERO, scale),
                jib,
                None,
                Some(Motion::Orbit(Orbit {
                    center: top + Vec3::Y * (jib_center.y * site.size),
                    ..orbit_jib
                })),
            );
            let orbit = Orbit {
                center: top - Vec3::Y * drop,
                radius: reach,
                speed,
                phase: phase + site.yaw,
                bob: 0.0,
                facing_speed: speed,
                facing_phase: phase + site.yaw,
            };
            city.mover(
                &world,
                part_at(Vec3::ZERO, Vec3::splat(0.35)),
                load,
                None,
                Some(Motion::Orbit(orbit)),
            );
        }
    }
}

fn ferris(city: &mut Metro, rng: &mut Rng, site: &Site) {
    let hub = compose(&site.anchor.frame, &at(site.at)).translation;
    let base = Quat::from_rotation_y(site.yaw);
    let (speed, phase) = (rng.range(0.08, 0.2), rng.range(0.0, TAU));
    let spin = Spin {
        base,
        axis_z: true,
        speed,
        phase,
    };
    let rim = look(Model::FerrisRim, Mat::Fixture, WHITE);
    let radius = 6.4 * site.size;
    let scale = Vec3::splat(site.size);
    let world = city.world();
    let pivot = city.pivot(
        Transform {
            translation: hub,
            rotation: base,
            scale: Vec3::ONE,
        },
        Motion::Spin(spin),
    );
    match pivot {
        Some(pivot) => {
            city.mover(&pivot, part_at(Vec3::ZERO, scale), rim, None, None);
        }
        None => {
            city.mover(
                &world,
                Transform {
                    translation: hub,
                    rotation: base,
                    scale,
                },
                rim,
                None,
                Some(Motion::Spin(spin)),
            );
        }
    }
    for cabin in 0..8 {
        let angle = TAU * cabin as f32 / 8.0;
        let (sin, cos) = math::sin_cos(angle);
        let paint = look(Model::FerrisCabin, Mat::Fixture, rng.pick(&PASTELS));
        match pivot {
            Some(pivot) => {
                let upright = Spin {
                    base: Quat::IDENTITY,
                    axis_z: true,
                    speed: -speed,
                    phase: -phase,
                };
                city.mover(
                    &pivot,
                    part_at(v(radius * cos, radius * sin, 0.0), scale),
                    paint,
                    Some(Motion::Spin(upright)),
                    None,
                );
            }
            None => {
                let wheel = Wheel {
                    center: hub,
                    u: base * Vec3::X * radius,
                    v: Vec3::Y * radius,
                    speed,
                    phase: phase + angle,
                };
                city.mover(
                    &world,
                    Transform {
                        translation: hub,
                        rotation: base,
                        scale,
                    },
                    paint,
                    None,
                    Some(Motion::Wheel(wheel)),
                );
            }
        }
    }
}

fn smoke(seed: u32) -> ParticleEmitter {
    ParticleEmitter {
        rate: 3.0,
        lifetime: 6.0,
        speed: 1.2,
        direction: Vec3::Y,
        spread: 10.0_f32.to_radians(),
        acceleration: Vec3::new(0.6, 0.2, 0.25),
        size: [1.0, 5.5],
        color: [[0.5, 0.5, 0.52, 0.5], [0.72, 0.72, 0.75, 0.0]],
        emission: 1.0,
        blend: ParticleBlend::Alpha,
        stretch: 0.0,
        offset: Vec3::ZERO,
        radius: 0.25,
        seed,
    }
}

fn spray(seed: u32) -> ParticleEmitter {
    ParticleEmitter {
        rate: 60.0,
        lifetime: 1.2,
        speed: 5.0,
        direction: Vec3::Y,
        spread: 10.0_f32.to_radians(),
        acceleration: Vec3::new(0.0, -9.8, 0.0),
        size: [0.12, 0.06],
        color: [[0.8, 0.92, 1.0, 0.6], [0.85, 0.93, 1.0, 0.0]],
        emission: 1.0,
        blend: ParticleBlend::Alpha,
        stretch: 0.05,
        offset: Vec3::ZERO,
        radius: 0.05,
        seed,
    }
}

fn steam(seed: u32) -> ParticleEmitter {
    ParticleEmitter {
        rate: 6.0,
        lifetime: 2.5,
        speed: 0.8,
        direction: Vec3::Y,
        spread: 20.0_f32.to_radians(),
        acceleration: Vec3::new(0.2, 0.3, 0.1),
        size: [0.4, 2.2],
        color: [[0.9, 0.9, 0.92, 0.35], [0.95, 0.95, 0.97, 0.0]],
        emission: 1.0,
        blend: ParticleBlend::Alpha,
        stretch: 0.0,
        offset: Vec3::ZERO,
        radius: 0.2,
        seed,
    }
}

fn welding(seed: u32) -> ParticleEmitter {
    ParticleEmitter {
        rate: 40.0,
        lifetime: 0.7,
        speed: 3.0,
        direction: Vec3::Y,
        spread: 70.0_f32.to_radians(),
        acceleration: Vec3::new(0.0, -9.8, 0.0),
        size: [0.08, 0.02],
        color: [[1.0, 0.8, 0.4, 1.0], [1.0, 0.4, 0.1, 0.0]],
        emission: 3.0,
        blend: ParticleBlend::Additive,
        stretch: 0.08,
        offset: Vec3::ZERO,
        radius: 0.05,
        seed,
    }
}

/// Billboard ads: every slot's material and the pictures it cycles through.
#[derive(Resource)]
pub(crate) struct Adverts {
    slots: Vec<(MaterialHandle, Vec<Material>)>,
    shown: u64,
}

/// Replaces each board slot's material with its next advertisement every
/// `AD_PERIOD` seconds of simulated time.
fn rotate_ads(time: Res<FixedTime>, mut ads: ResMut<Adverts>, mut assets: ResMut<AssetLibrary>) {
    let cycle = (time.elapsed_seconds / AD_PERIOD) as u64;
    if cycle == ads.shown {
        return;
    }
    ads.shown = cycle;
    for (slot, (handle, pictures)) in ads.slots.iter().enumerate() {
        let picture = pictures[(slot + cycle as usize) % pictures.len()];
        // A failure keeps the previous picture: the handle lives as long as
        // the workload.
        let _ = assets.replace_material(*handle, picture);
    }
}

/// Registers the metro's motion and advertising systems.
pub(crate) fn add_systems(app: &mut App) {
    crate::kit::motion::add_systems(app);
    app.add_systems(FixedSimulation, rotate_ads.after(Animation));
}

/// The spawned metro, for the workload's observations.
pub(crate) struct Spawned {
    pub transform_nodes: u32,
    pub moving_nodes: u32,
    pub sensor_nodes: u32,
    pub probes: Vec<Entity>,
    pub bounds: (Vec3, Vec3),
}

/// Builds and spawns the metro of `config` into `world`.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(crate) fn populate(world: &mut World, config: &StressConfig) -> engine::Result<Spawned> {
    let mut city = build(config);
    let mut assets = world.resource_mut::<AssetLibrary>();
    let mut library = Library::new(
        config.mesh_variety,
        config.textured,
        config.compressed_textures,
    );
    // Meshes, materials and the size stand-ins scale to.
    let mut drawn = Vec::with_capacity(city.nodes.len());
    for node in &city.nodes {
        let Some(look) = node.look else {
            drawn.push(None);
            continue;
        };
        let (mesh, size) = library.mesh(&mut assets, look.model)?;
        let material = library.material(&mut assets, look.mat)?;
        drawn.push(Some((Mesh3d { mesh, material }, size)));
    }
    let mut pictures = Vec::with_capacity(AD_SLOTS as usize);
    for slot in 0..AD_SLOTS {
        let handle = library.material(&mut assets, Mat::Ad(slot))?;
        let ads = (0..textures::ADS)
            .map(|ad| library.advert(&mut assets, slot, ad))
            .collect::<Result<Vec<_>, _>>()?;
        pictures.push((handle, ads));
    }
    // Each clip once, in the order the city's players first name them, so
    // the library holds them in the same slots every run.
    let mut named: Vec<Clip> = Vec::new();
    for (_, extra) in &city.extras {
        if let Extra::Player { clip, .. } = extra
            && !named.contains(clip)
        {
            named.push(*clip);
        }
    }
    let clips: HashMap<Clip, ClipHandle> = named
        .into_iter()
        .map(|clip| library.clip(&mut assets, clip).map(|handle| (clip, handle)))
        .collect::<Result<_, _>>()?;
    let renderables = drawn.iter().filter(|mesh| mesh.is_some()).count() as u32;
    debug_assert_eq!(renderables, config.renderables, "renderables as configured");
    let dense = if config.dense_colliders {
        config.colliders
    } else {
        0
    };
    let mut entities = Vec::with_capacity(city.nodes.len());
    let mut parents = Vec::new();
    let mut probes = Vec::with_capacity(config.colliders as usize);
    let (mut leaf, mut moving_nodes, mut sensor_nodes) = (0_u32, 0_u32, 0_u32);
    let nodes = std::mem::take(&mut city.nodes);
    for (node, mesh) in nodes.iter().zip(drawn) {
        let mut transform = node
            .motion
            .map_or(node.transform, |motion| motion.apply(node.transform, 0.0));
        if let (Some(look), Some((_, drawn_size))) = (node.look, mesh.as_ref()) {
            // Stand-ins of fewer meshes take the size of what they replace,
            // except along flat axes.
            if look.model.resolve(config.mesh_variety) != look.model {
                let size = city.size(look.model);
                let ratio = Vec3::select(
                    drawn_size.cmpgt(Vec3::splat(1.0e-3)),
                    size / *drawn_size,
                    Vec3::ONE,
                );
                transform.scale *= Vec3::select(ratio.cmpgt(Vec3::splat(1.0e-3)), ratio, Vec3::ONE);
            }
        }
        let mut entity = world.spawn(transform);
        if let Some(motion) = node.motion {
            match motion {
                Motion::Travel(travel) => entity.insert(travel),
                Motion::Orbit(orbit) => entity.insert(orbit),
                Motion::Wheel(wheel) => entity.insert(wheel),
                Motion::Spin(spin) => entity.insert(spin),
                Motion::Slide(slide) => entity.insert(slide),
            };
            if config.interpolation {
                entity.insert(InterpolateTransform);
            }
        }
        if let (Some((drawn_mesh, _)), Some(look)) = (mesh, node.look) {
            entity.insert(drawn_mesh);
            if look.tint != WHITE {
                entity.insert(Tint(look.tint));
            }
            moving_nodes += u32::from(node.moving);
            let collider = if dense > 0 {
                leaf < dense
            } else {
                u64::from(leaf + 1) * u64::from(config.colliders) / u64::from(config.renderables)
                    > u64::from(leaf) * u64::from(config.colliders) / u64::from(config.renderables)
            };
            if collider {
                let sensor = probes.len() as u32 % 100 < config.sensor_percent;
                let size = city.size(look.model);
                let shape = if look.model.round() {
                    Collider::sphere(math::max(math::max(size.x, size.y), size.z) * 0.5)
                } else {
                    Collider::cuboid(size.max(Vec3::splat(0.02)) * 0.5)
                }?
                .with_sensor(sensor);
                entity.insert(shape);
                probes.push(entity.id());
                sensor_nodes += u32::from(sensor);
            }
            leaf += 1;
        }
        let id = entity.id();
        if let Some(parent) = node.parent {
            parents.push((id, Some(entities[parent as usize])));
        }
        entities.push(id);
    }
    set_parents(world, &parents)?;
    // Components naming other nodes, once every node exists.
    for (node, extra) in &city.extras {
        let entity = entities[*node as usize];
        let mut target = world.entity_mut(entity);
        match extra {
            Extra::Light(light) => {
                target.insert(*light);
            }
            Extra::Emitter(emitter) => {
                target.insert(*emitter);
            }
            Extra::Player {
                clip,
                targets,
                time,
                speed,
            } => {
                let mut player = AnimationPlayer::new(
                    clips[clip],
                    targets
                        .iter()
                        .map(|target| Some(entities[*target as usize]))
                        .collect(),
                );
                player.time = *time;
                player.speed = *speed;
                target.insert(player);
            }
            Extra::Skin(joints) => {
                target.insert(Skin::new(
                    joints
                        .iter()
                        .map(|joint| entities[*joint as usize])
                        .collect(),
                    people::inverse_binds(),
                ));
            }
            Extra::Animated => {
                if config.interpolation {
                    target.insert(InterpolateTransform);
                }
            }
            Extra::Advert => {}
        }
    }
    world.insert_resource(Adverts {
        slots: pictures,
        shown: u64::MAX,
    });
    // Reflection probes at every street crossing (blocks'
    // corners lie on multiples of the period), at the
    // height of cars and people: their boxes overlap their neighbors' by
    // twice the blend so one fades out as the next is fully in, and surfaces
    // reflect the crossing nearest them.
    let crossings: std::collections::BTreeSet<(i32, i32)> = city
        .blocks
        .iter()
        .flat_map(|&(bx, bz, _)| [(bx, bz), (bx + 1, bz), (bx, bz + 1), (bx + 1, bz + 1)])
        .collect();
    for (cx, cz) in crossings {
        world.spawn((
            Transform {
                translation: v(cx as f32 * PERIOD, PROBE_HEIGHT, cz as f32 * PERIOD),
                ..Transform::default()
            },
            ReflectionProbe::new(Vec3::splat(PERIOD * 0.5 + 2.0 * PROBE_BLEND))
                .with_blend(PROBE_BLEND),
        ));
    }
    let ring = city
        .blocks
        .iter()
        .map(|&(bx, bz, _)| bx.abs().max(bz.abs()))
        .max()
        .unwrap_or(0);
    let half = (ring as f32 + 1.5) * PERIOD;
    Ok(Spawned {
        transform_nodes: entities.len() as u32,
        moving_nodes,
        sensor_nodes,
        probes,
        bounds: (v(-half, -2.0, -half), v(half, 300.0, half)),
    })
}
