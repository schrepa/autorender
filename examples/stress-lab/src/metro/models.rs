//! Low-poly models of the metro, built at their nominal world size and
//! centered on their bounds (skinned people stand on their origin). Each
//! model's texture coordinates suit one family of texture sets: façades
//! repeat per 8 bays and 8 floors, tiling ground and metal sets repeat in
//! world units, and atlases (street fixtures, vehicles) map faces into
//! uniform regions.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use engine::prelude::*;

use super::builder::{Builder, Map, Uv1};
use super::people;

/// Width of a façade bay and height of a floor, in world units.
pub(crate) const BAY: f32 = 1.4;
pub(crate) const FLOOR: f32 = 1.8;
/// Façade textures repeat every 8 bays and 8 floors.
const FACADE: Map = Map::Tile(BAY * 8.0, FLOOR * 8.0);
/// A wall-colored spot of every façade texture, for roofs and ledges.
const WALL: Map = Map::patch(0.002, 0.002);

/// Footprints of towers, midrises and houses.
pub(crate) const TOWER_WIDTH: f32 = 6.4;
pub(crate) const MID_WIDTH: f32 = 4.8;
pub(crate) const HOUSE_WIDTH: f32 = 3.2;
pub(crate) const TOWER_FLOORS: [u32; 5] = [20, 30, 42, 56, 72];
pub(crate) const MID_FLOORS: [u32; 4] = [4, 7, 11, 16];

/// Region `(col, row)` of a 4 × 4 atlas, inset from its edges.
const fn cell(col: u32, row: u32) -> Map {
    let (u, v) = (col as f32 * 0.25, row as f32 * 0.25);
    Map::Rect(u + 0.04, v + 0.04, u + 0.21, v + 0.21)
}

// Street fixture atlas regions (textures::fixture).
const DARK: Map = cell(0, 0);
const STEEL: Map = cell(1, 0);
const CHROME: Map = cell(2, 0);
const WARM: Map = cell(3, 0);
const COOL: Map = cell(0, 1);
const RED: Map = cell(1, 1);
const GREEN: Map = cell(2, 1);
const AMBER: Map = cell(3, 1);
const SIGN: Map = cell(0, 2);
const HAZARD: Map = cell(2, 2);
const TIMBER: Map = cell(3, 2);
const PAINTED: Map = cell(1, 3);
const STONE: Map = cell(2, 3);

// Vehicle atlas regions (textures::vehicle).
const PAINT: Map = Map::Rect(0.03, 0.03, 0.47, 0.22);
const WINDOW: Map = cell(2, 0);
const TRIM: Map = cell(3, 0);
const HEADLIGHT: Map = cell(0, 1);
const TAILLIGHT: Map = cell(1, 1);
const TIRE: Map = cell(2, 1);
const GRILLE: Map = cell(3, 1);
const SIREN_BLUE: Map = cell(0, 2);
const SIREN_RED: Map = cell(1, 2);
const INTERIOR: Map = cell(2, 2);
const TAXI_SIGN: Map = cell(3, 2);

// Foliage textures: leaves above, bark in the bottom fifth.
const CANOPY: Map = Map::Rect(0.0, 0.0, 1.0, 0.8);
const DENSE: Map = Map::Rect(0.3, 0.25, 0.7, 0.55);
const TRUNK: Map = Map::Rect(0.0, 0.82, 1.0, 1.0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Style {
    Box,
    Setback,
    Round,
    Twisted,
    Tapered,
    Hex,
    Deco,
    Fins,
    Slant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Mid {
    Block,
    Ell,
    Court,
    Walkup,
    Terrace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Car {
    Sedan,
    Hatch,
    Suv,
    Taxi,
    Van,
    Pickup,
    Sports,
    Police,
}

/// Every model of the metro.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Model {
    // Buildings: façade-textured, with second coordinates for grime.
    Tower(Style, u8),
    Midrise(Mid, u8),
    House(u8),
    Townhouse,
    Shop,
    Warehouse,
    Shed,
    // Roofs and tops.
    RoofSlab(u8),
    Gable,
    Hip,
    Spire,
    Crown,
    Dome,
    Helipad,
    // Industry.
    Tank,
    Silo,
    Stack,
    Container,
    // Building details.
    AcUnit,
    WaterTank,
    Antenna,
    DishPost,
    DishHead,
    Solar,
    Railing(u8),
    Billboard,
    Trivision,
    NeonBlade,
    Awning,
    Balcony,
    BalconyRail,
    FireEscape,
    Gondola,
    Flagpole,
    Flag,
    GlassBox(u8),
    DoorCanopy,
    DoorWings,
    // Streets.
    Road,
    RoadCross,
    Sidewalk,
    Ground,
    Lamp,
    TrafficLight,
    ShelterFrame,
    ShelterGlass,
    Bench,
    Bin,
    Hydrant,
    Bollard,
    Planter,
    Flowers,
    KioskBody,
    KioskRoof,
    BoothFrame,
    BoothGlass,
    Umbrella,
    CafeTable,
    FoodCart,
    Plinth,
    Figure,
    Sculpture,
    Basin,
    Pool,
    Hedge,
    Manhole,
    StreetSign,
    // Nature.
    Tree(u8),
    Poplar,
    Pine,
    Palm,
    Bush,
    Rock,
    // Vehicles and machines.
    Car(Car),
    Bus,
    DoubleDecker,
    Truck,
    Mixer,
    MixerDrum,
    TrainCar,
    Locomotive,
    Speedboat,
    Ferry,
    Barge,
    Sailboat,
    Sail,
    Drone,
    Heli,
    Rotor,
    Blimp,
    ExcavatorBody,
    ExcavatorBoom,
    ExcavatorStick,
    CraneMast,
    CraneJib,
    FerrisRim,
    FerrisCabin,
    FerrisLeg,
    CarouselBase,
    CarouselTop,
    TurbineTower,
    TurbineRotor,
    Beacon,
    // Infrastructure and landmarks.
    RailDeck,
    RailPillar,
    StationFrame,
    StationRoof,
    BridgeDeck,
    BridgeArch,
    BridgeLeaf,
    Dock,
    Stadium,
    StadiumRoof,
    Pitch,
    Floodlight,
    TvTower,
    ObsPod,
    TowerRing,
    // Skinned people.
    Person(u8),
}

impl Model {
    /// The primitive standing in for this model when a workload keeps fewer
    /// meshes (`mesh_variety` below 64): eight shared stand-ins, or the
    /// cube at 1. People keep their skinned mesh.
    pub(crate) fn resolve(self, variety: u32) -> Model {
        use Model::*;
        if variety >= 64 || matches!(self, Person(_)) {
            return self;
        }
        if variety <= 1 {
            return Container;
        }
        match self {
            Tower(..) => Tower(Style::Box, 2),
            Midrise(..) | House(_) | Townhouse | Shop | Warehouse | Shed => Midrise(Mid::Block, 1),
            Road | RoadCross | Sidewalk | Ground | BridgeDeck | RailDeck | Dock | Pitch | Pool
            | Helipad | RoofSlab(_) => Road,
            Tree(_) | Poplar | Pine | Palm | Bush | Hedge | Flowers | Planter | Rock => Tree(0),
            Car(_) | Bus | DoubleDecker | Truck | Mixer | TrainCar | Locomotive | Speedboat
            | Ferry | Barge | Sailboat | FoodCart | Drone | Heli | Blimp | ExcavatorBody => {
                Car(self::Car::Sedan)
            }
            Lamp | TrafficLight | Flagpole | StreetSign | Bollard | Antenna | Floodlight
            | RailPillar | TurbineTower | CraneMast | Spire | Stack | Silo | TvTower => Lamp,
            Tank | WaterTank | Bin | Hydrant | Basin | Dome | ObsPod | Sculpture | Crown
            | Beacon | TowerRing | CarouselBase | CarouselTop | Umbrella | Hip | Gable => Dome,
            _ => Container,
        }
    }

    /// Whether a sphere collider fits this model better than a box.
    pub(crate) fn round(self) -> bool {
        use Model::*;
        matches!(
            self,
            Tree(_)
                | Poplar
                | Pine
                | Palm
                | Bush
                | Rock
                | Dome
                | ObsPod
                | Sculpture
                | Tank
                | Basin
                | FerrisRim
                | Blimp
        )
    }

    pub(crate) fn build(self) -> Result<Mesh, String> {
        self.assemble().map(|(mesh, _)| mesh)
    }

    /// The mesh and the build-space point at its origin: pivots of animated
    /// parts and light positions are given in build space.
    pub(crate) fn assemble(self) -> Result<(Mesh, Vec3), String> {
        let mut b = Builder::default();
        match self {
            Model::Tower(style, class) => tower(&mut b, style, TOWER_FLOORS[class as usize]),
            Model::Midrise(style, class) => midrise(&mut b, style, MID_FLOORS[class as usize]),
            Model::House(floors) => {
                building_frame(&mut b);
                let height = FLOOR * floors as f32;
                let half = HOUSE_WIDTH * 0.5;
                b.top = Some(height);
                b.cuboid(
                    v(-half, 0.0, -half),
                    v(half, height, half),
                    FACADE,
                    WALL,
                    false,
                );
            }
            Model::Townhouse => {
                building_frame(&mut b);
                for (index, floors) in [3.0, 4.0, 3.0].into_iter().enumerate() {
                    let height = FLOOR * floors;
                    b.top = Some(height);
                    let x0 = -3.2 + index as f32 * 2.1334;
                    b.cuboid(
                        v(x0, 0.0, -1.6),
                        v(x0 + 2.1334, height, 1.6),
                        FACADE,
                        WALL,
                        false,
                    );
                    // A cornice along the front.
                    b.cuboid(
                        v(x0, height - 0.3, 1.6),
                        v(x0 + 2.1334, height - 0.05, 1.8),
                        WALL,
                        WALL,
                        true,
                    );
                }
            }
            Model::Shop => {
                building_frame(&mut b);
                b.top = Some(FLOOR * 2.0);
                b.cuboid(
                    v(-2.4, 0.0, -1.6),
                    v(2.4, FLOOR * 2.0, 1.6),
                    FACADE,
                    WALL,
                    false,
                );
                // A sign band over the shop window.
                b.cuboid(
                    v(-2.4, FLOOR * 0.95, 1.6),
                    v(2.4, FLOOR * 1.2, 1.85),
                    WALL,
                    WALL,
                    true,
                );
            }
            Model::Warehouse => {
                building_frame(&mut b);
                let height = FLOOR * 3.0;
                b.top = Some(height);
                b.cuboid(v(-7.0, 0.0, -3.5), v(7.0, height, 3.5), FACADE, WALL, false);
                // Sawtooth roof lights along X.
                for tooth in 0..4 {
                    let z0 = -3.5 + tooth as f32 * 1.75;
                    b.ridge(
                        -7.0,
                        7.0,
                        [(z0, height), (z0 + 1.75, height), (z0, height + 1.3)],
                        FACADE,
                        FACADE,
                    );
                }
            }
            Model::Shed => {
                building_frame(&mut b);
                let height = FLOOR * 2.0;
                b.top = Some(height);
                b.cuboid(v(-5.0, 0.0, -3.0), v(5.0, height, 3.0), FACADE, WALL, false);
                // A barrel vault along X.
                let steps = 8;
                let arc = |index: u32| {
                    let angle = PI * index as f32 / steps as f32;
                    let (sin, cos) = math::sin_cos(angle);
                    (3.0 * cos, height + 1.6 * sin)
                };
                for index in 0..steps {
                    let ((z0, y0), (z1, y1)) = (arc(index), arc(index + 1));
                    let outward = v(0.0, (y0 + y1) * 0.5 - height, (z0 + z1) * 0.5);
                    b.quad(
                        [
                            v(-5.0, y0, z0),
                            v(5.0, y0, z0),
                            v(5.0, y1, z1),
                            v(-5.0, y1, z1),
                        ],
                        FACADE,
                        outward,
                    );
                }
                for x in [-5.0, 5.0] {
                    let corners: Vec<Vec3> = (0..=steps)
                        .map(|index| {
                            let (z, y) = arc(index);
                            v(x, y, z)
                        })
                        .collect();
                    b.poly(&corners, FACADE, v(x, 0.0, 0.0));
                }
            }
            Model::RoofSlab(size) => {
                let half = [TOWER_WIDTH, MID_WIDTH, HOUSE_WIDTH][size as usize] * 0.5 - 0.3;
                let roof = Map::Tile(8.0, 8.0);
                b.cuboid(v(-half, 0.0, -half), v(half, 0.35, half), roof, roof, false);
            }
            Model::Gable => {
                let roof = Map::Tile(4.0, 4.0);
                b.ridge(
                    -1.8,
                    1.8,
                    [(-1.8, 0.0), (1.8, 0.0), (0.0, 1.6)],
                    roof,
                    Map::Rect(0.0, 0.0, 1.0, 0.5),
                );
            }
            Model::Hip => {
                b.lathe(
                    &[(2.5, 0.0), (0.0, 1.5)],
                    4,
                    FRAC_PI_4,
                    Map::Tile(4.0, 4.0),
                    Map::Tile(4.0, 4.0),
                    false,
                );
            }
            Model::Spire => {
                b.lathe(
                    &[(1.4, 0.0), (1.0, 2.0), (0.25, 10.0), (0.0, 14.0)],
                    8,
                    0.0,
                    Map::Tile(2.0, 2.0),
                    Map::Tile(2.0, 2.0),
                    false,
                );
            }
            Model::Crown => {
                b.lathe(
                    &[
                        (3.0, 0.0),
                        (3.0, 1.0),
                        (2.4, 1.0),
                        (2.4, 3.0),
                        (1.8, 3.0),
                        (1.8, 5.0),
                        (1.0, 5.0),
                        (0.6, 8.0),
                        (0.0, 9.0),
                    ],
                    8,
                    PI / 8.0,
                    Map::Tile(2.0, 2.0),
                    Map::Tile(2.0, 2.0),
                    false,
                );
            }
            Model::Dome => {
                let profile: Vec<(f32, f32)> = (0..=6)
                    .map(|index| {
                        let angle = FRAC_PI_2 * index as f32 / 6.0;
                        let (sin, cos) = math::sin_cos(angle);
                        (2.5 * cos, 2.0 * sin)
                    })
                    .collect();
                b.lathe(
                    &profile,
                    14,
                    0.0,
                    Map::Tile(2.0, 2.0),
                    Map::Tile(2.0, 2.0),
                    false,
                );
            }
            Model::Helipad => {
                b.lathe(
                    &[(2.8, 0.0), (2.8, 0.15)],
                    16,
                    0.0,
                    Map::Rect(0.0, 0.0, 1.0, 0.1),
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    false,
                );
            }
            Model::Tank => {
                b.lathe(
                    &[(2.2, 0.0), (2.2, 6.0), (1.6, 6.8), (0.0, 7.2)],
                    14,
                    0.0,
                    Map::Tile(4.0, 4.0),
                    Map::Tile(4.0, 4.0),
                    false,
                );
                // A ladder up its side.
                b.cuboid(
                    v(2.2, 0.0, -0.3),
                    v(2.35, 6.0, 0.3),
                    Map::Tile(1.0, 1.0),
                    Map::Tile(1.0, 1.0),
                    false,
                );
            }
            Model::Silo => {
                b.lathe(
                    &[(1.5, 0.0), (1.5, 12.0), (0.9, 13.4), (0.0, 13.8)],
                    12,
                    0.0,
                    Map::Tile(4.0, 4.0),
                    Map::Tile(4.0, 4.0),
                    false,
                );
            }
            Model::Stack => {
                b.lathe(
                    &[(1.0, 0.0), (0.7, 26.0), (0.85, 26.4), (0.85, 27.0)],
                    10,
                    0.0,
                    Map::Tile(4.0, 4.0),
                    Map::Tile(4.0, 4.0),
                    false,
                );
            }
            Model::Container => {
                let side = Map::Rect(0.0, 0.0, 1.0, 1.0);
                b.cuboid(v(-3.05, 0.0, -1.2), v(3.05, 2.6, 1.2), side, side, true);
            }
            Model::AcUnit => {
                b.cuboid(v(-0.6, 0.0, -0.45), v(0.6, 0.8, 0.45), STEEL, STEEL, false);
                let start = b.mark();
                b.lathe(&[(0.3, 0.8), (0.3, 0.85)], 10, 0.0, DARK, DARK, false);
                b.place(start, Mat4::from_translation(v(0.2, 0.0, 0.0)));
            }
            Model::WaterTank => {
                let wood = Map::Tile(2.0, 2.0);
                for (x, z) in [(0.6, 0.6), (-0.6, 0.6), (-0.6, -0.6), (0.6, -0.6)] {
                    b.cuboid(
                        v(x - 0.08, 0.0, z - 0.08),
                        v(x + 0.08, 1.4, z + 0.08),
                        wood,
                        wood,
                        false,
                    );
                }
                b.lathe(
                    &[(0.95, 1.4), (0.95, 3.0), (1.05, 3.0), (0.0, 3.8)],
                    10,
                    0.0,
                    wood,
                    wood,
                    true,
                );
            }
            Model::Antenna => {
                b.bar(v(0.0, 0.0, 0.0), v(0.0, 9.0, 0.0), 0.12, 6, STEEL);
                for (height, width) in [(3.0, 1.4), (5.5, 1.0), (7.5, 0.6)] {
                    b.block(
                        v(-width, height, -0.04),
                        v(width, height + 0.08, 0.04),
                        STEEL,
                    );
                }
                b.lathe(
                    &[(0.0, 9.0), (0.18, 9.2), (0.0, 9.4)],
                    6,
                    0.0,
                    RED,
                    RED,
                    false,
                );
            }
            Model::DishPost => {
                b.bar(v(0.0, 0.0, 0.0), v(0.0, 1.2, 0.0), 0.08, 6, STEEL);
                b.block(v(-0.3, 0.0, -0.3), v(0.3, 0.12, 0.3), DARK);
            }
            Model::DishHead => {
                // A dish facing +X around its pivot.
                let start = b.mark();
                b.lathe(
                    &[(0.0, -0.25), (0.45, -0.18), (0.8, 0.0), (0.82, 0.03)],
                    12,
                    0.0,
                    CHROME,
                    CHROME,
                    false,
                );
                b.bar(v(0.0, -0.2, 0.0), v(0.0, 0.5, 0.0), 0.03, 4, STEEL);
                b.place(start, Mat4::from_rotation_z(-FRAC_PI_2));
            }
            Model::Solar => {
                let start = b.mark();
                b.cuboid(
                    v(-1.5, -0.04, -0.8),
                    v(1.5, 0.04, 0.8),
                    Map::Rect(0.0, 0.0, 1.0, 0.03),
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    true,
                );
                b.place(
                    start,
                    Mat4::from_translation(v(0.0, 0.9, 0.0)) * Mat4::from_rotation_x(-0.45),
                );
                for x in [-1.3, 1.3] {
                    b.cuboid(
                        v(x - 0.05, 0.0, -0.4),
                        v(x + 0.05, 0.75, -0.3),
                        Map::Rect(0.0, 0.0, 0.02, 0.02),
                        Map::Rect(0.0, 0.0, 0.02, 0.02),
                        false,
                    );
                    b.cuboid(
                        v(x - 0.05, 0.0, 0.3),
                        v(x + 0.05, 1.1, 0.4),
                        Map::Rect(0.0, 0.0, 0.02, 0.02),
                        Map::Rect(0.0, 0.0, 0.02, 0.02),
                        false,
                    );
                }
            }
            Model::Railing(size) => {
                // A see-through fence around a roof: four upright sheets.
                let half = [TOWER_WIDTH, MID_WIDTH, HOUSE_WIDTH][size as usize] * 0.5 - 0.15;
                let height = 1.0;
                for (a, b2, outward) in [
                    (v(-half, 0.0, half), v(half, 0.0, half), Vec3::Z),
                    (v(half, 0.0, -half), v(-half, 0.0, -half), Vec3::NEG_Z),
                    (v(half, 0.0, half), v(half, 0.0, -half), Vec3::X),
                    (v(-half, 0.0, -half), v(-half, 0.0, half), Vec3::NEG_X),
                ] {
                    let repeats = (a - b2).length() / 2.0;
                    b.quad(
                        [a, b2, b2 + Vec3::Y * height, a + Vec3::Y * height],
                        Map::Rect(0.0, 0.0, repeats, 0.5),
                        outward,
                    );
                }
            }
            Model::Billboard => {
                // Posts, then the face along X showing the whole picture.
                let frame = Map::patch(0.002, 0.002);
                for x in [-1.3, 1.3] {
                    b.cuboid(
                        v(x - 0.1, 0.0, -0.1),
                        v(x + 0.1, 2.0, 0.1),
                        frame,
                        frame,
                        false,
                    );
                }
                b.cuboid(v(-2.2, 2.0, -0.12), v(2.2, 4.2, 0.12), frame, frame, true);
                b.quad(
                    [
                        v(-2.1, 2.1, 0.121),
                        v(2.1, 2.1, 0.121),
                        v(2.1, 4.1, 0.121),
                        v(-2.1, 4.1, 0.121),
                    ],
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    Vec3::Z,
                );
                b.quad(
                    [
                        v(2.1, 2.1, -0.121),
                        v(-2.1, 2.1, -0.121),
                        v(-2.1, 4.1, -0.121),
                        v(2.1, 4.1, -0.121),
                    ],
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    Vec3::NEG_Z,
                );
            }
            Model::Trivision => {
                // A three-sided prism along X turning about its axis.
                for face in 0..3 {
                    let angle = |step: f32| TAU * (face as f32 + step) / 3.0;
                    let at = |step: f32, x: f32| {
                        let (sin, cos) = math::sin_cos(angle(step));
                        v(x, cos * 0.9, sin * 0.9)
                    };
                    let (sin, cos) = math::sin_cos(angle(0.5));
                    b.quad(
                        [at(1.0, -2.0), at(1.0, 2.0), at(0.0, 2.0), at(0.0, -2.0)],
                        Map::Rect(0.0, 0.0, 1.0, 1.0),
                        v(0.0, cos, sin),
                    );
                }
                for x in [-2.0, 2.0] {
                    let corners: Vec<Vec3> = (0..3)
                        .map(|index| {
                            let (sin, cos) = math::sin_cos(TAU * index as f32 / 3.0);
                            v(x, cos * 0.9, sin * 0.9)
                        })
                        .collect();
                    b.poly(&corners, Map::patch(0.002, 0.002), v(x, 0.0, 0.0));
                }
            }
            Model::NeonBlade => {
                let face = Map::Rect(0.02, 0.02, 0.48, 0.98);
                b.cuboid(
                    v(-0.12, 0.0, -0.7),
                    v(0.12, 3.2, 0.7),
                    face,
                    Map::patch(0.01, 0.01),
                    true,
                );
            }
            Model::Awning => {
                // A sloping canvas over a shop front, open below.
                let canvas = Map::Rect(0.0, 0.0, 3.0, 1.0);
                b.quad(
                    [
                        v(-1.6, 0.0, 1.0),
                        v(1.6, 0.0, 1.0),
                        v(1.6, 0.6, 0.0),
                        v(-1.6, 0.6, 0.0),
                    ],
                    canvas,
                    v(0.0, 1.0, 0.6),
                );
                b.quad(
                    [
                        v(-1.6, -0.25, 1.0),
                        v(1.6, -0.25, 1.0),
                        v(1.6, 0.0, 1.0),
                        v(-1.6, 0.0, 1.0),
                    ],
                    canvas,
                    Vec3::Z,
                );
                for x in [-1.6, 1.6] {
                    b.poly(
                        &[v(x, 0.0, 0.0), v(x, 0.0, 1.0), v(x, 0.6, 0.0)],
                        canvas,
                        v(x, 0.0, 0.0),
                    );
                }
            }
            Model::Balcony => {
                let concrete = Map::Tile(2.0, 2.0);
                b.block(v(-1.2, 0.0, 0.0), v(1.2, 0.15, 1.0), concrete);
            }
            Model::BalconyRail => {
                let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
                b.quad(
                    [
                        v(-1.2, 0.0, 1.0),
                        v(1.2, 0.0, 1.0),
                        v(1.2, 0.8, 1.0),
                        v(-1.2, 0.8, 1.0),
                    ],
                    glass,
                    Vec3::Z,
                );
                b.quad(
                    [
                        v(1.2, 0.0, 1.0),
                        v(1.2, 0.0, 0.0),
                        v(1.2, 0.8, 0.0),
                        v(1.2, 0.8, 1.0),
                    ],
                    glass,
                    Vec3::X,
                );
                b.quad(
                    [
                        v(-1.2, 0.0, 0.0),
                        v(-1.2, 0.0, 1.0),
                        v(-1.2, 0.8, 1.0),
                        v(-1.2, 0.8, 0.0),
                    ],
                    glass,
                    Vec3::NEG_X,
                );
            }
            Model::FireEscape => {
                // Landings every two floors joined by stair flights, all
                // see-through lattice.
                let lattice = Map::Rect(0.0, 0.0, 2.0, 1.0);
                for level in 0..3 {
                    let y = level as f32 * FLOOR * 2.0 + FLOOR;
                    b.quad(
                        [
                            v(-1.5, y, 0.9),
                            v(1.5, y, 0.9),
                            v(1.5, y, 0.0),
                            v(-1.5, y, 0.0),
                        ],
                        lattice,
                        Vec3::Y,
                    );
                    b.quad(
                        [
                            v(-1.5, y, 0.9),
                            v(1.5, y, 0.9),
                            v(1.5, y + 0.8, 0.9),
                            v(-1.5, y + 0.8, 0.9),
                        ],
                        lattice,
                        Vec3::Z,
                    );
                    if level < 2 {
                        let (y0, y1) = (y, y + FLOOR * 2.0);
                        b.quad(
                            [
                                v(-1.3, y0, 0.45),
                                v(1.3, y1, 0.45),
                                v(1.3, y1, 0.0),
                                v(-1.3, y0, 0.0),
                            ],
                            lattice,
                            Vec3::Y,
                        );
                    }
                }
            }
            Model::Gondola => {
                b.block(v(-1.2, 0.0, 0.0), v(1.2, 0.12, 0.7), HAZARD);
                b.block(v(-1.2, 0.12, 0.6), v(1.2, 0.6, 0.7), STEEL);
                for x in [-1.1, 1.1] {
                    b.bar(v(x, 0.6, 0.35), v(x, 2.5, 0.2), 0.02, 4, STEEL);
                }
            }
            Model::Flagpole => {
                b.bar(v(0.0, 0.0, 0.0), v(0.0, 7.0, 0.0), 0.07, 6, CHROME);
                b.lathe(
                    &[(0.0, 7.0), (0.14, 7.1), (0.0, 7.25)],
                    6,
                    0.0,
                    CHROME,
                    CHROME,
                    false,
                );
            }
            Model::Flag => {
                // Hangs from its pole edge at the origin toward +X.
                let cloth = Map::Rect(0.0, 0.0, 1.0, 1.0);
                b.quad(
                    [
                        v(0.0, -0.5, 0.0),
                        v(1.6, -0.5, 0.0),
                        v(1.6, 0.5, 0.0),
                        v(0.0, 0.5, 0.0),
                    ],
                    cloth,
                    Vec3::Z,
                );
            }
            Model::GlassBox(kind) => {
                let (half, height) = match kind {
                    0 => (v(2.4, 0.0, 2.4), 3.2),
                    1 => (v(4.0, 0.0, 2.0), 2.4),
                    _ => (v(1.2, 0.0, 5.0), 1.6),
                };
                let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
                b.cuboid(
                    v(-half.x, 0.0, -half.z),
                    v(half.x, height, half.z),
                    glass,
                    glass,
                    false,
                );
            }
            Model::DoorCanopy => {
                b.block(v(-1.2, 1.4, 0.0), v(1.2, 1.55, 1.3), CHROME);
                b.lathe(&[(0.75, 0.0), (0.75, 0.05)], 12, 0.0, DARK, DARK, false);
            }
            Model::DoorWings => {
                // Four glass wings about a post, turning about Y.
                let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
                for wing in 0..4 {
                    let start = b.mark();
                    b.quad(
                        [
                            v(0.0, 0.0, 0.0),
                            v(0.7, 0.0, 0.0),
                            v(0.7, 1.35, 0.0),
                            v(0.0, 1.35, 0.0),
                        ],
                        glass,
                        Vec3::Z,
                    );
                    b.place(start, Mat4::from_rotation_y(FRAC_PI_2 * wing as f32));
                }
            }
            Model::Road => {
                b.cuboid(
                    v(-10.0, 0.0, -2.0),
                    v(10.0, 0.15, 2.0),
                    Map::patch(0.2, 0.3),
                    Map::Rect(0.0, 0.0, 2.5, 1.0),
                    false,
                );
            }
            Model::RoadCross => {
                b.cuboid(
                    v(-2.0, 0.0, -2.0),
                    v(2.0, 0.15, 2.0),
                    Map::patch(0.5, 0.5),
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    false,
                );
            }
            Model::Sidewalk => {
                // A raised ring 1.6 wide around the block.
                let pavers = Map::Tile(2.0, 2.0);
                let (outer, inner, height) = (10.0, 8.4, 0.3);
                for (min, max) in [
                    (v(-outer, 0.0, -outer), v(outer, height, -inner)),
                    (v(-outer, 0.0, inner), v(outer, height, outer)),
                    (v(-outer, 0.0, -inner), v(-inner, height, inner)),
                    (v(inner, 0.0, -inner), v(outer, height, inner)),
                ] {
                    b.cuboid(min, max, pavers, pavers, false);
                }
            }
            Model::Ground => {
                let tile = Map::Tile(4.0, 4.0);
                b.cuboid(v(-8.4, 0.0, -8.4), v(8.4, 0.2, 8.4), tile, tile, false);
            }
            Model::Lamp => {
                b.block(v(-0.2, 0.0, -0.2), v(0.2, 0.3, 0.2), DARK);
                b.bar(v(0.0, 0.3, 0.0), v(0.0, 3.6, 0.0), 0.07, 6, CHROME);
                b.bar(v(0.0, 3.5, 0.0), v(0.9, 3.7, 0.0), 0.05, 6, CHROME);
                b.block(v(0.6, 3.55, -0.2), v(1.2, 3.72, 0.2), CHROME);
                b.quad(
                    [
                        v(0.62, 3.549, 0.18),
                        v(1.18, 3.549, 0.18),
                        v(1.18, 3.549, -0.18),
                        v(0.62, 3.549, -0.18),
                    ],
                    WARM,
                    Vec3::NEG_Y,
                );
            }
            Model::TrafficLight => {
                b.bar(v(0.0, 0.0, 0.0), v(0.0, 3.4, 0.0), 0.08, 6, DARK);
                b.bar(v(0.0, 3.2, 0.0), v(2.6, 3.2, 0.0), 0.06, 6, DARK);
                b.block(v(2.3, 2.3, -0.18), v(2.7, 3.3, 0.18), DARK);
                for (index, lens) in [RED, AMBER, GREEN].into_iter().enumerate() {
                    let y = 3.1 - index as f32 * 0.3;
                    b.block(v(2.38, y - 0.1, 0.18), v(2.62, y + 0.1, 0.22), lens);
                    b.block(v(2.38, y - 0.1, -0.22), v(2.62, y + 0.1, -0.18), lens);
                }
            }
            Model::ShelterFrame => {
                for (x, z) in [(-1.8, -0.6), (1.8, -0.6), (-1.8, 0.6), (1.8, 0.6)] {
                    b.bar(v(x, 0.0, z), v(x, 2.2, z), 0.05, 6, CHROME);
                }
                b.block(v(-2.0, 2.2, -0.8), v(2.0, 2.32, 0.8), CHROME);
                b.block(v(-1.5, 0.35, -0.5), v(1.5, 0.45, -0.2), TIMBER);
                // An ad panel at one end, lit from inside.
                b.block(v(1.75, 0.3, -0.55), v(1.85, 2.1, 0.55), COOL);
            }
            Model::ShelterGlass => {
                let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
                b.quad(
                    [
                        v(1.8, 0.2, -0.6),
                        v(-1.8, 0.2, -0.6),
                        v(-1.8, 2.1, -0.6),
                        v(1.8, 2.1, -0.6),
                    ],
                    glass,
                    Vec3::NEG_Z,
                );
                b.quad(
                    [
                        v(-1.8, 0.2, -0.6),
                        v(-1.8, 0.2, 0.6),
                        v(-1.8, 2.1, 0.6),
                        v(-1.8, 2.1, -0.6),
                    ],
                    glass,
                    Vec3::NEG_X,
                );
            }
            Model::Bench => {
                for index in 0..3 {
                    let z = -0.2 + index as f32 * 0.14;
                    b.block(v(-0.9, 0.42, z), v(0.9, 0.48, z + 0.1), TIMBER);
                }
                b.block(v(-0.9, 0.55, 0.22), v(0.9, 0.9, 0.28), TIMBER);
                for x in [-0.75, 0.75] {
                    b.block(v(x - 0.04, 0.0, -0.22), v(x + 0.04, 0.42, 0.28), DARK);
                }
            }
            Model::Bin => {
                b.lathe(
                    &[(0.25, 0.0), (0.3, 0.8), (0.32, 0.85), (0.2, 0.9)],
                    8,
                    0.0,
                    DARK,
                    STEEL,
                    false,
                );
            }
            Model::Hydrant => {
                b.lathe(
                    &[
                        (0.18, 0.0),
                        (0.14, 0.1),
                        (0.14, 0.45),
                        (0.18, 0.5),
                        (0.1, 0.6),
                        (0.0, 0.66),
                    ],
                    8,
                    0.0,
                    PAINTED,
                    PAINTED,
                    false,
                );
                b.block(v(-0.25, 0.3, -0.05), v(0.25, 0.4, 0.05), PAINTED);
            }
            Model::Bollard => {
                b.lathe(
                    &[(0.14, 0.0), (0.14, 0.8), (0.0, 0.9)],
                    8,
                    0.0,
                    CHROME,
                    CHROME,
                    false,
                );
                b.lathe(&[(0.145, 0.62), (0.145, 0.7)], 8, 0.0, WARM, WARM, false);
            }
            Model::Planter => {
                b.cuboid(
                    v(-0.9, 0.0, -0.9),
                    v(0.9, 0.7, 0.9),
                    STONE,
                    Map::patch(0.62, 0.62),
                    false,
                );
            }
            Model::Flowers => {
                // Clumps of blossoms in vertex colors over a matte material.
                let colors = [
                    [0.95, 0.2, 0.35],
                    [1.0, 0.85, 0.2],
                    [0.7, 0.35, 0.95],
                    [1.0, 0.55, 0.2],
                    [0.95, 0.95, 0.95],
                ];
                b.paint([0.15, 0.35, 0.1]);
                b.block(
                    v(-0.8, 0.0, -0.8),
                    v(0.8, 0.12, 0.8),
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                );
                for index in 0..14 {
                    let random = crate::scene::hash(index * 977 + 13);
                    let (x, z) = (
                        (random % 150) as f32 / 100.0 - 0.75,
                        ((random >> 8) % 150) as f32 / 100.0 - 0.75,
                    );
                    b.paint(colors[(random >> 16) as usize % colors.len()]);
                    let size = 0.08 + ((random >> 20) % 5) as f32 * 0.015;
                    b.block(
                        v(x - size, 0.1, z - size),
                        v(x + size, 0.2 + size, z + size),
                        Map::Rect(0.0, 0.0, 1.0, 1.0),
                    );
                }
            }
            Model::KioskBody => {
                b.block(v(-1.0, 0.0, -0.8), v(1.0, 2.0, 0.8), DARK);
                b.block(v(-0.9, 0.9, 0.8), v(0.9, 1.7, 0.84), WARM);
                b.block(v(-1.05, 0.8, 0.8), v(1.05, 0.9, 1.1), STEEL);
            }
            Model::KioskRoof => {
                b.lathe(
                    &[(1.6, 0.0), (0.0, 0.8)],
                    4,
                    FRAC_PI_4,
                    Map::Rect(0.0, 0.0, 3.0, 1.0),
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    true,
                );
            }
            Model::BoothFrame => {
                b.block(v(-0.5, 0.0, -0.5), v(0.5, 0.1, 0.5), PAINTED);
                b.block(v(-0.5, 2.1, -0.5), v(0.5, 2.3, 0.5), PAINTED);
                for (x, z) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
                    b.block(
                        v(x - 0.05, 0.1, z - 0.05),
                        v(x + 0.05, 2.1, z + 0.05),
                        PAINTED,
                    );
                }
                b.block(v(-0.4, 2.12, 0.5), v(0.4, 2.28, 0.52), COOL);
            }
            Model::BoothGlass => {
                let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
                b.cuboid(
                    v(-0.45, 0.1, -0.45),
                    v(0.45, 2.1, 0.45),
                    glass,
                    glass,
                    false,
                );
            }
            Model::Umbrella => {
                b.bar(
                    v(0.0, 0.0, 0.0),
                    v(0.0, 2.1, 0.0),
                    0.03,
                    4,
                    Map::Rect(0.0, 0.9, 0.1, 1.0),
                );
                b.lathe(
                    &[(1.3, 1.8), (0.0, 2.3)],
                    8,
                    0.0,
                    Map::Rect(0.0, 0.0, 4.0, 1.0),
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    true,
                );
            }
            Model::CafeTable => {
                b.lathe(&[(0.35, 0.72), (0.35, 0.76)], 10, 0.0, STEEL, STEEL, true);
                b.bar(v(0.0, 0.0, 0.0), v(0.0, 0.72, 0.0), 0.04, 6, STEEL);
                for (x, z) in [(0.6, 0.0), (-0.6, 0.0)] {
                    b.block(v(x - 0.2, 0.42, z - 0.2), v(x + 0.2, 0.46, z + 0.2), STEEL);
                    b.bar(v(x, 0.0, z), v(x, 0.42, z), 0.03, 4, STEEL);
                }
            }
            Model::FoodCart => {
                b.block(v(-1.0, 0.35, -0.6), v(1.0, 1.3, 0.6), PAINT);
                b.block(v(-0.9, 1.3, -0.5), v(0.9, 1.35, 0.5), TRIM);
                for (x, z) in [(-0.6, 0.6), (0.6, 0.6), (-0.6, -0.6), (0.6, -0.6)] {
                    let start = b.mark();
                    b.lathe(&[(0.3, -0.06), (0.3, 0.06)], 10, 0.0, TIRE, TIRE, true);
                    b.place(
                        start,
                        Mat4::from_translation(v(x, 0.3, z)) * Mat4::from_rotation_x(FRAC_PI_2),
                    );
                }
            }
            Model::Plinth => {
                let stone = Map::Tile(2.0, 2.0);
                b.block(v(-1.2, 0.0, -1.2), v(1.2, 0.4, 1.2), stone);
                b.block(v(-0.9, 0.4, -0.9), v(0.9, 2.2, 0.9), stone);
            }
            Model::Figure => {
                // A blocky standing figure with a raised arm.
                let metal = Map::Tile(1.0, 1.0);
                b.block(v(-0.4, 0.0, -0.25), v(-0.05, 1.3, 0.25), metal);
                b.block(v(0.05, 0.0, -0.25), v(0.4, 1.3, 0.25), metal);
                b.block(v(-0.5, 1.3, -0.3), v(0.5, 2.6, 0.3), metal);
                b.block(v(-0.25, 2.6, -0.25), v(0.25, 3.1, 0.25), metal);
                b.block(v(-0.8, 1.6, -0.15), v(-0.5, 2.5, 0.15), metal);
                b.block(v(0.5, 2.3, -0.15), v(0.8, 3.6, 0.15), metal);
            }
            Model::Sculpture => {
                // A bean of mirror metal.
                let profile: Vec<(f32, f32)> = (0..=10)
                    .map(|index| {
                        let angle = PI * index as f32 / 10.0;
                        let (sin, cos) = math::sin_cos(angle);
                        (1.8 * sin * (1.0 + 0.25 * cos), 1.5 - 1.5 * cos)
                    })
                    .collect();
                let start = b.mark();
                b.lathe(
                    &profile,
                    16,
                    0.0,
                    Map::Tile(2.0, 2.0),
                    Map::Tile(2.0, 2.0),
                    false,
                );
                b.place(start, Mat4::from_rotation_z(0.25));
            }
            Model::Basin => {
                let stone = Map::Tile(2.0, 2.0);
                b.lathe(
                    &[
                        (3.0, 0.0),
                        (3.0, 0.6),
                        (2.7, 0.6),
                        (2.7, 0.3),
                        (0.6, 0.3),
                        (0.6, 1.4),
                        (1.2, 1.5),
                        (1.2, 1.7),
                        (0.3, 1.7),
                        (0.2, 2.3),
                        (0.0, 2.4),
                    ],
                    16,
                    0.0,
                    stone,
                    stone,
                    false,
                );
            }
            Model::Pool => {
                let water = Map::Tile(8.0, 8.0);
                b.lathe(&[(2.72, 0.0), (2.72, 0.05)], 16, 0.0, water, water, false);
            }
            Model::Hedge => {
                let leaves = Map::Tile(2.0, 2.0);
                b.cuboid(v(-1.6, 0.0, -0.4), v(1.6, 1.0, 0.4), leaves, leaves, false);
            }
            Model::Manhole => {
                b.lathe(&[(0.45, 0.0), (0.45, 0.03)], 10, 0.0, DARK, DARK, false);
            }
            Model::StreetSign => {
                b.bar(v(0.0, 0.0, 0.0), v(0.0, 2.8, 0.0), 0.05, 6, STEEL);
                b.block(v(-0.05, 2.4, -0.6), v(0.05, 2.7, 0.6), SIGN);
                b.block(v(-0.6, 2.05, -0.05), v(0.6, 2.35, 0.05), SIGN);
            }
            Model::Tree(kind) => {
                // A trunk and crossed leaf cards around a crown.
                let (height, crown, width) =
                    [(4.2, 2.2, 3.4), (5.0, 2.8, 3.0), (3.4, 1.8, 3.8)][kind as usize % 3];
                b.paint([0.55, 0.55, 0.55]);
                b.lathe(
                    &[(0.16, 0.0), (0.1, height - crown * 0.4)],
                    6,
                    0.0,
                    TRUNK,
                    TRUNK,
                    false,
                );
                for layer in 0..3 {
                    let t = layer as f32 / 2.0;
                    let shade = 0.65 + t * 0.35;
                    b.paint([shade, shade, shade]);
                    let size = width * (1.0 - t * 0.35);
                    b.cards(
                        v(0.0, height - crown - 0.2 + t * crown * 0.45, 0.0),
                        size,
                        crown * (1.0 - t * 0.2),
                        t * 0.9 + kind as f32,
                        CANOPY,
                    );
                }
            }
            Model::Poplar => {
                b.paint([0.55, 0.55, 0.55]);
                b.lathe(&[(0.12, 0.0), (0.08, 1.5)], 6, 0.0, TRUNK, TRUNK, false);
                for layer in 0..2 {
                    let shade = 0.7 + layer as f32 * 0.3;
                    b.paint([shade, shade, shade]);
                    b.cards(
                        v(0.0, 1.0 + layer as f32 * 0.4, 0.0),
                        1.6,
                        5.2,
                        layer as f32 * 0.8,
                        CANOPY,
                    );
                }
            }
            Model::Pine => {
                b.paint([0.5, 0.5, 0.5]);
                b.lathe(&[(0.15, 0.0), (0.1, 1.2)], 6, 0.0, TRUNK, TRUNK, false);
                for (index, (radius, low, high)) in
                    [(1.6, 1.0, 3.2), (1.25, 2.3, 4.3), (0.85, 3.4, 5.4)]
                        .into_iter()
                        .enumerate()
                {
                    let shade = 0.6 + index as f32 * 0.2;
                    b.paint([shade, shade, shade]);
                    b.lathe(
                        &[(radius, low), (0.0, high)],
                        8,
                        index as f32 * 0.4,
                        DENSE,
                        DENSE,
                        true,
                    );
                }
            }
            Model::Palm => {
                b.paint([0.7, 0.7, 0.7]);
                // A leaning trunk of segments, fronds drooping from the top.
                let mut top = Vec3::ZERO;
                for segment in 0..6 {
                    let next = top + v(0.12, 1.0, 0.0);
                    b.bar(top, next, 0.16 - segment as f32 * 0.012, 6, TRUNK);
                    top = next;
                }
                b.paint([1.0, 1.0, 1.0]);
                for frond in 0..7 {
                    let yaw = TAU * frond as f32 / 7.0;
                    let (sin, cos) = math::sin_cos(yaw);
                    let out = v(cos, 0.0, sin);
                    let side = v(-sin, 0.0, cos);
                    let tip = top + out * 2.4 - Vec3::Y * 0.9;
                    let mid = top + out * 1.2 + Vec3::Y * 0.2;
                    // The frond's base half toward the trunk, its tip half beyond.
                    let (base, end) = (
                        Map::Rect(0.0, 0.39, 0.5, 0.78),
                        Map::Rect(0.0, 0.0, 0.5, 0.39),
                    );
                    b.quad(
                        [
                            top - side * 0.35,
                            top + side * 0.35,
                            mid + side * 0.5,
                            mid - side * 0.5,
                        ],
                        base,
                        Vec3::Y,
                    );
                    b.quad(
                        [
                            mid - side * 0.5,
                            mid + side * 0.5,
                            tip + side * 0.15,
                            tip - side * 0.15,
                        ],
                        end,
                        Vec3::Y,
                    );
                }
            }
            Model::Bush => {
                for layer in 0..2 {
                    let shade = 0.7 + layer as f32 * 0.3;
                    b.paint([shade, shade, shade]);
                    b.cards(
                        v(0.0, layer as f32 * 0.3, 0.0),
                        1.8 - layer as f32 * 0.4,
                        1.2,
                        layer as f32,
                        CANOPY,
                    );
                }
            }
            Model::Rock => {
                b.lathe(
                    &[(0.0, 0.0), (1.0, 0.1), (0.8, 0.7), (0.0, 1.0)],
                    6,
                    0.3,
                    Map::Tile(2.0, 2.0),
                    Map::Tile(2.0, 2.0),
                    false,
                );
            }
            Model::Car(kind) => car(&mut b, kind),
            Model::Bus => {
                b.block(v(-2.7, 0.35, -0.65), v(2.7, 1.9, 0.65), PAINT);
                b.block(v(-2.5, 1.05, -0.66), v(2.4, 1.7, 0.66), WINDOW);
                b.block(v(2.6, 0.9, -0.55), v(2.71, 1.7, 0.55), WINDOW);
                lights(&mut b, 2.7, 0.6, 0.5);
                wheels(&mut b, &[-1.8, 1.8], 0.66, 0.35);
            }
            Model::DoubleDecker => {
                b.block(v(-2.5, 0.35, -0.65), v(2.5, 3.1, 0.65), PAINT);
                for y in [1.0, 2.2] {
                    b.block(v(-2.3, y, -0.66), v(2.3, y + 0.6, 0.66), WINDOW);
                }
                lights(&mut b, 2.5, 0.6, 0.5);
                wheels(&mut b, &[-1.6, 1.6], 0.66, 0.35);
            }
            Model::Truck => {
                b.block(v(-2.4, 0.4, -0.6), v(0.9, 2.2, 0.6), PAINT);
                b.block(v(0.95, 0.4, -0.58), v(2.1, 1.7, 0.58), TRIM);
                b.block(v(1.5, 1.1, -0.59), v(2.11, 1.6, 0.59), WINDOW);
                lights(&mut b, 2.1, 0.7, 0.45);
                wheels(&mut b, &[-1.6, -0.9, 1.4], 0.62, 0.38);
            }
            Model::Mixer => {
                b.block(v(-2.2, 0.4, -0.55), v(0.7, 0.8, 0.55), GRILLE);
                b.block(v(0.8, 0.4, -0.58), v(2.0, 1.7, 0.58), PAINT);
                b.block(v(1.4, 1.1, -0.59), v(2.01, 1.6, 0.59), WINDOW);
                lights(&mut b, 2.0, 0.7, 0.45);
                wheels(&mut b, &[-1.5, -0.8, 1.4], 0.6, 0.38);
            }
            Model::MixerDrum => {
                let start = b.mark();
                b.lathe(
                    &[(0.3, -1.4), (0.9, -0.6), (0.9, 0.5), (0.4, 1.3)],
                    10,
                    0.0,
                    PAINT,
                    TRIM,
                    true,
                );
                b.place(start, Mat4::from_rotation_z(FRAC_PI_2 - 0.2));
            }
            Model::TrainCar | Model::Locomotive => {
                let front = if self == Model::Locomotive { 2.2 } else { 2.5 };
                b.block(v(-2.5, 0.3, -0.85), v(front, 2.1, 0.85), TRIM);
                b.block(v(-2.3, 1.2, -0.86), v(front - 0.2, 1.8, 0.86), WINDOW);
                if self == Model::Locomotive {
                    // A raked nose: a low block and a slope up to the roof.
                    b.cuboid(v(2.2, 0.3, -0.85), v(2.8, 1.3, 0.85), TRIM, TRIM, true);
                    b.quad(
                        [
                            v(2.8, 1.3, 0.85),
                            v(2.8, 1.3, -0.85),
                            v(2.2, 2.1, -0.85),
                            v(2.2, 2.1, 0.85),
                        ],
                        WINDOW,
                        v(1.0, 0.8, 0.0),
                    );
                    for z in [-0.85, 0.85] {
                        b.poly(
                            &[v(2.2, 1.3, z), v(2.8, 1.3, z), v(2.2, 2.1, z)],
                            TRIM,
                            v(0.0, 0.0, z),
                        );
                    }
                    b.block(v(2.79, 0.6, -0.6), v(2.83, 0.8, -0.3), HEADLIGHT);
                    b.block(v(2.79, 0.6, 0.3), v(2.83, 0.8, 0.6), HEADLIGHT);
                }
                b.block(v(-2.3, 0.0, -0.7), v(front - 0.2, 0.3, 0.7), GRILLE);
            }
            Model::Speedboat => {
                hull(&mut b, 2.2, 0.8, 0.6);
                b.block(v(-0.4, 0.6, -0.5), v(0.4, 1.0, 0.5), WINDOW);
                b.block(v(-1.8, 0.6, -0.6), v(-0.6, 0.8, 0.6), INTERIOR);
            }
            Model::Ferry => {
                hull(&mut b, 6.0, 2.0, 1.0);
                b.block(v(-4.0, 1.0, -1.6), v(3.0, 2.4, 1.6), PAINT);
                b.block(v(-3.8, 1.5, -1.61), v(2.8, 2.1, 1.61), WINDOW);
                b.block(v(-2.0, 2.4, -1.2), v(1.0, 3.3, 1.2), PAINT);
                b.block(v(-1.8, 2.7, -1.21), v(0.8, 3.1, 1.21), WINDOW);
                b.lathe(
                    &[(0.35, 3.3), (0.35, 4.3)],
                    8,
                    0.0,
                    SIREN_RED,
                    SIREN_RED,
                    false,
                );
            }
            Model::Barge => {
                hull(&mut b, 7.0, 2.2, 0.9);
                for index in 0..3 {
                    let x = -5.0 + index as f32 * 3.2;
                    b.block(v(x, 0.9, -1.6), v(x + 2.8, 1.8, 1.6), GRILLE);
                }
                b.block(v(4.6, 0.9, -1.4), v(6.2, 2.6, 1.4), PAINT);
                b.block(v(4.7, 1.9, -1.41), v(6.1, 2.4, 1.41), WINDOW);
            }
            Model::Sailboat => {
                hull(&mut b, 2.6, 0.9, 0.5);
                b.bar(v(0.3, 0.5, 0.0), v(0.3, 5.5, 0.0), 0.05, 6, TRIM);
                b.block(v(-1.2, 0.5, -0.6), v(0.0, 0.9, 0.6), INTERIOR);
            }
            Model::Sail => {
                let cloth = Map::Rect(0.0, 0.0, 1.0, 1.0);
                b.poly(
                    &[v(0.0, 0.0, 0.0), v(-2.4, 0.0, 0.0), v(0.0, 4.6, 0.0)],
                    cloth,
                    Vec3::Z,
                );
            }
            Model::Drone => {
                b.block(v(-0.35, -0.12, -0.35), v(0.35, 0.12, 0.35), PAINT);
                for (x, z) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                    b.bar(v(0.0, 0.05, 0.0), v(x * 0.7, 0.08, z * 0.7), 0.04, 4, TRIM);
                    let start = b.mark();
                    b.lathe(&[(0.32, 0.0), (0.32, 0.04)], 10, 0.0, GRILLE, WINDOW, true);
                    b.place(start, Mat4::from_translation(v(x * 0.7, 0.1, z * 0.7)));
                }
                b.block(v(0.34, -0.05, -0.1), v(0.37, 0.05, 0.1), SIREN_BLUE);
            }
            Model::Heli => {
                b.block(v(-0.9, -0.6, -0.7), v(1.2, 0.6, 0.7), PAINT);
                b.block(v(0.6, -0.3, -0.71), v(1.21, 0.5, 0.71), WINDOW);
                b.block(v(-3.2, -0.15, -0.15), v(-0.9, 0.2, 0.15), PAINT);
                b.block(v(-3.2, 0.2, -0.05), v(-2.8, 1.1, 0.05), PAINT);
                for z in [-0.7, 0.7] {
                    b.block(v(-0.8, -1.0, z - 0.06), v(1.2, -0.92, z + 0.06), TRIM);
                    b.block(v(0.0, -0.92, z - 0.03), v(0.1, -0.6, z + 0.03), TRIM);
                }
                b.bar(v(0.0, 0.6, 0.0), v(0.0, 0.9, 0.0), 0.07, 6, TRIM);
            }
            Model::Rotor => {
                for blade in 0..4 {
                    let start = b.mark();
                    b.block(v(0.1, -0.03, -0.12), v(3.6, 0.03, 0.12), GRILLE);
                    b.place(start, Mat4::from_rotation_y(FRAC_PI_2 * blade as f32));
                }
                b.lathe(&[(0.2, -0.1), (0.2, 0.1)], 8, 0.0, TRIM, TRIM, true);
            }
            Model::Blimp => {
                let profile: Vec<(f32, f32)> = (0..=10)
                    .map(|index| {
                        let angle = PI * index as f32 / 10.0;
                        let (sin, cos) = math::sin_cos(angle);
                        (2.2 * sin, -7.0 * cos)
                    })
                    .collect();
                let start = b.mark();
                b.lathe(&profile, 12, 0.0, PAINT, PAINT, false);
                b.place(start, Mat4::from_rotation_z(-FRAC_PI_2));
                b.block(v(-1.2, -2.8, -0.5), v(1.4, -2.1, 0.5), WINDOW);
                for angle in [0.0, FRAC_PI_2, PI, -FRAC_PI_2] {
                    let start = b.mark();
                    b.block(v(-7.4, 0.0, -0.06), v(-5.0, 2.2, 0.06), PAINT);
                    b.place(start, Mat4::from_rotation_x(angle));
                }
            }
            Model::ExcavatorBody => {
                b.block(v(-1.6, 0.0, -1.2), v(1.6, 0.6, -0.6), TIRE);
                b.block(v(-1.6, 0.0, 0.6), v(1.6, 0.6, 1.2), TIRE);
                b.block(v(-1.3, 0.6, -1.0), v(1.3, 1.6, 1.0), PAINT);
                b.block(v(0.3, 1.6, 0.1), v(1.3, 2.6, 1.0), PAINT);
                b.block(v(0.35, 1.8, 0.09), v(1.25, 2.5, 1.01), WINDOW);
            }
            Model::ExcavatorBoom => {
                // From its pivot at the origin up and out along +X.
                b.block(v(0.0, -0.25, -0.2), v(3.6, 0.25, 0.2), PAINT);
                b.block(v(3.2, -0.25, -0.2), v(3.8, 1.6, 0.2), PAINT);
            }
            Model::ExcavatorStick => {
                b.block(v(-0.18, -2.6, -0.16), v(0.18, 0.0, 0.16), PAINT);
                b.block(v(-0.5, -3.1, -0.4), v(0.4, -2.5, 0.4), GRILLE);
            }
            Model::CraneMast => {
                lattice_column(&mut b, 1.1, 40.0);
            }
            Model::CraneJib => {
                // A lattice arm along +X, a counter-arm, a cab and weights.
                let lattice = |length: f32| Map::Rect(0.0, 0.0, length / 2.0, 1.0);
                let (low, high) = (-0.45, 0.45);
                for (y0, y1, z) in [(low, high, 0.45), (low, high, -0.45)] {
                    b.quad(
                        [
                            v(-7.0, y0, z),
                            v(22.0, y0, z),
                            v(22.0, y1, z),
                            v(-7.0, y1, z),
                        ],
                        lattice(29.0),
                        v(0.0, 0.0, z),
                    );
                }
                b.quad(
                    [
                        v(-7.0, high, 0.45),
                        v(22.0, high, 0.45),
                        v(22.0, high, -0.45),
                        v(-7.0, high, -0.45),
                    ],
                    lattice(29.0),
                    Vec3::Y,
                );
                b.block(
                    v(-7.0, -1.8, -0.9),
                    v(-4.5, -0.45, 0.9),
                    Map::Rect(0.02, 0.02, 0.1, 0.1),
                );
                b.block(
                    v(-1.0, -1.6, -0.8),
                    v(0.8, -0.45, 0.8),
                    Map::Rect(0.02, 0.02, 0.1, 0.1),
                );
                b.lathe(
                    &[(0.6, high), (0.0, 4.0)],
                    4,
                    FRAC_PI_4,
                    lattice(2.0),
                    lattice(1.0),
                    false,
                );
            }
            Model::FerrisRim => {
                // A wheel in the XY plane: two rims, spokes and a hub.
                let segments = 24;
                for z in [-0.6, 0.6] {
                    for segment in 0..segments {
                        let start = b.mark();
                        b.block(v(6.2, -0.8, z - 0.08), v(6.5, 0.8, z + 0.08), CHROME);
                        b.place(
                            start,
                            Mat4::from_rotation_z(TAU * segment as f32 / segments as f32),
                        );
                    }
                }
                for spoke in 0..12 {
                    let start = b.mark();
                    b.block(v(0.0, -0.05, -0.05), v(6.3, 0.05, 0.05), STEEL);
                    b.place(start, Mat4::from_rotation_z(TAU * spoke as f32 / 12.0));
                    // Lights along the rim.
                    let start = b.mark();
                    b.block(v(6.45, -0.12, 0.62), v(6.6, 0.12, 0.72), WARM);
                    b.place(
                        start,
                        Mat4::from_rotation_z(TAU * (spoke as f32 + 0.5) / 12.0),
                    );
                }
                let start = b.mark();
                b.lathe(&[(0.5, -0.8), (0.5, 0.8)], 10, 0.0, CHROME, CHROME, true);
                b.place(start, Mat4::from_rotation_x(FRAC_PI_2));
            }
            Model::FerrisCabin => {
                b.block(v(-0.6, -0.9, -0.5), v(0.6, 0.2, 0.5), PAINTED);
                b.block(v(-0.55, -0.4, -0.51), v(0.55, 0.1, 0.51), COOL);
                b.lathe(
                    &[(0.75, 0.2), (0.0, 0.6)],
                    4,
                    FRAC_PI_4,
                    CHROME,
                    CHROME,
                    true,
                );
            }
            Model::FerrisLeg => {
                b.block(v(-0.25, 0.0, -0.25), v(0.25, 8.0, 0.25), STEEL);
            }
            Model::CarouselBase => {
                b.lathe(&[(3.2, 0.0), (3.2, 0.4)], 16, 0.0, TIMBER, TIMBER, false);
                b.bar(v(0.0, 0.4, 0.0), v(0.0, 3.2, 0.0), 0.2, 8, CHROME);
            }
            Model::CarouselTop => {
                // Canopy, poles and horses, turning together.
                b.lathe(
                    &[(3.3, 2.8), (3.3, 3.1), (0.0, 4.2)],
                    16,
                    0.0,
                    Map::Rect(0.0, 0.0, 8.0, 1.0),
                    CHROME,
                    true,
                );
                for index in 0..8 {
                    let (sin, cos) = math::sin_cos(TAU * index as f32 / 8.0);
                    let at = v(cos * 2.4, 0.4, sin * 2.4);
                    b.bar(at, at + Vec3::Y * 2.4, 0.04, 4, CHROME);
                    let start = b.mark();
                    b.block(v(-0.45, 0.9, -0.15), v(0.45, 1.3, 0.15), PAINTED);
                    b.block(v(0.3, 1.2, -0.1), v(0.55, 1.65, 0.1), PAINTED);
                    b.place(
                        start,
                        Mat4::from_translation(at - Vec3::Y * 0.4)
                            * Mat4::from_rotation_y(-TAU * index as f32 / 8.0 + FRAC_PI_2),
                    );
                }
            }
            Model::TurbineTower => {
                b.lathe(&[(0.7, 0.0), (0.35, 24.0)], 10, 0.0, STEEL, STEEL, false);
                b.block(v(-0.6, 24.0, -0.5), v(1.8, 25.0, 0.5), STEEL);
            }
            Model::TurbineRotor => {
                // A nose along +X and three blades in the YZ plane.
                let start = b.mark();
                b.lathe(
                    &[(0.0, -0.6), (0.4, -0.3), (0.4, 0.3)],
                    8,
                    0.0,
                    STEEL,
                    STEEL,
                    false,
                );
                b.place(start, Mat4::from_rotation_z(FRAC_PI_2));
                for blade in 0..3 {
                    let start = b.mark();
                    b.block(v(0.05, 0.0, -0.25), v(0.18, 10.0, 0.25), STEEL);
                    b.place(start, Mat4::from_rotation_x(TAU * blade as f32 / 3.0));
                }
            }
            Model::Beacon => {
                b.bar(v(0.0, 0.0, 0.0), v(0.0, 1.6, 0.0), 0.08, 6, STEEL);
                b.block(v(-1.2, 1.5, -0.12), v(1.2, 1.7, 0.12), WARM);
                b.block(v(-0.12, 1.5, -1.2), v(0.12, 1.7, 1.2), WARM);
                b.lathe(
                    &[(0.0, 1.6), (0.35, 1.9), (0.0, 2.3)],
                    8,
                    0.0,
                    RED,
                    RED,
                    false,
                );
            }
            Model::RailDeck => {
                b.cuboid(
                    v(-12.0, 0.0, -1.6),
                    v(12.0, 0.9, 1.6),
                    Map::patch(0.2, 0.05),
                    Map::Rect(0.0, 0.0, 3.0, 1.0),
                    true,
                );
            }
            Model::RailPillar => {
                let concrete = Map::Tile(4.0, 4.0);
                b.cuboid(
                    v(-0.6, 0.0, -0.6),
                    v(0.6, 8.8, 0.6),
                    concrete,
                    concrete,
                    false,
                );
                b.block(v(-0.8, 8.8, -1.8), v(0.8, 9.5, 1.8), concrete);
            }
            Model::StationFrame => {
                for x in [-9.0, -3.0, 3.0, 9.0] {
                    for z in [-2.6, 2.6] {
                        b.bar(v(x, 0.0, z), v(x, 3.6, z * 0.9), 0.1, 6, CHROME);
                    }
                    b.bar(v(x, 3.6, -2.4), v(x, 3.6, 2.4), 0.08, 6, CHROME);
                }
                b.block(v(-10.0, 0.0, -2.8), v(10.0, 0.3, 2.8), STONE);
            }
            Model::StationRoof => {
                // A shallow glass vault along X.
                let steps = 6;
                let arc = |index: u32| {
                    let angle = PI * (0.15 + 0.7 * index as f32 / steps as f32);
                    let (sin, cos) = math::sin_cos(angle);
                    (2.9 * cos, 1.0 * sin)
                };
                for index in 0..steps {
                    let ((z0, y0), (z1, y1)) = (arc(index), arc(index + 1));
                    b.quad(
                        [
                            v(-10.0, y0, z0),
                            v(10.0, y0, z0),
                            v(10.0, y1, z1),
                            v(-10.0, y1, z1),
                        ],
                        Map::Rect(0.0, 0.0, 1.0, 1.0),
                        v(0.0, (y0 + y1) * 0.5, (z0 + z1) * 0.5),
                    );
                }
            }
            Model::BridgeDeck => {
                b.cuboid(
                    v(-10.0, -0.6, -2.25),
                    v(10.0, 0.15, 2.25),
                    Map::patch(0.2, 0.3),
                    Map::Rect(0.0, 0.0, 2.5, 1.0),
                    true,
                );
            }
            Model::BridgeArch => {
                // A chrome arch over the deck with hangers.
                let steps = 16;
                let point = |index: u32| {
                    let t = index as f32 / steps as f32;
                    let (sin, cos) = math::sin_cos(PI * t);
                    v(-10.0 * cos, 7.0 * sin, 0.0)
                };
                for side in [-2.3, 2.3] {
                    for index in 0..steps {
                        b.bar(
                            point(index) + Vec3::Z * side,
                            point(index + 1) + Vec3::Z * side,
                            0.22,
                            6,
                            Map::Tile(2.0, 2.0),
                        );
                    }
                    for index in 1..steps {
                        let top = point(index) + Vec3::Z * side;
                        b.bar(v(top.x, 0.0, side), top, 0.03, 4, Map::Tile(2.0, 2.0));
                    }
                }
                for index in [4, 8, 12] {
                    b.bar(
                        point(index) + Vec3::Z * -2.3,
                        point(index) + Vec3::Z * 2.3,
                        0.12,
                        6,
                        Map::Tile(2.0, 2.0),
                    );
                }
            }
            Model::BridgeLeaf => {
                // Hinged at the origin, reaching along +X.
                b.cuboid(
                    v(0.0, -0.5, -2.25),
                    v(10.0, 0.15, 2.25),
                    Map::patch(0.2, 0.3),
                    Map::Rect(0.0, 0.0, 1.25, 1.0),
                    true,
                );
            }
            Model::Dock => {
                let wood = Map::Tile(2.0, 2.0);
                b.block(v(-6.0, 0.0, -1.5), v(6.0, 0.25, 1.5), wood);
                for x in [-5.5, -2.0, 2.0, 5.5] {
                    for z in [-1.3, 1.3] {
                        b.bar(v(x, -1.6, z), v(x, 0.6, z), 0.15, 6, wood);
                    }
                }
            }
            Model::Stadium => {
                // An outer wall and seating stepping down to the pitch.
                let sides = 20;
                let seats = |rows: f32| Map::Rect(0.0, 0.0, 6.0, rows);
                for index in 0..sides {
                    let angle = |step: f32| TAU * (index as f32 + step) / sides as f32;
                    let at = |radius: f32, y: f32, step: f32| {
                        let (sin, cos) = math::sin_cos(angle(step));
                        v(radius * cos * 1.2, y, radius * sin)
                    };
                    let (sin, cos) = math::sin_cos(angle(0.5));
                    let radial = v(cos, 0.0, sin);
                    b.quad(
                        [
                            at(9.5, 0.0, 1.0),
                            at(9.5, 0.0, 0.0),
                            at(9.5, 7.0, 0.0),
                            at(9.5, 7.0, 1.0),
                        ],
                        Map::Rect(0.0, 0.96, 6.0, 1.0),
                        radial,
                    );
                    b.quad(
                        [
                            at(9.0, 7.0, 1.0),
                            at(9.0, 7.0, 0.0),
                            at(9.5, 7.0, 0.0),
                            at(9.5, 7.0, 1.0),
                        ],
                        Map::Rect(0.0, 0.96, 6.0, 1.0),
                        Vec3::Y,
                    );
                    b.quad(
                        [
                            at(6.0, 0.8, 0.0),
                            at(6.0, 0.8, 1.0),
                            at(9.0, 7.0, 1.0),
                            at(9.0, 7.0, 0.0),
                        ],
                        seats(3.0),
                        Vec3::Y - radial,
                    );
                }
            }
            Model::StadiumRoof => {
                let sides = 20;
                for index in 0..sides {
                    let angle = |step: f32| TAU * (index as f32 + step) / sides as f32;
                    let at = |radius: f32, y: f32, step: f32| {
                        let (sin, cos) = math::sin_cos(angle(step));
                        v(radius * cos * 1.2, y, radius * sin)
                    };
                    b.quad(
                        [
                            at(7.2, 0.6, 0.0),
                            at(7.2, 0.6, 1.0),
                            at(9.7, 0.0, 1.0),
                            at(9.7, 0.0, 0.0),
                        ],
                        Map::Tile(2.0, 2.0),
                        Vec3::Y,
                    );
                    b.quad(
                        [
                            at(7.2, 0.4, 1.0),
                            at(7.2, 0.4, 0.0),
                            at(9.7, -0.2, 0.0),
                            at(9.7, -0.2, 1.0),
                        ],
                        Map::Tile(2.0, 2.0),
                        Vec3::NEG_Y,
                    );
                }
            }
            Model::Pitch => {
                b.cuboid(
                    v(-7.2, 0.0, -4.8),
                    v(7.2, 0.1, 4.8),
                    Map::patch(0.1, 0.5),
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    false,
                );
            }
            Model::Floodlight => {
                b.bar(v(0.0, 0.0, 0.0), v(0.0, 12.0, 0.0), 0.15, 6, STEEL);
                b.block(v(-1.0, 11.6, -0.2), v(1.0, 12.8, 0.2), STEEL);
                b.block(v(-0.9, 11.7, -0.25), v(0.9, 12.7, -0.2), COOL);
            }
            Model::TvTower => {
                b.lathe(
                    &[
                        (4.0, 0.0),
                        (2.2, 6.0),
                        (1.4, 40.0),
                        (1.2, 80.0),
                        (1.0, 118.0),
                        (3.0, 122.0),
                        (3.0, 126.0),
                        (1.0, 128.0),
                        (0.7, 150.0),
                        (0.3, 175.0),
                        (0.0, 190.0),
                    ],
                    12,
                    0.0,
                    Map::Tile(2.0, 2.0),
                    Map::Tile(2.0, 2.0),
                    false,
                );
            }
            Model::ObsPod => {
                let profile: Vec<(f32, f32)> = (0..=8)
                    .map(|index| {
                        let angle = PI * index as f32 / 8.0;
                        let (sin, cos) = math::sin_cos(angle);
                        (5.5 * sin, -3.0 * cos)
                    })
                    .collect();
                b.lathe(
                    &profile,
                    16,
                    0.0,
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    Map::Rect(0.0, 0.0, 1.0, 1.0),
                    false,
                );
            }
            Model::TowerRing => {
                b.lathe(
                    &[(6.5, 0.0), (6.5, 1.2), (5.8, 1.6), (4.0, 1.6)],
                    20,
                    0.0,
                    Map::Tile(2.0, 2.0),
                    Map::Tile(2.0, 2.0),
                    true,
                );
            }
            Model::Person(variant) => return people::mesh(variant).map(|mesh| (mesh, Vec3::ZERO)),
        }
        let center = !matches!(self, Model::Person(_));
        b.finish(center)
    }
}

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// Building meshes carry second coordinates spanning them, for grime.
fn building_frame(b: &mut Builder) {
    b.uv1 = Uv1::Span;
}

fn tower(b: &mut Builder, style: Style, floors: u32) {
    building_frame(b);
    let height = FLOOR * floors as f32;
    let half = TOWER_WIDTH * 0.5;
    b.top = Some(height);
    // Floors of a fraction of the height, whole floors.
    let at = |fraction: f32| (floors as f32 * fraction).round() * FLOOR;
    match style {
        Style::Box => b.cuboid(
            v(-half, 0.0, -half),
            v(half, height, half),
            FACADE,
            WALL,
            false,
        ),
        Style::Setback => {
            for (from, to, width) in [(0.0, 0.55, 1.0), (0.55, 0.8, 0.75), (0.8, 1.0, 0.5)] {
                let reach = half * width;
                b.cuboid(
                    v(-reach, at(from), -reach),
                    v(reach, at(to), reach),
                    FACADE,
                    WALL,
                    false,
                );
            }
        }
        Style::Round => b.lathe(&[(half, 0.0), (half, height)], 14, 0.0, FACADE, WALL, false),
        Style::Hex => b.lathe(
            &[(half * 1.05, 0.0), (half * 1.05, height)],
            6,
            0.0,
            FACADE,
            WALL,
            false,
        ),
        Style::Tapered => b.lathe(
            &[(half * 1.35, 0.0), (half * 0.85, height)],
            4,
            FRAC_PI_4,
            FACADE,
            WALL,
            false,
        ),
        Style::Deco => b.lathe(
            &[
                (half * 1.1, 0.0),
                (half * 1.1, at(0.72)),
                (half * 0.85, at(0.72)),
                (half * 0.85, at(0.86)),
                (half * 0.6, at(0.86)),
                (half * 0.6, height),
            ],
            8,
            PI / 8.0,
            FACADE,
            WALL,
            false,
        ),
        Style::Twisted => {
            let slabs = floors / 4;
            for slab in 0..slabs {
                let (y0, y1) = (slab as f32 * 4.0 * FLOOR, (slab + 1) as f32 * 4.0 * FLOOR);
                let start = b.mark();
                let reach = half * 0.85;
                b.cuboid(
                    v(-reach, y0, -reach),
                    v(reach, y1, reach),
                    FACADE,
                    WALL,
                    false,
                );
                b.place(
                    start,
                    Mat4::from_rotation_y(
                        FRAC_PI_4 * 1.3 * slab as f32 / math::max(slabs as f32 - 1.0, 1.0),
                    ),
                );
            }
        }
        Style::Fins => {
            let reach = half * 0.9;
            b.cuboid(
                v(-reach, 0.0, -reach),
                v(reach, height, reach),
                FACADE,
                WALL,
                false,
            );
            // Vertical fins on every face, one per bay.
            let count = (TOWER_WIDTH / BAY).floor() as i32;
            for index in 0..=count {
                let along = -reach + index as f32 * (2.0 * reach / count as f32);
                for (min, max) in [
                    (
                        v(along - 0.08, 0.0, reach),
                        v(along + 0.08, height + 0.6, reach + 0.45),
                    ),
                    (
                        v(along - 0.08, 0.0, -reach - 0.45),
                        v(along + 0.08, height + 0.6, -reach),
                    ),
                    (
                        v(reach, 0.0, along - 0.08),
                        v(reach + 0.45, height + 0.6, along + 0.08),
                    ),
                    (
                        v(-reach - 0.45, 0.0, along - 0.08),
                        v(-reach, height + 0.6, along + 0.08),
                    ),
                ] {
                    b.cuboid(min, max, FACADE, WALL, false);
                }
            }
        }
        Style::Slant => {
            // One side rises 8 floors above the other: a sloped roof.
            let low = height - 8.0 * FLOOR;
            let corners = [
                v(-half, 0.0, half),
                v(half, 0.0, half),
                v(half, 0.0, -half),
                v(-half, 0.0, -half),
            ];
            let top_of = |corner: Vec3| if corner.x > 0.0 { height } else { low };
            for index in 0..4 {
                let (a, c) = (corners[index], corners[(index + 1) % 4]);
                let outward = (a + c).normalize_or_zero();
                b.poly(
                    &[a, c, v(c.x, top_of(c), c.z), v(a.x, top_of(a), a.z)],
                    FACADE,
                    outward,
                );
            }
            let roof: Vec<Vec3> = corners
                .iter()
                .rev()
                .map(|corner| v(corner.x, top_of(*corner), corner.z))
                .collect();
            b.poly(&roof, WALL, v(-(height - low), TOWER_WIDTH, 0.0));
        }
    }
}

fn midrise(b: &mut Builder, style: Mid, floors: u32) {
    building_frame(b);
    let height = FLOOR * floors as f32;
    let half = MID_WIDTH * 0.5;
    b.top = Some(height);
    match style {
        Mid::Block => b.cuboid(
            v(-half, 0.0, -half),
            v(half, height, half),
            FACADE,
            WALL,
            false,
        ),
        Mid::Ell => {
            b.cuboid(
                v(-half, 0.0, -half),
                v(half, height, 0.0),
                FACADE,
                WALL,
                false,
            );
            let wing = (floors as f32 * 0.6).round() * FLOOR;
            b.cuboid(v(-half, 0.0, 0.0), v(0.0, wing, half), FACADE, WALL, false);
        }
        Mid::Court => {
            b.cuboid(
                v(-half, 0.0, -half),
                v(-half * 0.4, height, half),
                FACADE,
                WALL,
                false,
            );
            b.cuboid(
                v(half * 0.4, 0.0, -half),
                v(half, height, half),
                FACADE,
                WALL,
                false,
            );
            let low = (floors as f32 * 0.7).round() * FLOOR;
            b.cuboid(
                v(-half * 0.4, 0.0, -half),
                v(half * 0.4, low, -half * 0.4),
                FACADE,
                WALL,
                false,
            );
        }
        Mid::Walkup => {
            b.cuboid(
                v(-half, 0.0, -half),
                v(half, height, half),
                FACADE,
                WALL,
                false,
            );
            // A cornice and a plinth.
            b.cuboid(
                v(-half - 0.2, height - 0.35, -half - 0.2),
                v(half + 0.2, height, half + 0.2),
                WALL,
                WALL,
                true,
            );
            b.cuboid(
                v(-half - 0.1, 0.0, -half - 0.1),
                v(half + 0.1, 0.6, half + 0.1),
                WALL,
                WALL,
                false,
            );
        }
        Mid::Terrace => {
            let low = (floors as f32 * 0.6).round() * FLOOR;
            b.cuboid(
                v(-half, 0.0, -half),
                v(half, low, half),
                FACADE,
                WALL,
                false,
            );
            b.cuboid(
                v(-half, low, -half),
                v(half * 0.3, height, half * 0.4),
                FACADE,
                WALL,
                false,
            );
        }
    }
}

/// Wheels at `xs` along each side `z` from the center, of `radius`.
fn wheels(b: &mut Builder, xs: &[f32], z: f32, radius: f32) {
    for &x in xs {
        for side in [-z, z] {
            let start = b.mark();
            b.lathe(&[(radius, -0.1), (radius, 0.1)], 8, 0.0, TIRE, TIRE, true);
            b.place(
                start,
                Mat4::from_translation(v(x, radius, side)) * Mat4::from_rotation_x(FRAC_PI_2),
            );
        }
    }
}

/// Headlights at the front `x` and taillights at the back, at `y`, `z` apart.
fn lights(b: &mut Builder, x: f32, y: f32, z: f32) {
    for side in [-z, z] {
        b.block(
            v(x - 0.02, y - 0.08, side - 0.12),
            v(x + 0.02, y + 0.08, side + 0.12),
            HEADLIGHT,
        );
        b.block(
            v(-x - 0.02, y - 0.08, side - 0.12),
            v(-x + 0.02, y + 0.08, side + 0.12),
            TAILLIGHT,
        );
    }
}

fn car(b: &mut Builder, kind: Car) {
    // Body length, width, body height, cabin extent and height.
    let (length, width, body, cabin, roof) = match kind {
        Car::Sedan | Car::Taxi | Car::Police => (2.1, 0.95, 0.55, (-0.55, 0.45), 0.95),
        Car::Hatch => (1.8, 0.9, 0.55, (-0.75, 0.35), 0.98),
        Car::Suv => (2.2, 1.0, 0.7, (-0.9, 0.5), 1.2),
        Car::Van => (2.4, 1.0, 0.75, (-1.15, 0.8), 1.3),
        Car::Pickup => (2.4, 1.0, 0.65, (0.0, 0.75), 1.15),
        Car::Sports => (2.1, 0.95, 0.45, (-0.35, 0.35), 0.78),
    };
    let (half, side) = (length * 0.5, width * 0.5);
    b.block(v(-half, 0.22, -side), v(half, body, side), PAINT);
    b.block(
        v(cabin.0, body, -side * 0.88),
        v(cabin.1, roof, side * 0.88),
        WINDOW,
    );
    b.block(
        v(cabin.0 + 0.08, roof - 0.04, -side * 0.84),
        v(cabin.1 - 0.06, roof, side * 0.84),
        PAINT,
    );
    b.block(
        v(half - 0.02, 0.25, -side * 0.6),
        v(half + 0.03, 0.4, side * 0.6),
        GRILLE,
    );
    lights(b, half + 0.01, body - 0.12, side * 0.65);
    wheels(b, &[-half * 0.65, half * 0.65], side * 0.95, 0.22);
    match kind {
        Car::Taxi => b.block(v(-0.15, roof, -0.12), v(0.15, roof + 0.12, 0.12), TAXI_SIGN),
        Car::Police => {
            b.block(v(-0.1, roof, -0.3), v(0.1, roof + 0.08, 0.0), SIREN_BLUE);
            b.block(v(-0.1, roof, 0.0), v(0.1, roof + 0.08, 0.3), SIREN_RED);
        }
        Car::Sports => b.block(
            v(-half, body, -side * 0.9),
            v(-half + 0.25, body + 0.12, side * 0.9),
            TRIM,
        ),
        Car::Pickup => b.block(v(-half, body, -side), v(-0.05, body + 0.2, side), TRIM),
        _ => {}
    }
}

/// A hull `length` long and `width` wide with a pointed bow toward +X, its
/// deck `height` up.
fn hull(b: &mut Builder, length: f32, width: f32, height: f32) {
    let (half, side) = (length * 0.5, width * 0.5);
    let bow = half * 0.6;
    b.cuboid(
        v(-half, 0.0, -side),
        v(bow, height, side),
        PAINT,
        TRIM,
        false,
    );
    let low = [v(bow, 0.0, -side), v(bow, 0.0, side), v(half, 0.0, 0.0)];
    let top = [
        v(bow, height, -side),
        v(bow, height, side),
        v(half, height, 0.0),
    ];
    b.poly(&top, TRIM, Vec3::Y);
    b.quad([low[1], low[2], top[2], top[1]], PAINT, v(1.0, 0.0, 1.0));
    b.quad([low[2], low[0], top[0], top[2]], PAINT, v(1.0, 0.0, -1.0));
    b.block(
        v(-half, height * 0.1, -side - 0.02),
        v(bow, height * 0.25, side + 0.02),
        SIREN_RED,
    );
}

/// A square lattice column `width` wide and `height` tall.
fn lattice_column(b: &mut Builder, width: f32, height: f32) {
    let half = width * 0.5;
    let map = Map::Rect(0.0, 0.0, 1.0, height / width);
    for (a, c, outward) in [
        (v(-half, 0.0, half), v(half, 0.0, half), Vec3::Z),
        (v(half, 0.0, -half), v(-half, 0.0, -half), Vec3::NEG_Z),
        (v(half, 0.0, half), v(half, 0.0, -half), Vec3::X),
        (v(-half, 0.0, -half), v(-half, 0.0, half), Vec3::NEG_X),
    ] {
        b.quad(
            [a, c, c + Vec3::Y * height, a + Vec3::Y * height],
            map,
            outward,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn every() -> Vec<Model> {
        let mut all = Vec::new();
        for style in [
            Style::Box,
            Style::Setback,
            Style::Round,
            Style::Twisted,
            Style::Tapered,
            Style::Hex,
            Style::Deco,
            Style::Fins,
            Style::Slant,
        ] {
            for class in 0..TOWER_FLOORS.len() as u8 {
                all.push(Model::Tower(style, class));
            }
        }
        for style in [Mid::Block, Mid::Ell, Mid::Court, Mid::Walkup, Mid::Terrace] {
            for class in 0..MID_FLOORS.len() as u8 {
                all.push(Model::Midrise(style, class));
            }
        }
        all.extend([1, 2, 3].map(Model::House));
        all.extend([0, 1, 2].map(Model::RoofSlab));
        all.extend([0, 1, 2].map(Model::Railing));
        all.extend([0, 1, 2].map(Model::GlassBox));
        all.extend([0, 1, 2].map(Model::Tree));
        all.extend(
            [
                self::Car::Sedan,
                self::Car::Hatch,
                self::Car::Suv,
                self::Car::Taxi,
                self::Car::Van,
                self::Car::Pickup,
                self::Car::Sports,
                self::Car::Police,
            ]
            .map(Model::Car),
        );
        all.extend((0..people::VARIANTS).map(Model::Person));
        use Model::*;
        all.extend([
            Townhouse,
            Shop,
            Warehouse,
            Shed,
            Gable,
            Hip,
            Spire,
            Crown,
            Dome,
            Helipad,
            Tank,
            Silo,
            Stack,
            Container,
            AcUnit,
            WaterTank,
            Antenna,
            DishPost,
            DishHead,
            Solar,
            Billboard,
            Trivision,
            NeonBlade,
            Awning,
            Balcony,
            BalconyRail,
            FireEscape,
            Gondola,
            Flagpole,
            Flag,
            DoorCanopy,
            DoorWings,
            Road,
            RoadCross,
            Sidewalk,
            Ground,
            Lamp,
            TrafficLight,
            ShelterFrame,
            ShelterGlass,
            Bench,
            Bin,
            Hydrant,
            Bollard,
            Planter,
            Flowers,
            KioskBody,
            KioskRoof,
            BoothFrame,
            BoothGlass,
            Umbrella,
            CafeTable,
            FoodCart,
            Plinth,
            Figure,
            Sculpture,
            Basin,
            Pool,
            Hedge,
            Manhole,
            StreetSign,
            Poplar,
            Pine,
            Palm,
            Bush,
            Rock,
            Bus,
            DoubleDecker,
            Truck,
            Mixer,
            MixerDrum,
            TrainCar,
            Locomotive,
            Speedboat,
            Ferry,
            Barge,
            Sailboat,
            Sail,
            Drone,
            Heli,
            Rotor,
            Blimp,
            ExcavatorBody,
            ExcavatorBoom,
            ExcavatorStick,
            CraneMast,
            CraneJib,
            FerrisRim,
            FerrisCabin,
            FerrisLeg,
            CarouselBase,
            CarouselTop,
            TurbineTower,
            TurbineRotor,
            Beacon,
            RailDeck,
            RailPillar,
            StationFrame,
            StationRoof,
            BridgeDeck,
            BridgeArch,
            BridgeLeaf,
            Dock,
            Stadium,
            StadiumRoof,
            Pitch,
            Floodlight,
            TvTower,
            ObsPod,
            TowerRing,
        ]);
        all
    }

    #[test]
    fn every_model_builds_a_valid_mesh() {
        let models = every();
        assert!(models.len() > 150, "{} models", models.len());
        for model in models {
            let mesh = model
                .build()
                .unwrap_or_else(|error| panic!("{model:?}: {error}"));
            assert!(!mesh.indices().is_empty(), "{model:?}");
            let bounds = mesh.bounds();
            if !matches!(model, Model::Person(_)) {
                let center = (bounds.min() + bounds.max()) * 0.5;
                assert!(center.length() < 1.0e-3, "{model:?} is centered: {center}");
            }
        }
    }

    #[test]
    fn buildings_span_their_floors_and_carry_grime_coordinates() {
        let mesh = Model::Tower(Style::Box, 0).build().expect("tower");
        let bounds = mesh.bounds();
        let height = bounds.max().y - bounds.min().y;
        assert!((height - TOWER_FLOORS[0] as f32 * FLOOR).abs() < 1.0e-3);
        assert!(mesh.uv1().is_some());
        assert!(Model::Road.build().expect("road").uv1().is_none());
        assert!(Model::Tree(0).build().expect("tree").colors().is_some());
    }

    #[test]
    fn fewer_meshes_resolve_to_eight_stand_ins() {
        let mut stand_ins: Vec<Model> = every().into_iter().map(|model| model.resolve(8)).collect();
        stand_ins.retain(|model| !matches!(model, Model::Person(_)));
        stand_ins.sort_by_key(|model| format!("{model:?}"));
        stand_ins.dedup();
        assert!(stand_ins.len() <= 8, "{stand_ins:?}");
        assert!(
            every()
                .into_iter()
                .all(|model| matches!(model.resolve(1), Model::Container | Model::Person(_)))
        );
    }
}
