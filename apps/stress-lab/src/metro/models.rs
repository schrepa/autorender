//! Low-poly models of the metro, built at their nominal world size and
//! centered on their bounds (skinned people stand on their origin). Each
//! model's texture coordinates suit one family of texture sets: façades
//! repeat per 8 bays and 8 floors, tiling ground and metal sets repeat in
//! world units, and atlases (street fixtures, vehicles) map faces into
//! uniform regions.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use engine::prelude::*;

use super::people;
use crate::kit::model_builder::{Builder, Map, Uv1};

mod buildings;
mod details;
mod landmarks;
mod nature;
mod streets;
mod vehicles;

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

// Foliage textures: leaves above, bark below a transparent gap. Cards stop
// short of the bark and trunks start inside it: filtering and coarser mip
// levels would otherwise draw an opaque bark line along every card's lower
// edge (and its shadow).
const CANOPY: Map = Map::Rect(0.0, 0.0, 1.0, 0.78);
const DENSE: Map = Map::Rect(0.3, 0.25, 0.7, 0.55);
const TRUNK: Map = Map::Rect(0.0, 0.86, 1.0, 1.0);

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
            Model::Tower(style, class) => {
                buildings::tower(&mut b, style, TOWER_FLOORS[class as usize])
            }
            Model::Midrise(style, class) => {
                buildings::midrise(&mut b, style, MID_FLOORS[class as usize])
            }
            Model::House(floors) => buildings::house(&mut b, floors),
            Model::Townhouse => buildings::townhouse(&mut b),
            Model::Shop => buildings::shop(&mut b),
            Model::Warehouse => buildings::warehouse(&mut b),
            Model::Shed => buildings::shed(&mut b),
            Model::RoofSlab(size) => buildings::roof_slab(&mut b, size),
            Model::Gable => buildings::gable(&mut b),
            Model::Hip => buildings::hip(&mut b),
            Model::Spire => buildings::spire(&mut b),
            Model::Crown => buildings::crown(&mut b),
            Model::Dome => buildings::dome(&mut b),
            Model::Helipad => buildings::helipad(&mut b),
            Model::Tank => buildings::tank(&mut b),
            Model::Silo => buildings::silo(&mut b),
            Model::Stack => buildings::stack(&mut b),
            Model::Container => buildings::container(&mut b),
            Model::AcUnit => details::ac_unit(&mut b),
            Model::WaterTank => details::water_tank(&mut b),
            Model::Antenna => details::antenna(&mut b),
            Model::DishPost => details::dish_post(&mut b),
            Model::DishHead => details::dish_head(&mut b),
            Model::Solar => details::solar(&mut b),
            Model::Railing(size) => details::railing(&mut b, size),
            Model::Billboard => details::billboard(&mut b),
            Model::Trivision => details::trivision(&mut b),
            Model::NeonBlade => details::neon_blade(&mut b),
            Model::Awning => details::awning(&mut b),
            Model::Balcony => details::balcony(&mut b),
            Model::BalconyRail => details::balcony_rail(&mut b),
            Model::FireEscape => details::fire_escape(&mut b),
            Model::Gondola => details::gondola(&mut b),
            Model::Flagpole => details::flagpole(&mut b),
            Model::Flag => details::flag(&mut b),
            Model::GlassBox(kind) => details::glass_box(&mut b, kind),
            Model::DoorCanopy => details::door_canopy(&mut b),
            Model::DoorWings => details::door_wings(&mut b),
            Model::Road => streets::road(&mut b),
            Model::RoadCross => streets::road_cross(&mut b),
            Model::Sidewalk => streets::sidewalk(&mut b),
            Model::Ground => streets::ground(&mut b),
            Model::Lamp => streets::lamp(&mut b),
            Model::TrafficLight => streets::traffic_light(&mut b),
            Model::ShelterFrame => streets::shelter_frame(&mut b),
            Model::ShelterGlass => streets::shelter_glass(&mut b),
            Model::Bench => streets::bench(&mut b),
            Model::Bin => streets::bin(&mut b),
            Model::Hydrant => streets::hydrant(&mut b),
            Model::Bollard => streets::bollard(&mut b),
            Model::Planter => streets::planter(&mut b),
            Model::Flowers => streets::flowers(&mut b),
            Model::KioskBody => streets::kiosk_body(&mut b),
            Model::KioskRoof => streets::kiosk_roof(&mut b),
            Model::BoothFrame => streets::booth_frame(&mut b),
            Model::BoothGlass => streets::booth_glass(&mut b),
            Model::Umbrella => streets::umbrella(&mut b),
            Model::CafeTable => streets::cafe_table(&mut b),
            Model::FoodCart => streets::food_cart(&mut b),
            Model::Plinth => streets::plinth(&mut b),
            Model::Figure => streets::figure(&mut b),
            Model::Sculpture => streets::sculpture(&mut b),
            Model::Basin => streets::basin(&mut b),
            Model::Pool => streets::pool(&mut b),
            Model::Hedge => streets::hedge(&mut b),
            Model::Manhole => streets::manhole(&mut b),
            Model::StreetSign => streets::street_sign(&mut b),
            Model::Tree(kind) => nature::tree(&mut b, kind),
            Model::Poplar => nature::poplar(&mut b),
            Model::Pine => nature::pine(&mut b),
            Model::Palm => nature::palm(&mut b),
            Model::Bush => nature::bush(&mut b),
            Model::Rock => nature::rock(&mut b),
            Model::Car(kind) => vehicles::car(&mut b, kind),
            Model::Bus => vehicles::bus(&mut b),
            Model::DoubleDecker => vehicles::double_decker(&mut b),
            Model::Truck => vehicles::truck(&mut b),
            Model::Mixer => vehicles::mixer(&mut b),
            Model::MixerDrum => vehicles::mixer_drum(&mut b),
            Model::TrainCar | Model::Locomotive => vehicles::train(&mut b, self),
            Model::Speedboat => vehicles::speedboat(&mut b),
            Model::Ferry => vehicles::ferry(&mut b),
            Model::Barge => vehicles::barge(&mut b),
            Model::Sailboat => vehicles::sailboat(&mut b),
            Model::Sail => vehicles::sail(&mut b),
            Model::Drone => vehicles::drone(&mut b),
            Model::Heli => vehicles::heli(&mut b),
            Model::Rotor => vehicles::rotor(&mut b),
            Model::Blimp => vehicles::blimp(&mut b),
            Model::ExcavatorBody => vehicles::excavator_body(&mut b),
            Model::ExcavatorBoom => vehicles::excavator_boom(&mut b),
            Model::ExcavatorStick => vehicles::excavator_stick(&mut b),
            Model::CraneMast => vehicles::crane_mast(&mut b),
            Model::CraneJib => vehicles::crane_jib(&mut b),
            Model::FerrisRim => vehicles::ferris_rim(&mut b),
            Model::FerrisCabin => vehicles::ferris_cabin(&mut b),
            Model::FerrisLeg => vehicles::ferris_leg(&mut b),
            Model::CarouselBase => vehicles::carousel_base(&mut b),
            Model::CarouselTop => vehicles::carousel_top(&mut b),
            Model::TurbineTower => vehicles::turbine_tower(&mut b),
            Model::TurbineRotor => vehicles::turbine_rotor(&mut b),
            Model::Beacon => vehicles::beacon(&mut b),
            Model::RailDeck => landmarks::rail_deck(&mut b),
            Model::RailPillar => landmarks::rail_pillar(&mut b),
            Model::StationFrame => landmarks::station_frame(&mut b),
            Model::StationRoof => landmarks::station_roof(&mut b),
            Model::BridgeDeck => landmarks::bridge_deck(&mut b),
            Model::BridgeArch => landmarks::bridge_arch(&mut b),
            Model::BridgeLeaf => landmarks::bridge_leaf(&mut b),
            Model::Dock => landmarks::dock(&mut b),
            Model::Stadium => landmarks::stadium(&mut b),
            Model::StadiumRoof => landmarks::stadium_roof(&mut b),
            Model::Pitch => landmarks::pitch(&mut b),
            Model::Floodlight => landmarks::floodlight(&mut b),
            Model::TvTower => landmarks::tv_tower(&mut b),
            Model::ObsPod => landmarks::obs_pod(&mut b),
            Model::TowerRing => landmarks::tower_ring(&mut b),
            Model::Person(variant) => return people::mesh(variant).map(|mesh| (mesh, Vec3::ZERO)),
        }
        let center = !matches!(self, Model::Person(_));
        b.finish(center)
    }
}

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}
