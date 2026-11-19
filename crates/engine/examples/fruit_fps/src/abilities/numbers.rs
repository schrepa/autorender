use std::collections::BTreeMap;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::shapes::cos_degrees;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Slot {
    C,
    Q,
    E,
    X,
}

impl Slot {
    pub const ALL: [Slot; 4] = [Slot::C, Slot::Q, Slot::E, Slot::X];

    pub fn index(self) -> usize {
        match self {
            Slot::C => 0,
            Slot::Q => 1,
            Slot::E => 2,
            Slot::X => 3,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Slot::C => "c",
            Slot::Q => "q",
            Slot::E => "e",
            Slot::X => "x",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbilityId {
    pub character: usize,
    pub slot: Slot,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Modifiers {
    pub speed: f32,
    pub fire_rate: f32,
    pub reload: f32,
    pub damage_taken: f32,
    pub spread: f32,
    pub sight_m: Option<f32>,
}

impl Modifiers {
    pub const NONE: Modifiers = Modifiers {
        speed: 1.0,
        fire_rate: 1.0,
        reload: 1.0,
        damage_taken: 1.0,
        spread: 1.0,
        sight_m: None,
    };

    #[must_use]
    pub fn combine(self, other: Modifiers) -> Modifiers {
        Modifiers {
            speed: self.speed * other.speed,
            fire_rate: self.fire_rate * other.fire_rate,
            reload: self.reload * other.reload,
            damage_taken: self.damage_taken * other.damage_taken,
            spread: self.spread * other.spread,
            sight_m: match (self.sight_m, other.sight_m) {
                (Some(a), Some(b)) => Some(if b < a { b } else { a }),
                (a, None) => a,
                (None, b) => b,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Onto {
    Ground,
    Surface,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Throw {
    pub speed: f32,
    pub gravity: f32,
    pub radius_m: f32,
    pub bounces: u32,
    pub bounce_keep: f32,
    pub fuse_ticks: Option<u32>,
    pub max_travel_m: Option<f32>,
    pub body_bounce: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub range_m: f32,
    pub onto: Onto,
    pub lift_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Delivery {
    Caster,
    Throw(Throw),
    Place(Place),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlashSpec {
    pub range_m: f32,
    pub max_blind_ticks: u32,
    pub away_factor: f32,
    pub fade_ticks: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Falloff {
    pub radius_m: f32,
    pub max: f32,
    pub min: f32,
    pub hurts_team: bool,
    pub line_of_sight: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmokeSpec {
    pub radius_m: f32,
    pub ticks: u32,
    pub grow_ticks: u32,
    pub fade_ticks: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneSpec {
    pub radius_m: f32,
    pub ticks: u32,
    pub damage_per_tick: Option<f32>,
    pub hurts_team: bool,
    pub slow: Option<f32>,
    pub heal_per_tick: Option<f32>,
    pub sight_m: Option<f32>,
    pub reveal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lay {
    AlongAim,
    AcrossAim,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WallSpec {
    pub length_m: f32,
    pub height_m: f32,
    pub thickness_m: f32,
    pub ticks: u32,
    pub health: Option<f32>,
    pub solid: bool,
    pub lay: Lay,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrapShape {
    Radius(f32),
    Wire { max_length_m: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrapSpec {
    pub shape: TrapShape,
    pub arm_ticks: u32,
    pub life_ticks: Option<u32>,
    pub health: f32,
    pub hit_radius_m: f32,
    pub slow: f32,
    pub slow_ticks: u32,
    pub reveal_ticks: Option<u32>,
    pub chain: Option<Chain>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chain {
    pub windup_ticks: u32,
    pub leash_m: f32,
    pub slow: f32,
    pub zap_damage: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensorSpec {
    pub radius_m: f32,
    pub reveal_ticks: u32,
    pub arm_ticks: u32,
    pub life_ticks: Option<u32>,
    pub health: f32,
    pub hit_radius_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Flash(FlashSpec),
    FlashCone {
        windup_ticks: u32,
        cos_half_angle: f32,
        flash: FlashSpec,
    },
    Explosion(Falloff),
    Strike {
        warn_ticks: u32,
        falloff: Falloff,
    },
    Smoke(SmokeSpec),
    Zone(ZoneSpec),
    Wall(WallSpec),
    Trap(TrapSpec),
    Sensor(SensorSpec),
    Pulse {
        radius_m: f32,
        reveal_ticks: u32,
    },
    TeamHeal {
        radius_m: f32,
        heal_per_tick: f32,
        ticks: u32,
        overshield: Option<f32>,
    },
    Dash {
        distance_m: f32,
        ticks: u32,
    },
    Transform {
        ticks: u32,
        modifiers: Modifiers,
        overshield: Option<f32>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbilitySpec {
    pub id: AbilityId,
    pub name: String,
    pub block: String,
    pub delivery: Delivery,
    pub action: Action,
    pub prop: Option<String>,
    pub solid_kind: Option<String>,
    pub drop: Option<DropSpec>,
    pub recover_ticks: u32,
    pub object_damage: f32,
    pub enemy_visibility: f32,
    pub ally_damage: f32,
    pub affects_team: bool,
    pub heal_pause_ticks: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropSpec {
    pub height_m: f32,
    pub ticks: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThrowSurface {
    pub kind: String,
    pub bounce: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CharacterAbilities {
    pub key: String,
    pub name: String,
    pub abilities: [AbilitySpec; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbilityBook {
    pub characters: Vec<CharacterAbilities>,
    pub throw_surfaces: Vec<ThrowSurface>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbilityError {
    pub character: String,
    pub slot: Option<Slot>,
    pub message: String,
}

impl AbilityBook {
    pub fn parse(numbers_json: &str, ticks_per_second: u32) -> Result<AbilityBook, AbilityError> {
        let root: Root = serde_json::from_str(numbers_json).map_err(|error| AbilityError {
            character: String::new(),
            slot: None,
            message: error.to_string(),
        })?;
        let rate = Rate(ticks_per_second);
        let mut characters = Vec::with_capacity(root.characters.len());
        for (character, (key, raw)) in root.characters.into_iter().enumerate() {
            let recover = root.ability_handling.recover_seconds;
            let read = |slot: Slot, value: &Value| {
                spec(AbilityId { character, slot }, value, (rate, recover)).map_err(|message| {
                    AbilityError {
                        character: key.clone(),
                        slot: Some(slot),
                        message,
                    }
                })
            };
            let mut abilities = [
                read(Slot::C, &raw.c)?,
                read(Slot::Q, &raw.q)?,
                read(Slot::E, &raw.e)?,
                read(Slot::X, &raw.x)?,
            ];
            for ability in &mut abilities {
                if let Delivery::Throw(throw) = &mut ability.delivery {
                    throw.body_bounce = root.thrown_into_bodies.bounce;
                }
            }
            characters.push(CharacterAbilities {
                key,
                name: raw.name,
                abilities,
            });
        }
        let throw_surfaces = root
            .throw_surfaces
            .into_iter()
            .map(|(kind, raw)| ThrowSurface {
                kind,
                bounce: raw.bounce,
            })
            .collect();
        Ok(AbilityBook {
            characters,
            throw_surfaces,
        })
    }

    pub fn ability(&self, id: AbilityId) -> &AbilitySpec {
        &self.characters[id.character].abilities[id.slot.index()]
    }

    pub fn character(&self, key: &str) -> Option<usize> {
        self.characters.iter().position(|entry| entry.key == key)
    }
}

#[derive(Deserialize)]
struct Root {
    characters: BTreeMap<String, RawCharacter>,
    #[serde(default)]
    throw_surfaces: BTreeMap<String, RawThrowSurface>,
    thrown_into_bodies: RawThrowSurface,
    ability_handling: RawHandling,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHandling {
    recover_seconds: RawRecover,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecover {
    throw: f32,
    place: f32,
    caster: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawThrowSurface {
    bounce: f32,
}

#[derive(Deserialize)]
struct RawCharacter {
    name: String,
    c: Value,
    q: Value,
    e: Value,
    x: Value,
}

#[derive(Clone, Copy)]
struct Rate(u32);

impl Rate {
    fn ticks(self, seconds: f32) -> u32 {
        (seconds * self.0 as f32).round() as u32
    }

    fn per_tick(self, per_second: f32) -> f32 {
        per_second / self.0 as f32
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawDelivery {
    Caster,
    Throw(RawThrow),
    Place(RawPlace),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawThrow {
    speed_mps: f32,
    gravity_mps2: f32,
    radius_m: f32,
    bounces: u32,
    bounce_keep: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPlace {
    range_m: f32,
    onto: Onto,
    lift_m: f32,
}

impl RawDelivery {
    fn resolve(
        &self,
        rate: Rate,
        fuse_seconds: Option<f32>,
        max_travel_m: Option<f32>,
    ) -> Delivery {
        match self {
            RawDelivery::Caster => Delivery::Caster,
            RawDelivery::Throw(raw) => Delivery::Throw(Throw {
                speed: rate.per_tick(raw.speed_mps),
                gravity: rate.per_tick(rate.per_tick(raw.gravity_mps2)),
                radius_m: raw.radius_m,
                bounces: raw.bounces,
                bounce_keep: raw.bounce_keep,
                fuse_ticks: fuse_seconds.map(|seconds| rate.ticks(seconds)),
                max_travel_m,
                body_bounce: 0.0,
            }),
            RawDelivery::Place(raw) => Delivery::Place(Place {
                range_m: raw.range_m,
                onto: raw.onto,
                lift_m: raw.lift_m,
            }),
        }
    }
}

#[derive(Deserialize)]
struct Head {
    name: String,
    block: String,
    prop: Option<String>,
    kind: Option<String>,
    drop_height_m: Option<f32>,
    drop_seconds: Option<f32>,
    recover_seconds: Option<f32>,
    object_damage: Option<f32>,
    enemy_visibility: Option<f32>,
    ally_damage: Option<f32>,
    affects_team: Option<bool>,
    heal_pause_seconds: Option<f32>,
}

#[derive(Deserialize)]
struct Life {
    lasts_seconds: Option<f32>,
    lasts_whole_round: Option<bool>,
}

impl Life {
    fn ticks(&self, rate: Rate) -> Result<Option<u32>, String> {
        match (self.lasts_seconds, self.lasts_whole_round) {
            (Some(seconds), None) => Ok(Some(rate.ticks(seconds))),
            (None, Some(true)) => Ok(None),
            _ => {
                Err("needs exactly one of `lasts_seconds` or `lasts_whole_round: true`".to_string())
            }
        }
    }
}

#[derive(Deserialize)]
struct TrapEffect {
    stun_seconds: f32,
    slow: f32,
}

#[derive(Deserialize)]
struct TrapFields {
    delivery: RawDelivery,
    arm_seconds: f32,
    trigger_radius_m: f32,
    health: f32,
    hit_radius_m: f32,
    effect: TrapEffect,
}

#[derive(Deserialize)]
struct WireFields {
    delivery: RawDelivery,
    arm_seconds: f32,
    max_length_m: f32,
    slow: f32,
    slow_seconds: f32,
    reveal_seconds: f32,
    health: f32,
    hit_radius_m: f32,
    windup_seconds: f32,
    leash_m: f32,
    chained_slow: f32,
    zap_damage: f32,
}

#[derive(Deserialize)]
struct SensorFields {
    delivery: RawDelivery,
    arm_seconds: f32,
    radius_m: f32,
    reveal_seconds: f32,
    health: f32,
    hit_radius_m: f32,
}

#[derive(Deserialize)]
struct FlashFields {
    delivery: RawDelivery,
    fuse_seconds: f32,
    max_blind_seconds: f32,
    range_m: f32,
    away_factor: f32,
    fade_seconds: f32,
}

#[derive(Deserialize)]
struct FlashConeFields {
    windup_seconds: f32,
    cone_deg: f32,
    range_m: f32,
    max_blind_seconds: f32,
    away_factor: f32,
    fade_seconds: f32,
}

#[derive(Deserialize)]
struct GrenadeFields {
    delivery: RawDelivery,
    fuse_seconds: f32,
    radius_m: f32,
    max_damage: f32,
    min_damage: f32,
    hurts_team: bool,
}

#[derive(Deserialize)]
struct StrikeFields {
    delivery: RawDelivery,
    delay_seconds: f32,
    radius_m: f32,
    max_damage: f32,
    min_damage: f32,
    hurts_team: bool,
}

#[derive(Deserialize)]
struct SmokeFields {
    delivery: RawDelivery,
    radius_m: f32,
    seconds: f32,
    range_m: f32,
    grow_seconds: f32,
    fade_seconds: f32,
}

#[derive(Deserialize)]
struct ZoneFields {
    delivery: RawDelivery,
    radius_m: f32,
    seconds: f32,
}

#[derive(Deserialize)]
struct Slow {
    slow: f32,
}

#[derive(Deserialize)]
struct DamageRate {
    damage_per_second: f32,
    hurts_team: bool,
}

#[derive(Deserialize)]
struct HealRate {
    heal_per_second: f32,
}

#[derive(Deserialize)]
struct Sight {
    sight_m: f32,
}

#[derive(Deserialize)]
struct CurtainFields {
    length_m: f32,
    height_m: f32,
    thickness_m: f32,
    seconds: f32,
}

#[derive(Deserialize)]
struct SolidWallFields {
    delivery: RawDelivery,
    width_m: f32,
    height_m: f32,
    thickness_m: f32,
    health: f32,
    seconds: f32,
}

#[derive(Deserialize)]
struct PulseFields {
    radius_m: f32,
    reveal_seconds: f32,
}

#[derive(Deserialize)]
struct TeamHealFields {
    radius_m: f32,
    heal_seconds: f32,
    heal_per_second: f32,
    overshield: Option<f32>,
}

#[derive(Deserialize)]
struct DashFields {
    distance_m: f32,
    seconds: f32,
}

#[derive(Deserialize)]
struct SelfBuffFields {
    seconds: f32,
    speed_factor: Option<f32>,
    fire_rate_factor: Option<f32>,
    reload_factor: Option<f32>,
    damage_taken_factor: Option<f32>,
    spread_factor: Option<f32>,
    overshield: Option<f32>,
}

fn fields<T: DeserializeOwned>(value: &Value) -> Result<T, String> {
    T::deserialize(value).map_err(|error| error.to_string())
}

fn zone(
    value: &Value,
    rate: Rate,
    fill: impl FnOnce(&mut ZoneSpec) -> Result<(), String>,
) -> Built {
    let base: ZoneFields = fields(value)?;
    let mut spec = ZoneSpec {
        radius_m: base.radius_m,
        ticks: rate.ticks(base.seconds),
        damage_per_tick: None,
        hurts_team: false,
        slow: None,
        heal_per_tick: None,
        sight_m: None,
        reveal: false,
    };
    fill(&mut spec)?;
    Ok((base.delivery.resolve(rate, None, None), Action::Zone(spec)))
}

fn flash_spec(
    range_m: f32,
    max_blind_seconds: f32,
    away_factor: f32,
    fade_seconds: f32,
    rate: Rate,
) -> FlashSpec {
    FlashSpec {
        range_m,
        max_blind_ticks: rate.ticks(max_blind_seconds),
        away_factor,
        fade_ticks: rate.ticks(fade_seconds),
    }
}

type Built = Result<(Delivery, Action), String>;

fn unknown(block: &str) -> Built {
    Err(format!("unknown block `{block}`"))
}

fn spec(
    id: AbilityId,
    value: &Value,
    (rate, recover): (Rate, RawRecover),
) -> Result<AbilitySpec, String> {
    let head: Head = fields(value)?;
    let block = head.block.as_str();
    if block == "wall_solid" && head.kind.is_none() {
        return Err(
            "a solid wall needs `kind` (the surface it adds to the collision world)".to_owned(),
        );
    }
    let (delivery, action) = match block {
        "trap" | "trap_tripwire" | "trap_reveal" => traps(block, value, rate),
        "flash" | "flash_cone" | "grenade" | "airstrike" => bursts(block, value, rate),
        "smoke" | "zone_slow" | "zone_slow_reveal" | "zone_damage" | "zone_heal"
        | "zone_nearsight" => clouds(block, value, rate),
        "wall_vision" | "wall_solid" => walls(block, value, rate),
        _ => personal(block, value, rate),
    }?;
    let drop = head.drop_height_m.map(|height_m| DropSpec {
        height_m,
        ticks: head.drop_seconds.map_or(0, |seconds| rate.ticks(seconds)),
    });
    Ok(AbilitySpec {
        id,
        name: head.name,
        block: head.block,
        delivery,
        action,
        prop: head.prop,
        solid_kind: head.kind,
        drop,
        recover_ticks: rate.ticks(head.recover_seconds.unwrap_or(match delivery {
            Delivery::Caster => recover.caster,
            Delivery::Throw(_) => recover.throw,
            Delivery::Place(_) => recover.place,
        })),
        object_damage: head.object_damage.unwrap_or(1.0),
        enemy_visibility: head.enemy_visibility.unwrap_or(1.0),
        ally_damage: head.ally_damage.unwrap_or(1.0),
        affects_team: head.affects_team == Some(true),
        heal_pause_ticks: head.heal_pause_seconds.map(|seconds| rate.ticks(seconds)),
    })
}

fn traps(block: &str, value: &Value, rate: Rate) -> Built {
    let life = fields::<Life>(value)?.ticks(rate)?;
    match block {
        "trap" => {
            let f: TrapFields = fields(value)?;
            Ok((
                f.delivery.resolve(rate, None, None),
                Action::Trap(TrapSpec {
                    shape: TrapShape::Radius(f.trigger_radius_m),
                    arm_ticks: rate.ticks(f.arm_seconds),
                    life_ticks: life,
                    health: f.health,
                    hit_radius_m: f.hit_radius_m,
                    slow: f.effect.slow,
                    slow_ticks: rate.ticks(f.effect.stun_seconds),
                    reveal_ticks: None,
                    chain: None,
                }),
            ))
        }
        "trap_tripwire" => {
            let f: WireFields = fields(value)?;
            Ok((
                f.delivery.resolve(rate, None, None),
                Action::Trap(TrapSpec {
                    shape: TrapShape::Wire {
                        max_length_m: f.max_length_m,
                    },
                    arm_ticks: rate.ticks(f.arm_seconds),
                    life_ticks: life,
                    health: f.health,
                    hit_radius_m: f.hit_radius_m,
                    slow: f.slow,
                    slow_ticks: rate.ticks(f.slow_seconds),
                    reveal_ticks: Some(rate.ticks(f.reveal_seconds)),
                    chain: Some(Chain {
                        windup_ticks: rate.ticks(f.windup_seconds),
                        leash_m: f.leash_m,
                        slow: f.chained_slow,
                        zap_damage: f.zap_damage,
                    }),
                }),
            ))
        }
        "trap_reveal" => {
            let f: SensorFields = fields(value)?;
            Ok((
                f.delivery.resolve(rate, None, None),
                Action::Sensor(SensorSpec {
                    radius_m: f.radius_m,
                    reveal_ticks: rate.ticks(f.reveal_seconds),
                    arm_ticks: rate.ticks(f.arm_seconds),
                    life_ticks: life,
                    health: f.health,
                    hit_radius_m: f.hit_radius_m,
                }),
            ))
        }
        other => unknown(other),
    }
}

fn bursts(block: &str, value: &Value, rate: Rate) -> Built {
    match block {
        "flash" => {
            let f: FlashFields = fields(value)?;
            let flash = flash_spec(
                f.range_m,
                f.max_blind_seconds,
                f.away_factor,
                f.fade_seconds,
                rate,
            );
            Ok((
                f.delivery.resolve(rate, Some(f.fuse_seconds), None),
                Action::Flash(flash),
            ))
        }
        "flash_cone" => {
            let f: FlashConeFields = fields(value)?;
            let flash = flash_spec(
                f.range_m,
                f.max_blind_seconds,
                f.away_factor,
                f.fade_seconds,
                rate,
            );
            Ok((
                Delivery::Caster,
                Action::FlashCone {
                    windup_ticks: rate.ticks(f.windup_seconds),
                    cos_half_angle: cos_degrees(f.cone_deg * 0.5),
                    flash,
                },
            ))
        }
        "grenade" => {
            let f: GrenadeFields = fields(value)?;
            Ok((
                f.delivery.resolve(rate, Some(f.fuse_seconds), None),
                Action::Explosion(Falloff {
                    radius_m: f.radius_m,
                    max: f.max_damage,
                    min: f.min_damage,
                    hurts_team: f.hurts_team,
                    line_of_sight: true,
                }),
            ))
        }
        "airstrike" => {
            let f: StrikeFields = fields(value)?;
            Ok((
                f.delivery.resolve(rate, None, None),
                Action::Strike {
                    warn_ticks: rate.ticks(f.delay_seconds),
                    falloff: Falloff {
                        radius_m: f.radius_m,
                        max: f.max_damage,
                        min: f.min_damage,
                        hurts_team: f.hurts_team,
                        line_of_sight: false,
                    },
                },
            ))
        }
        other => unknown(other),
    }
}

fn clouds(block: &str, value: &Value, rate: Rate) -> Built {
    match block {
        "smoke" => {
            let f: SmokeFields = fields(value)?;
            Ok((
                f.delivery.resolve(rate, None, Some(f.range_m)),
                Action::Smoke(SmokeSpec {
                    radius_m: f.radius_m,
                    ticks: rate.ticks(f.seconds),
                    grow_ticks: rate.ticks(f.grow_seconds),
                    fade_ticks: rate.ticks(f.fade_seconds),
                }),
            ))
        }
        "zone_slow" => zone(value, rate, |spec| {
            spec.slow = Some(fields::<Slow>(value)?.slow);
            Ok(())
        }),
        "zone_slow_reveal" => zone(value, rate, |spec| {
            spec.slow = Some(fields::<Slow>(value)?.slow);
            spec.reveal = true;
            Ok(())
        }),
        "zone_damage" => zone(value, rate, |spec| {
            let f: DamageRate = fields(value)?;
            spec.damage_per_tick = Some(rate.per_tick(f.damage_per_second));
            spec.hurts_team = f.hurts_team;
            Ok(())
        }),
        "zone_heal" => zone(value, rate, |spec| {
            spec.heal_per_tick = Some(rate.per_tick(fields::<HealRate>(value)?.heal_per_second));
            Ok(())
        }),
        "zone_nearsight" => zone(value, rate, |spec| {
            spec.sight_m = Some(fields::<Sight>(value)?.sight_m);
            Ok(())
        }),
        other => unknown(other),
    }
}

fn walls(block: &str, value: &Value, rate: Rate) -> Built {
    match block {
        "wall_vision" => {
            let f: CurtainFields = fields(value)?;
            Ok((
                Delivery::Caster,
                Action::Wall(WallSpec {
                    length_m: f.length_m,
                    height_m: f.height_m,
                    thickness_m: f.thickness_m,
                    ticks: rate.ticks(f.seconds),
                    health: None,
                    solid: false,
                    lay: Lay::AlongAim,
                }),
            ))
        }
        "wall_solid" => {
            let f: SolidWallFields = fields(value)?;
            Ok((
                f.delivery.resolve(rate, None, None),
                Action::Wall(WallSpec {
                    length_m: f.width_m,
                    height_m: f.height_m,
                    thickness_m: f.thickness_m,
                    ticks: rate.ticks(f.seconds),
                    health: Some(f.health),
                    solid: true,
                    lay: Lay::AcrossAim,
                }),
            ))
        }
        other => unknown(other),
    }
}

fn personal(block: &str, value: &Value, rate: Rate) -> Built {
    match block {
        "reveal_pulse" => {
            let f: PulseFields = fields(value)?;
            Ok((
                Delivery::Caster,
                Action::Pulse {
                    radius_m: f.radius_m,
                    reveal_ticks: rate.ticks(f.reveal_seconds),
                },
            ))
        }
        "team_heal" => {
            let f: TeamHealFields = fields(value)?;
            Ok((
                Delivery::Caster,
                Action::TeamHeal {
                    radius_m: f.radius_m,
                    heal_per_tick: rate.per_tick(f.heal_per_second),
                    ticks: rate.ticks(f.heal_seconds),
                    overshield: f.overshield,
                },
            ))
        }
        "dash" => {
            let f: DashFields = fields(value)?;
            Ok((
                Delivery::Caster,
                Action::Dash {
                    distance_m: f.distance_m,
                    ticks: rate.ticks(f.seconds),
                },
            ))
        }
        "self_buff" => {
            let f: SelfBuffFields = fields(value)?;
            let unchanged = Modifiers::NONE;
            Ok((
                Delivery::Caster,
                Action::Transform {
                    ticks: rate.ticks(f.seconds),
                    modifiers: Modifiers {
                        speed: f.speed_factor.unwrap_or(unchanged.speed),
                        fire_rate: f.fire_rate_factor.unwrap_or(unchanged.fire_rate),
                        reload: f.reload_factor.unwrap_or(unchanged.reload),
                        damage_taken: f.damage_taken_factor.unwrap_or(unchanged.damage_taken),
                        spread: f.spread_factor.unwrap_or(unchanged.spread),
                        sight_m: None,
                    },
                    overshield: f.overshield,
                },
            ))
        }
        other => unknown(other),
    }
}
