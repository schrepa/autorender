use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GunKind {
    Sidearm,
    Smg,
    Rifle,
    Sniper,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeaponNumbers {
    pub sidearm: GunNumbers,
    pub smg: GunNumbers,
    pub rifle: GunNumbers,
    pub sniper: GunNumbers,
    pub melee: MeleeNumbers,
}

impl WeaponNumbers {
    pub fn gun(&self, kind: GunKind) -> &GunNumbers {
        match kind {
            GunKind::Sidearm => &self.sidearm,
            GunKind::Smg => &self.smg,
            GunKind::Rifle => &self.rifle,
            GunKind::Sniper => &self.sniper,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GunNumbers {
    pub name: String,
    pub cost: u32,
    pub auto: bool,
    pub shots_per_second: f32,
    pub magazine: u32,
    pub reserve: u32,
    pub reload_seconds: f32,
    pub equip_seconds: f32,
    pub inspect_seconds: f32,
    pub run_speed_factor: f32,
    pub damage: Vec<DamageBand>,
    pub spread_deg: SpreadDeg,
    pub spread_bloom: SpreadBloom,
    pub scope: Option<ScopeNumbers>,
    pub recoil: RecoilNumbers,
    pub tag_slow: f32,
}

impl GunNumbers {
    pub fn range_m(&self) -> f32 {
        self.damage.last().map_or(0.0, |band| band.to_m)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DamageBand {
    pub to_m: f32,
    pub head: f32,
    pub body: f32,
    pub leg: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpreadDeg {
    pub stand: f32,
    pub crouch: f32,
    pub walk: f32,
    pub run: f32,
    pub air: f32,
    pub crouch_move: f32,
    pub land: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpreadBloom {
    pub per_shot_deg: f32,
    pub max_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeNumbers {
    pub zoom: Vec<f32>,
    pub aim_speed_factor: f32,
    pub spread_deg: SpreadDeg,
    pub fire_rate_factor: f32,
    pub recoil_factor: f32,
    pub scope_in_seconds: f32,
    pub breaks_on_shot: bool,
}

impl RecoilNumbers {
    #[must_use]
    pub fn scaled(&self, factor: f32) -> RecoilNumbers {
        RecoilNumbers {
            up_deg_per_shot: self.up_deg_per_shot * factor,
            max_up_deg: self.max_up_deg * factor,
            side_deg: self.side_deg * factor,
            side_jitter_deg: self.side_jitter_deg * factor,
            ..*self
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoilNumbers {
    pub up_deg_per_shot: f32,
    pub max_up_deg: f32,
    pub side_deg: f32,
    pub side_starts_after_shots: u32,
    pub side_swing_shots: u32,
    pub side_jitter_deg: f32,
    pub recover_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeleeNumbers {
    pub name: String,
    pub damage_front: f32,
    pub damage_back: f32,
    pub range_m: f32,
    pub run_speed_factor: f32,
    pub swing_seconds: f32,
    pub equip_seconds: f32,
    pub heavy_front: f32,
    pub heavy_back: f32,
    pub heavy_swing_seconds: f32,
    pub combo_swings: u8,
    pub combo_reset_seconds: f32,
    pub strike_seconds: StrikeSeconds,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrikeSeconds {
    pub light: Vec<f32>,
    pub heavy: f32,
}

impl StrikeSeconds {
    pub fn of(&self, heavy: bool, step: u8) -> f32 {
        if heavy {
            return self.heavy;
        }
        let count = self.light.len().max(1);
        self.light
            .get(usize::from(step) % count)
            .copied()
            .unwrap_or(0.0)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HitShapeNumbers {
    pub torso_radius_m: f32,
    pub leg_radius_m: f32,
    pub head_lift_m: f32,
    pub head_radius_m: BTreeMap<String, f32>,
    pub stand_joints_m: StandJoints,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StandJoints {
    pub head: f32,
    pub spine: f32,
    pub hip: f32,
    pub feet_apart: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CombatNumbers {
    pub friendly_fire: bool,
    pub assist_seconds: f32,
    pub seed: u64,
    pub loadout_primary: Option<GunKind>,
    pub deadzone_share: f32,
    pub walk_share: f32,
    pub land_seconds: f32,
    pub tag_seconds: f32,
    pub pierce_tag_slow: f32,
    pub surfaces: BTreeMap<String, SurfaceNumbers>,
    pub pickup_reach_m: f32,
    pub pickup_aim_m: f32,
    pub device_hand: DeviceHand,
    pub key_drop: KeyDrop,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyDrop {
    pub throw_m: f32,
    pub lift_m: f32,
    pub radius_m: f32,
    pub floor_drop_m: f32,
    pub back_steps: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceHand {
    pub equip_seconds: f32,
    pub run_speed_factor: f32,
    pub busy_run_speed_factor: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceNumbers {
    pub pierce_m: ClassMeters,
    pub damage_kept: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassMeters {
    pub sidearm: f32,
    pub smg: f32,
    pub rifle: f32,
    pub sniper: f32,
}

impl ClassMeters {
    pub fn of(&self, kind: GunKind) -> f32 {
        match kind {
            GunKind::Sidearm => self.sidearm,
            GunKind::Smg => self.smg,
            GunKind::Rifle => self.rifle,
            GunKind::Sniper => self.sniper,
        }
    }
}

pub const DEFAULT_SURFACE: &str = "stone";

impl CombatNumbers {
    pub fn surface(&self, kind: &str) -> Option<&SurfaceNumbers> {
        self.surfaces
            .get(kind)
            .or_else(|| self.surfaces.get(DEFAULT_SURFACE))
    }

    pub fn surface_by(&self, same: impl Fn(&str) -> bool) -> Option<&SurfaceNumbers> {
        self.surfaces
            .iter()
            .find(|(name, _)| same(name))
            .map(|(_, table)| table)
            .or_else(|| self.surfaces.get(DEFAULT_SURFACE))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Absorb {
    pub armor_absorb: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ArmsNumbers {
    pub weapons: WeaponNumbers,
    pub hit_shapes: HitShapeNumbers,
    pub combat: CombatNumbers,
    pub economy: Absorb,
    pub player: PlayerShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct PlayerShape {
    pub eye_height_m: f32,
    pub run_speed_mps: f32,
}

pub fn ticks(seconds: f32, ticks_per_second: u32) -> u32 {
    (seconds * ticks_per_second as f32).round() as u32
}
