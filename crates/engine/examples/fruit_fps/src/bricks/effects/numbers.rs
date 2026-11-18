use std::collections::BTreeMap;

use serde::Deserialize;

use crate::solids::Kind;

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Pools {
    pub tracers: u32,
    pub flashes: u32,
    pub sprays: u32,
    pub decals: u32,
    pub rings: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TracerLook {
    pub seconds: f32,
    pub width_m: f32,
    pub core: [f32; 3],
    pub halo: f32,
    pub muzzle_offset_m: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FlashLook {
    pub seconds: f32,
    pub size_m: f32,
    pub color: [f32; 3],
    pub points: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Spray {
    pub sprites: u32,
    pub seconds: f32,
    pub speed_mps: f32,
    pub spread: f32,
    pub size_m: f32,
    pub gravity_mps2: f32,
    pub color: [f32; 3],
    pub opacity: f32,
    pub streak_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Impacts {
    pub default: Spray,
    pub kinds: BTreeMap<String, Spray>,
}

impl Impacts {
    pub fn of(&self, kind: Option<Kind>) -> &Spray {
        kind.and_then(|kind| {
            self.kinds
                .iter()
                .find(|(name, _)| Kind::named(name) == kind)
                .map(|(_, spray)| spray)
        })
        .unwrap_or(&self.default)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Juice {
    pub spray: Spray,
    pub death_scale: f32,
    pub colors: BTreeMap<String, [f32; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DecalLook {
    pub seconds: f32,
    pub fade_seconds: f32,
    pub size_m: f32,
    pub depth_m: f32,
    pub color: [f32; 3],
    pub opacity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SmokeLook {
    pub color: [f32; 3],
    pub rim_m: f32,
    pub tint: [f32; 3],
    pub tint_share: f32,
    pub under: f32,
    pub noise_scale: f32,
    pub noise_speed: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PanelLook {
    pub rind: RindLook,
    pub curtain: CurtainLook,
    pub trim_m: f32,
    pub edge_glow: f32,
    pub edge_power: f32,
    pub rim_glow: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RindLook {
    pub skin: [f32; 3],
    pub stripe: [f32; 3],
    pub pith: [f32; 3],
    pub flesh: [f32; 3],
    pub seed: [f32; 3],
    pub stripe_m: f32,
    pub skin_m: f32,
    pub pith_m: f32,
    pub seed_m: f32,
    pub slab_m: f32,
    pub bevel_m: f32,
    pub top_m: f32,
    pub bury_m: f32,
    pub round: f32,
    pub yaw_degrees: f32,
    pub lean_degrees: f32,
    pub rise_seconds: f32,
    pub stagger_seconds: f32,
    pub sink_seconds: f32,
    pub contact: f32,
    pub contact_m: f32,
    pub contact_height_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CurtainLook {
    pub peel: [f32; 3],
    pub pulp: [f32; 3],
    pub foam: [f32; 3],
    pub glow: f32,
    pub streak_m: f32,
    pub fleck_m: f32,
    pub fall_mps: f32,
    pub splash_m: f32,
    pub hop_m: f32,
    pub near_m: f32,
    pub end_m: f32,
    pub top_m: f32,
    pub fall_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ZoneLook {
    pub damage: [f32; 3],
    pub slow: [f32; 3],
    pub heal: [f32; 3],
    pub sight: [f32; 3],
    pub reveal: [f32; 3],
    pub fill: f32,
    pub rim_m: f32,
    pub pattern_m: f32,
    pub pattern_speed: f32,
    pub height_m: f32,
    pub dashes: f32,
    pub dash_speed: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrapLook {
    pub size_m: f32,
    pub wire_width_m: f32,
    pub projectile_size_m: f32,
    pub idle_glow: f32,
    pub armed_glow: f32,
    pub pulse_hz: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RingLook {
    pub color: [f32; 3],
    pub rim_m: f32,
    pub fill: f32,
    pub height_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BlastLook {
    pub flash_scale: f32,
    pub spray: Spray,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ScreenLook {
    pub flash: [f32; 3],
    pub damage: [f32; 3],
    pub damage_full: f32,
    pub damage_seconds: f32,
    pub damage_edge: f32,
    pub sight: [f32; 3],
    pub sight_fade_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DeviceLook {
    pub lamp_m: f32,
    pub light: [f32; 3],
    pub light_size_m: f32,
    pub blink_slow_seconds: f32,
    pub blink_fast_seconds: f32,
    pub peak_glow: f32,
    pub dim_glow: f32,
    pub pulse_power: f32,
    pub halo_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BarrierLook {
    pub friendly: [f32; 3],
    pub enemy: [f32; 3],
    pub glow: f32,
    pub fill: f32,
    pub line: f32,
    pub ground: f32,
    pub ground_m: f32,
    pub fade_from: f32,
    pub side_m: f32,
    pub cell_m: f32,
    pub line_px: f32,
    pub drift_mps: f32,
    pub band_m: f32,
    pub band_mps: f32,
    pub band_share: f32,
    pub eye_m: f32,
    pub eye_least: f32,
    pub extend_m: f32,
    pub near_m: f32,
    pub near_boost: f32,
    pub drop_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PropsLook {
    pub trail_seconds: f32,
    pub trail_width_m: f32,
    pub trail_glow: f32,
    pub puff_scale: f32,
    pub wire_color: [f32; 3],
    pub wire_glow: f32,
    pub wire_least_per_m: f32,
    pub wire_pulse_mps: f32,
    pub wire_pulse_m: f32,
    pub shadow_radius_m: f32,
    pub shadow_opacity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OrbLook {
    pub color: [f32; 3],
    pub lift_m: f32,
    pub core_m: f32,
    pub halo_m: f32,
    pub glow: f32,
    pub halo_glow: f32,
    pub dim: f32,
    pub pulse_hz: f32,
    pub channel_boost: f32,
    pub beam_width_m: f32,
    pub beam_glow: f32,
    pub beam_least: f32,
    pub fade_seconds: f32,
    pub burst: Spray,
    pub burst_flash: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ShatterLook {
    pub pieces: Spray,
    pub sparks: Spray,
    pub reference_m: f32,
    pub least_scale: f32,
    pub most_scale: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ShowcaseLook {
    pub every_seconds: f32,
    pub distance_m: f32,
    pub radius_m: f32,
    pub wall_m: [f32; 3],
    pub wire_share: f32,
    pub model_m: f32,
    pub model_prefix: String,
    pub trap_models: TrapModels,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrapModels {
    pub trap: String,
    pub sensor: String,
    pub wire: String,
    pub projectile: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EffectsNumbers {
    pub pools: Pools,
    pub tracer: TracerLook,
    pub flash: FlashLook,
    pub impacts: Impacts,
    pub juice: Juice,
    pub decal: DecalLook,
    pub smoke: SmokeLook,
    pub panels: PanelLook,
    pub teams: [[f32; 3]; 2],
    pub zones: ZoneLook,
    pub traps: TrapLook,
    pub strike: RingLook,
    pub pulse: RingLook,
    pub blast: BlastLook,
    pub screen: ScreenLook,
    pub device: DeviceLook,
    pub props: PropsLook,
    pub barrier: BarrierLook,
    pub orbs: OrbLook,
    pub shatter: ShatterLook,
    pub showcase: ShowcaseLook,
    #[serde(default)]
    pub enemy: Option<EnemyLook>,
    #[serde(default)]
    pub preview: Option<PreviewLook>,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreviewLook {
    pub valid: [f32; 3],
    pub invalid: [f32; 3],
    pub width_m: f32,
    pub rim_m: f32,
    pub height_m: f32,
    pub every_points: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EnemyLook {
    pub rim: [f32; 3],
    pub rim_strength: f32,
    pub reveal: [f32; 3],
}

impl EffectsNumbers {
    pub fn most_sprites(&self) -> u32 {
        [
            self.impacts.default.sprites,
            self.juice.spray.sprites,
            self.blast.spray.sprites,
            self.orbs.burst.sprites,
            self.shatter.pieces.sprites,
            self.shatter.sparks.sprites,
        ]
        .into_iter()
        .chain(self.impacts.kinds.values().map(|spray| spray.sprites))
        .fold(1, u32::max)
    }
}

#[derive(Deserialize)]
struct MatchPart {
    device_fuse_seconds: f32,
}

#[derive(Deserialize)]
struct FilePart {
    effects: EffectsNumbers,
    #[serde(rename = "match")]
    rules: MatchPart,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Parsed {
    pub look: EffectsNumbers,
    pub fuse_seconds: f32,
}

pub(super) fn parse(text: &str) -> Result<Parsed, String> {
    let part: FilePart = serde_json::from_str(text).map_err(|error| error.to_string())?;
    Ok(Parsed {
        look: part.effects,
        fuse_seconds: part.rules.device_fuse_seconds,
    })
}
