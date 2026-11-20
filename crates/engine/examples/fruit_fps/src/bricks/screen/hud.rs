use engine::prelude::Resource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::rules::{BuyRequest, Phase};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Shown {
    #[default]
    Hud,
    Buy,
    Scoreboard,
    RoundEnd,
    MatchEnd,
    Title,
    #[serde(rename = "character_select")]
    Select,
    Loading,
    Pause,
    Settings,
}

impl Shown {
    pub const KEYS: [&'static str; 10] = [
        "hud",
        "buy",
        "scoreboard",
        "round_end",
        "match_end",
        "title",
        "character_select",
        "loading",
        "pause",
        "settings",
    ];

    pub fn key(self) -> &'static str {
        Self::KEYS[self as usize]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tone {
    #[default]
    Text,
    Attack,
    Defend,
    Ready,
    Danger,
}

impl Tone {
    pub fn key(self) -> &'static str {
        match self {
            Tone::Text => "text",
            Tone::Attack => "attack",
            Tone::Defend => "defend",
            Tone::Ready => "ready",
            Tone::Danger => "danger",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoundFacts {
    pub time: f32,
    pub number: u32,
    pub planted: bool,
    pub fuse: f32,
    pub site: String,
    pub time_low: bool,
    pub fuse_low: bool,
    pub overtime: bool,
    pub decider: bool,
    pub swapped: bool,
    pub last_of_half: bool,
    pub device: String,
    pub buying: bool,
    pub ready: bool,
    pub ready_left: f32,
    pub ready_count: u32,
    pub go: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamFacts {
    pub color: Tone,
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlateFacts {
    pub icon: String,
    pub alive: bool,
    pub health: f32,
    pub ult_ready: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeedLine {
    pub killer: String,
    pub killer_color: Tone,
    pub weapon: String,
    pub headshot: bool,
    pub victim: String,
    pub victim_color: Tone,
    pub mine: bool,
    pub ability: String,
    pub world: bool,
    pub killer_char: String,
    pub victim_char: String,
    pub wall: bool,
    pub assist_char: String,
    pub assist_color: Tone,
    pub assist_more: u32,
    pub assisted: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlayerFacts {
    pub health: f32,
    pub health_low: bool,
    pub armor: f32,
    pub armor_icon: String,
    pub credits: u32,
    pub char_name: String,
    pub role: String,
    pub icon: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AbilityFacts {
    pub icon: String,
    pub max: u32,
    pub charges: u32,
    pub empty: bool,
    pub cooling: bool,
    pub cooldown: f32,
    pub cooldown_frac: f32,
    pub points: u32,
    pub points_needed: u32,
    pub ready: bool,
    pub equipped: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WeaponFacts {
    pub name: String,
    pub icon: String,
    pub mag: Option<u32>,
    pub mag_size: Option<u32>,
    pub reserve: Option<u32>,
    pub low: bool,
    pub reload: Option<f32>,
    pub down: bool,
    pub down_left: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ActionFacts {
    pub active: bool,
    pub label: String,
    pub left: f32,
    pub progress: f32,
    pub color: Tone,
    pub kind: String,
    pub checkpoint: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Card {
    pub kind: String,
    pub name: String,
    pub icon: String,
    pub price: String,
    pub owned: bool,
    pub afford: bool,
    pub focus: bool,
    pub need: u32,
    pub closed: bool,
    pub ask: bool,
    pub asked: bool,
    pub sell_credits: u32,
    pub max: u32,
    pub charges: u32,
    pub points: u32,
    pub stats: WeaponStats,
    pub buy: Option<BuyRequest>,
    pub sell: Option<BuyRequest>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WeaponStats {
    pub damage: f32,
    pub rate: f32,
    pub range: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mate {
    pub name: String,
    pub icon: String,
    pub credits: u32,
    pub can_rifle: bool,
    pub asked: bool,
    pub ask_icon: String,
    pub ask_price: u32,
    pub giveable: bool,
    pub focus: bool,
    pub give: Option<BuyRequest>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuyFacts {
    pub time: f32,
    pub time_low: bool,
    pub weapons: Vec<Card>,
    pub armor: Vec<Card>,
    pub abilities: Vec<Card>,
    pub reward_win: u32,
    pub reward_loss: u32,
    pub team: Vec<Mate>,
    pub signature: Card,
    pub gate: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MapItem {
    pub pos: [f32; 2],
    pub yaw_degrees: f32,
    pub color: Option<&'static str>,
    pub fade: f32,
    pub radius_m: f32,
    pub icon: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MinimapFacts {
    pub me: Vec<MapItem>,
    pub players: Vec<MapItem>,
    pub device: Vec<MapItem>,
    pub planted: Vec<MapItem>,
    pub orbs: Vec<MapItem>,
    pub enemies: Vec<MapItem>,
    pub lost: Vec<MapItem>,
    pub smokes: Vec<MapItem>,
    pub utility: Vec<MapItem>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrbFacts {
    pub hint: bool,
    pub channel: bool,
    pub progress: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UltFacts {
    pub active: bool,
    pub ally: bool,
    pub name: String,
    pub icon: String,
    pub ultimate: String,
    pub color: Tone,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StatusFacts {
    pub kind: String,
    pub label: String,
    pub left: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoardRow {
    pub name: String,
    pub icon: String,
    pub k: u32,
    pub d: u32,
    pub a: u32,
    pub adr: u32,
    pub loadout: String,
    pub credits: String,
    pub ult: u32,
    pub ult_need: u32,
    pub ult_ready: bool,
    pub alive: bool,
    pub me: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResultFacts {
    pub title: String,
    pub reason: String,
    pub color: Tone,
    pub mvp: String,
    pub mvp_icon: String,
    pub mvp_color: Tone,
    pub mvp_portrait: String,
    pub mvp_name: String,
    pub mvp_stats: String,
    pub mvp_char: String,
    pub next_seconds: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AbilityCard {
    pub icon: String,
    pub key: String,
    pub name: String,
    pub desc: String,
    pub ultimate: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectCard {
    pub key: String,
    pub image: String,
    pub name: String,
    pub role: String,
    pub selected: bool,
    pub abilities: [AbilityCard; 4],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectFacts {
    pub cards: Vec<SelectCard>,
    pub level: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadingFacts {
    pub map: String,
    pub mode: String,
    pub percent: u32,
    pub progress: f32,
    pub site_a_note: String,
    pub site_b_note: String,
    pub tips: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Slider {
    pub frac: f32,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyRow {
    pub action: String,
    pub binding: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SettingsFacts {
    pub sliders: [Slider; 11],
    pub reticle: Reticle,
    pub crosshair: usize,
    pub crosshair_color: String,
    pub invert: bool,
    pub level: Option<usize>,
    pub keys: Vec<KeyRow>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reticle {
    pub gap: f32,
    pub length: f32,
    pub thickness: f32,
    pub outline: f32,
    pub dot: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MenuFacts {
    pub screen: Option<Shown>,
    pub focus: Option<usize>,
    pub select: SelectFacts,
    pub loading: LoadingFacts,
    pub settings: SettingsFacts,
    pub can_quit: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpectateFacts {
    pub active: bool,
    pub name: String,
    pub icon: String,
    pub color: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointFacts {
    pub active: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PickupFacts {
    pub active: bool,
    pub key: String,
    pub weapon: String,
    pub icon: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CastFacts {
    pub active: bool,
    pub key: String,
    pub icon: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlaceFacts {
    pub callout: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KillFacts {
    pub active: bool,
    pub count: u32,
    pub label: String,
    pub victim_icon: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DamageMark {
    pub angle_radians: f32,
    pub age: f32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct HudFacts {
    pub version: u64,
    pub screen: Option<Shown>,
    pub keys: Vec<String>,
    pub round: RoundFacts,
    pub score: [u32; 2],
    pub teams: [TeamFacts; 2],
    pub players: [Vec<PlateFacts>; 2],
    pub feed: Vec<FeedLine>,
    pub player: PlayerFacts,
    pub abilities: [AbilityFacts; 4],
    pub weapon: WeaponFacts,
    pub spread_px: f32,
    pub action: ActionFacts,
    pub buy: BuyFacts,
    pub board: [Vec<BoardRow>; 2],
    pub result: ResultFacts,
    pub minimap: MinimapFacts,
    pub menu: MenuFacts,
    pub spectate: SpectateFacts,
    pub place: PlaceFacts,
    pub scope: u32,
    pub pickup: PickupFacts,
    pub cast: CastFacts,
    pub point: PointFacts,
    pub orb: OrbFacts,
    pub statuses: Vec<StatusFacts>,
    pub blind: f32,
    pub ult: UltFacts,
    pub armor_break: Option<f32>,
    pub kill: KillFacts,
    pub scope_ease: f32,
    pub damage: Vec<DamageMark>,
    pub phase: Option<Phase>,
    pub live: bool,
}

impl HudFacts {
    pub fn cards(&self) -> impl Iterator<Item = &Card> + '_ {
        self.buy
            .weapons
            .iter()
            .chain(&self.buy.armor)
            .chain(&self.buy.abilities)
    }
}
