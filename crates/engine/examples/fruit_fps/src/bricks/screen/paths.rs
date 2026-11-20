use brick_interface::{Area, Facts, Pen, Point, Seg, Slot, Val};

use super::control::CardList;
use super::hud::{
    AbilityCard, AbilityFacts, ActionFacts, BoardRow, BuyFacts, Card, CastFacts, FeedLine,
    HudFacts, KeyRow, KillFacts, LoadingFacts, MapItem, Mate, MinimapFacts, OrbFacts, PickupFacts,
    PlaceFacts, PlateFacts, PlayerFacts, PointFacts, ResultFacts, RoundFacts, SelectCard,
    SettingsFacts, Shown, Slider, SpectateFacts, StatusFacts, TeamFacts, UltFacts, WeaponFacts,
};
use super::{ScreenSource, menu, pieces};

macro_rules! fields {
    ($kind:ty, $me:ident { $($name:literal => $value:expr),* $(,)? }) => {
        impl $kind {
            const FIELDS: &'static [&'static str] = &[$($name),*];

            fn field(&self, field: u32) -> Val<'_> {
                let $me = self;
                let mut at = 0;
                $(
                    if field == at {
                        return $value;
                    }
                    at += 1;
                )*
                let _ = at;
                Val::Unknown
            }
        }
    };
}

fn whole(value: u32) -> Val<'static> {
    Val::Num(value as f32)
}

fn maybe(value: Option<u32>) -> Val<'static> {
    value.map_or(Val::Missing, whole)
}

fields!(RoundFacts, me {
    "time" => Val::Num(me.time),
    "number" => whole(me.number),
    "planted" => Val::Bool(me.planted),
    "fuse" => Val::Num(me.fuse),
    "site" => Val::Text(&me.site),
    "time_low" => Val::Bool(me.time_low),
    "fuse_low" => Val::Bool(me.fuse_low),
    "overtime" => Val::Bool(me.overtime),
    "decider" => Val::Bool(me.decider),
    "swapped" => Val::Bool(me.swapped),
    "last_of_half" => Val::Bool(me.last_of_half),
    "device" => Val::Text(&me.device),
    "buying" => Val::Bool(me.buying),
    "ready" => Val::Bool(me.ready),
    "ready_left" => Val::Num(me.ready_left),
    "ready_count" => whole(me.ready_count),
    "go" => Val::Bool(me.go),
});

fields!(TeamFacts, me {
    "color" => Val::Text(me.color.key()),
    "name" => Val::Text(&me.name),
});

fields!(PlateFacts, me {
    "icon" => Val::Text(&me.icon),
    "alive" => Val::Bool(me.alive),
    "health" => Val::Num(me.health),
    "ult_ready" => Val::Bool(me.ult_ready),
});

fields!(PickupFacts, me {
    "active" => Val::Bool(me.active),
    "key" => Val::Text(&me.key),
    "weapon" => Val::Text(&me.weapon),
    "icon" => Val::Text(&me.icon),
});

fields!(CastFacts, me {
    "active" => Val::Bool(me.active),
    "key" => Val::Text(&me.key),
    "icon" => Val::Text(&me.icon),
});

fields!(FeedLine, me {
    "killer" => Val::Text(&me.killer),
    "killer_color" => Val::Text(me.killer_color.key()),
    "weapon" => Val::Text(&me.weapon),
    "headshot" => Val::Bool(me.headshot),
    "victim" => Val::Text(&me.victim),
    "victim_color" => Val::Text(me.victim_color.key()),
    "mine" => Val::Bool(me.mine),
    "ability" => Val::Text(&me.ability),
    "world" => Val::Bool(me.world),
    "killer_char" => Val::Text(&me.killer_char),
    "victim_char" => Val::Text(&me.victim_char),
    "wall" => Val::Bool(me.wall),
    "assist_char" => Val::Text(&me.assist_char),
    "assist_color" => Val::Text(me.assist_color.key()),
    "assist_more" => whole(me.assist_more),
    "assisted" => Val::Bool(me.assisted),
});

fields!(PlayerFacts, me {
    "health" => Val::Num(me.health),
    "health_low" => Val::Bool(me.health_low),
    "armor" => Val::Num(me.armor),
    "armor_icon" => Val::Text(&me.armor_icon),
    "credits" => whole(me.credits),
    "char_name" => Val::Text(&me.char_name),
    "role" => Val::Text(&me.role),
    "icon" => Val::Text(&me.icon),
});

fields!(AbilityFacts, me {
    "icon" => Val::Text(&me.icon),
    "max" => whole(me.max),
    "charges" => whole(me.charges),
    "empty" => Val::Bool(me.empty),
    "cooling" => Val::Bool(me.cooling),
    "cooldown" => Val::Num(me.cooldown),
    "cooldown_frac" => Val::Num(me.cooldown_frac),
    "equipped" => Val::Bool(me.equipped),
    "points" => whole(me.points),
    "points_needed" => whole(me.points_needed),
    "ready" => Val::Bool(me.ready),
});

fields!(WeaponFacts, me {
    "name" => Val::Text(&me.name),
    "icon" => Val::Text(&me.icon),
    "mag" => maybe(me.mag),
    "mag_size" => maybe(me.mag_size),
    "reserve" => maybe(me.reserve),
    "low" => Val::Bool(me.low),
    "reload" => me.reload.map_or(Val::Missing, Val::Num),
    "down" => Val::Bool(me.down),
    "down_left" => Val::Num(me.down_left),
});

fields!(ActionFacts, me {
    "active" => Val::Bool(me.active),
    "label" => Val::Text(&me.label),
    "left" => Val::Num(me.left),
    "progress" => Val::Num(me.progress),
    "color" => Val::Text(me.color.key()),
    "checkpoint" => Val::Bool(me.checkpoint),
    "kind" => Val::Text(&me.kind),
});

fields!(Card, me {
    "points" => whole(me.points),
    "kind" => Val::Text(&me.kind),
    "name" => Val::Text(&me.name),
    "icon" => Val::Text(&me.icon),
    "price" => Val::Text(&me.price),
    "owned" => Val::Bool(me.owned),
    "afford" => Val::Bool(me.afford),
    "focus" => Val::Bool(me.focus),
    "need" => whole(me.need),
    "closed" => Val::Bool(me.closed),
    "ask" => Val::Bool(me.ask),
    "asked" => Val::Bool(me.asked),
    "sell" => whole(me.sell_credits),
    "max" => whole(me.max),
    "charges" => whole(me.charges),
    "stats.damage" => Val::Num(me.stats.damage),
    "stats.rate" => Val::Num(me.stats.rate),
    "stats.range" => Val::Num(me.stats.range),
});

fields!(BuyFacts, me {
    "time" => Val::Num(me.time),
    "time_low" => Val::Bool(me.time_low),
    "reward_win" => whole(me.reward_win),
    "gate" => Val::Text(&me.gate),
    "reward_loss" => whole(me.reward_loss),
});

fields!(Mate, me {
    "name" => Val::Text(&me.name),
    "icon" => Val::Text(&me.icon),
    "credits" => whole(me.credits),
    "can_rifle" => Val::Bool(me.can_rifle),
    "asked" => Val::Bool(me.asked),
    "ask_icon" => Val::Text(&me.ask_icon),
    "ask_price" => whole(me.ask_price),
    "giveable" => Val::Bool(me.giveable),
    "focus" => Val::Bool(me.focus),
});

fields!(BoardRow, me {
    "name" => Val::Text(&me.name),
    "icon" => Val::Text(&me.icon),
    "k" => whole(me.k),
    "d" => whole(me.d),
    "a" => whole(me.a),
    "adr" => whole(me.adr),
    "loadout" => Val::Text(&me.loadout),
    "credits" => Val::Text(&me.credits),
    "ult" => whole(me.ult),
    "ult_need" => whole(me.ult_need),
    "ult_ready" => Val::Bool(me.ult_ready),
    "alive" => Val::Bool(me.alive),
    "me" => Val::Bool(me.me),
});

fields!(ResultFacts, me {
    "title" => Val::Text(&me.title),
    "reason" => Val::Text(&me.reason),
    "color" => Val::Text(me.color.key()),
    "mvp" => Val::Text(&me.mvp),
    "mvp_icon" => Val::Text(&me.mvp_icon),
    "mvp_color" => Val::Text(me.mvp_color.key()),
    "mvp_portrait" => Val::Text(&me.mvp_portrait),
    "mvp_name" => Val::Text(&me.mvp_name),
    "mvp_stats" => Val::Text(&me.mvp_stats),
    "mvp_char" => Val::Text(&me.mvp_char),
    "next_seconds" => Val::Num(me.next_seconds),
});

fields!(Slider, me {
    "frac" => Val::Num(me.frac),
    "text" => Val::Text(&me.text),
});

fields!(KeyRow, me {
    "action" => Val::Text(&me.action),
    "binding" => Val::Text(&me.binding),
});

fields!(SettingsFacts, me {
    "crosshair_color" => Val::Text(&me.crosshair_color),
    "invert" => Val::Bool(me.invert),
    "crosshair_dot" => Val::Bool(me.reticle.dot),
    "hover_back" => Val::Bool(me.hover_back),
    "press_back" => Val::Bool(me.press_back),
    "hover_invert" => Val::Bool(me.hover_invert),
    "hover_dot" => Val::Bool(me.hover_dot),
});

fields!(AbilityCard, me {
    "icon" => Val::Text(&me.icon),
    "key" => Val::Text(&me.key),
    "name" => Val::Text(&me.name),
    "desc" => Val::Text(&me.desc),
    "ultimate" => Val::Bool(me.ultimate),
});

fields!(SelectCard, me {
    "key" => Val::Text(&me.key),
    "name" => Val::Text(&me.name),
    "role" => Val::Text(&me.role),
    "selected" => Val::Bool(me.selected),
    "hover" => Val::Bool(me.hover),
    "press" => Val::Bool(me.press),
});

fields!(LoadingFacts, me {
    "map" => Val::Text(&me.map),
    "mode" => Val::Text(&me.mode),
    "percent" => whole(me.percent),
    "progress" => Val::Num(me.progress),
    "site_a_note" => Val::Text(&me.site_a_note),
    "site_b_note" => Val::Text(&me.site_b_note),
});

fields!(SpectateFacts, me {
    "active" => Val::Bool(me.active),
    "name" => Val::Text(&me.name),
    "icon" => Val::Text(&me.icon),
    "color" => Val::Text(&me.color),
});

fields!(PlaceFacts, me {
    "callout" => Val::Text(&me.callout),
});

fields!(PointFacts, me {
    "active" => Val::Bool(me.active),
    "reason" => Val::Text(&me.reason),
});

fields!(OrbFacts, me {
    "hint" => Val::Bool(me.hint),
    "channel" => Val::Bool(me.channel),
    "progress" => Val::Num(me.progress),
});

fields!(StatusFacts, me {
    "kind" => Val::Text(&me.kind),
    "label" => Val::Text(&me.label),
    "left" => Val::Num(me.left),
});

fields!(KillFacts, me {
    "active" => Val::Bool(me.active),
    "count" => whole(me.count),
    "label" => Val::Text(&me.label),
    "victim_icon" => Val::Text(&me.victim_icon),
});

fields!(UltFacts, me {
    "active" => Val::Bool(me.active),
    "ally" => Val::Bool(me.ally),
    "name" => Val::Text(&me.name),
    "icon" => Val::Text(&me.icon),
    "ultimate" => Val::Text(&me.ultimate),
    "color" => Val::Text(me.color.key()),
});

const MINIMAP: [&str; 9] = [
    "me", "players", "device", "planted", "orbs", "enemies", "lost", "smokes", "utility",
];

impl MinimapFacts {
    fn list(&self, index: usize) -> &[MapItem] {
        match index {
            0 => &self.me,
            1 => &self.players,
            2 => &self.device,
            3 => &self.planted,
            4 => &self.orbs,
            5 => &self.enemies,
            6 => &self.lost,
            7 => &self.smokes,
            _ => &self.utility,
        }
    }
}

const SLIDERS: [&str; 11] = [
    "sensitivity",
    "fov",
    "master",
    "effects",
    "interface",
    "render_scale",
    "crosshair_gap",
    "crosshair_length",
    "crosshair_thickness",
    "crosshair_outline",
    "footsteps",
];

const ABILITY_SLOTS: [&str; 4] = ["c", "q", "e", "x"];
const CARD_LISTS: [&str; 3] = ["weapons", "armor", "abilities"];

const WHOLE: u32 = 0xFF;

const ROUND: u32 = 1;
const KEYS: u32 = 2;
const SCORE: u32 = 3;
const TEAMS: u32 = 4;
const PLAYERS: u32 = 5;
const FEED: u32 = 6;
const PLAYER: u32 = 7;
const ABILITIES: u32 = 8;
const WEAPON: u32 = 9;
const ACTION: u32 = 10;
const BUY: u32 = 11;
const BUY_CARDS: u32 = 12;
const BUY_TEAM: u32 = 13;
const BUY_SIGNATURE: u32 = 14;
const BOARD: u32 = 15;
const RESULT: u32 = 16;
const LISTS: u32 = 17;
const FOCUS: u32 = 18;
const CAN_QUIT: u32 = 19;
const SELECT_CARDS: u32 = 20;
const SELECT_ABILITIES: u32 = 21;
const LEVEL: u32 = 22;
const LOADING: u32 = 23;
const TIPS: u32 = 24;
const SETTINGS: u32 = 25;
const SETTINGS_KEYS: u32 = 26;
const CROSSHAIR: u32 = 27;
const SLIDER: u32 = 28;
const SPECTATE: u32 = 29;
const PLACE: u32 = 30;
const SCOPE: u32 = 31;
const PICKUP: u32 = 32;
const POINT: u32 = 33;
const CAST: u32 = 34;
const ORB: u32 = 35;
const STATUSES: u32 = 36;
const BLIND: u32 = 37;
const KILL: u32 = 38;
const ULT: u32 = 39;
const HOVER: u32 = 40;
const PRESS: u32 = 41;
const LEVEL_HOVER: u32 = 42;
const LEVEL_PRESS: u32 = 43;
const CROSSHAIR_HOVER: u32 = 44;

fn pack(group: u32, a: usize, b: usize, field: u32) -> Option<Slot> {
    let small = |value: usize| u32::try_from(value).ok().filter(|value| *value < 0xFF);
    Some((group << 24) | (small(a)? << 16) | (small(b)? << 8) | field)
}

fn unpack(slot: Slot) -> (u32, usize, usize, u32) {
    (
        slot >> 24,
        ((slot >> 16) & 0xFF) as usize,
        ((slot >> 8) & 0xFF) as usize,
        slot & 0xFF,
    )
}

fn key(path: &[Seg]) -> Option<(&str, &[Seg])> {
    match path {
        [Seg::Key(name), rest @ ..] => Some((name.as_str(), rest)),
        [Seg::Index(_), ..] | [] => None,
    }
}

fn leaf(path: &[Seg]) -> Option<&str> {
    match key(path)? {
        (name, []) => Some(name),
        (_, [_, ..]) => None,
    }
}

fn pick(path: &[Seg]) -> Option<(usize, &[Seg])> {
    let (seg, rest) = path.split_first()?;
    let index = match seg {
        Seg::Index(index) => *index,
        Seg::Key(name) => match name.as_str() {
            "left" => 0,
            "right" => 1,
            _ => return None,
        },
    };
    Some((index, rest))
}

fn among(names: &[&str], name: &str) -> Option<usize> {
    names.iter().position(|known| *known == name)
}

fn named(fields: &[&str], path: &[Seg]) -> Option<u32> {
    let mut name = String::new();
    for seg in path {
        let Seg::Key(part) = seg else {
            return None;
        };
        if !name.is_empty() {
            name.push('.');
        }
        name.push_str(part);
    }
    among(fields, &name).and_then(|at| u32::try_from(at).ok())
}

fn item(fields: &[&str], path: &[Seg]) -> Option<u32> {
    if path.is_empty() {
        return Some(WHOLE);
    }
    named(fields, path)
}

fn one(group: u32, fields: &[&str], path: &[Seg]) -> Option<Slot> {
    pack(group, 0, 0, named(fields, path)?)
}

fn listed(group: u32, fields: &[&str], path: &[Seg]) -> Option<Slot> {
    let (index, rest) = pick(path)?;
    pack(group, index, 0, item(fields, rest)?)
}

fn sided(group: u32, fields: &[&str], path: &[Seg]) -> Option<Slot> {
    let (side, rest) = pick(path)?;
    let (index, rest) = pick(rest)?;
    pack(group, side, index, item(fields, rest)?)
}

fn screen_buttons(shown: Shown) -> &'static [&'static str] {
    match shown {
        Shown::Title => &menu::TITLE,
        Shown::Pause => &menu::PAUSE,
        Shown::MatchEnd => &menu::MATCH_END,
        Shown::Settings => &menu::SETTINGS_ROWS,
        Shown::Select => &menu::SELECT,
        Shown::Loading | Shown::Hud | Shown::Buy | Shown::Scoreboard | Shown::RoundEnd => &[],
    }
}

fn focus(shown: Shown, path: &[Seg]) -> Option<Slot> {
    let name = leaf(path)?;
    let (group, button) = if let Some(button) = name.strip_prefix("focus_") {
        (FOCUS, button)
    } else if let Some(button) = name.strip_prefix("hover_") {
        (HOVER, button)
    } else {
        (PRESS, name.strip_prefix("press_")?)
    };
    pack(group, shown as usize, among(screen_buttons(shown), button)?, 0)
}

fn level(look: &ScreenSource, (place, name): (usize, &str)) -> Option<Slot> {
    let (group, level) = if let Some(level) = name.strip_prefix("bots_") {
        (LEVEL, level)
    } else if let Some(level) = name.strip_prefix("hover_bots_") {
        (LEVEL_HOVER, level)
    } else {
        (LEVEL_PRESS, name.strip_prefix("press_bots_")?)
    };
    let index = look.menu.levels.iter().position(|known| known == level)?;
    pack(group, place, index, 0)
}

fn select(look: &ScreenSource, path: &[Seg]) -> Option<Slot> {
    match key(path)? {
        ("cards", rest) => {
            let (card, rest) = pick(rest)?;
            match key(rest)? {
                ("abilities", rest) => {
                    let (ability, rest) = pick(rest)?;
                    pack(SELECT_ABILITIES, card, ability, named(AbilityCard::FIELDS, rest)?)
                }
                _ => pack(SELECT_CARDS, card, 0, named(SelectCard::FIELDS, rest)?),
            }
        }
        (name, []) => level(look, (0, name)),
        (_, [_, ..]) => None,
    }
}

fn loading(path: &[Seg]) -> Option<Slot> {
    match key(path)? {
        ("tips", rest) => match pick(rest)? {
            (index, []) => pack(TIPS, index, 0, 0),
            (_, [_, ..]) => None,
        },
        _ => one(LOADING, LoadingFacts::FIELDS, path),
    }
}

fn settings(look: &ScreenSource, path: &[Seg]) -> Option<Slot> {
    let (name, rest) = key(path)?;
    if name == "keys" {
        return listed(SETTINGS_KEYS, KeyRow::FIELDS, rest);
    }
    if let Some(found) = one(SETTINGS, SettingsFacts::FIELDS, path) {
        return Some(found);
    }
    if rest.is_empty()
        && (name.starts_with("bots_")
            || name.starts_with("hover_bots_")
            || name.starts_with("press_bots_"))
    {
        return level(look, (1, name));
    }
    if let Some(index) = name
        .strip_prefix("hover_crosshair_")
        .and_then(|at| at.parse::<usize>().ok())
    {
        return if rest.is_empty() {
            pack(CROSSHAIR_HOVER, index, 0, 0)
        } else {
            None
        };
    }
    if let Some(index) = name
        .strip_prefix("crosshair_")
        .and_then(|at| at.parse::<usize>().ok())
    {
        return if rest.is_empty() {
            pack(CROSSHAIR, index, 0, 0)
        } else {
            None
        };
    }
    pack(SLIDER, among(&SLIDERS, name)?, 0, named(Slider::FIELDS, rest)?)
}

fn menu_screen(look: &ScreenSource, shown: Shown, path: &[Seg]) -> Option<Slot> {
    if shown == Shown::Settings {
        return settings(look, path).or_else(|| focus(shown, path));
    }
    if let Some(found) = focus(shown, path) {
        return Some(found);
    }
    match shown {
        Shown::Select => select(look, path),
        Shown::Loading => loading(path),
        Shown::Settings => None,
        Shown::Title => match leaf(path)? {
            "can_quit" => pack(CAN_QUIT, 0, 0, 0),
            _ => None,
        },
        Shown::Pause
        | Shown::MatchEnd
        | Shown::Hud
        | Shown::Buy
        | Shown::Scoreboard
        | Shown::RoundEnd => None,
    }
}

fn buy(path: &[Seg]) -> Option<Slot> {
    let (name, rest) = key(path)?;
    if let Some(list) = among(&CARD_LISTS, name) {
        let (index, rest) = pick(rest)?;
        return pack(BUY_CARDS, list, index, item(Card::FIELDS, rest)?);
    }
    match name {
        "team" => listed(BUY_TEAM, Mate::FIELDS, rest),
        "signature" => pack(BUY_SIGNATURE, 0, 0, item(Card::FIELDS, rest)?),
        _ => one(BUY, BuyFacts::FIELDS, path),
    }
}

fn resolve(look: &ScreenSource, path: &[Seg]) -> Option<Slot> {
    let (name, rest) = key(path)?;
    match name {
        "round" => one(ROUND, RoundFacts::FIELDS, rest),
        "keys" => pack(KEYS, menu::action(leaf(rest)?)?, 0, 0),
        "score" => match pick(rest)? {
            (side, []) => pack(SCORE, side, 0, 0),
            (_, [_, ..]) => None,
        },
        "teams" => {
            let (side, rest) = pick(rest)?;
            pack(TEAMS, side, 0, named(TeamFacts::FIELDS, rest)?)
        }
        "players" => sided(PLAYERS, PlateFacts::FIELDS, rest),
        "feed" => listed(FEED, FeedLine::FIELDS, rest),
        "player" => one(PLAYER, PlayerFacts::FIELDS, rest),
        "abilities" => {
            let (slot, rest) = key(rest)?;
            pack(
                ABILITIES,
                among(&ABILITY_SLOTS, slot)?,
                0,
                named(AbilityFacts::FIELDS, rest)?,
            )
        }
        "weapon" => one(WEAPON, WeaponFacts::FIELDS, rest),
        "action" => one(ACTION, ActionFacts::FIELDS, rest),
        "buy" => buy(rest),
        "board" => sided(BOARD, BoardRow::FIELDS, rest),
        "result" => one(RESULT, ResultFacts::FIELDS, rest),
        "minimap" => pack(LISTS, among(&MINIMAP, leaf(rest)?)?, 0, 0),
        "spectate" => one(SPECTATE, SpectateFacts::FIELDS, rest),
        "place" => one(PLACE, PlaceFacts::FIELDS, rest),
        "scope" if rest.is_empty() => pack(SCOPE, 0, 0, 0),
        "pickup" => one(PICKUP, PickupFacts::FIELDS, rest),
        "point" => one(POINT, PointFacts::FIELDS, rest),
        "cast" => one(CAST, CastFacts::FIELDS, rest),
        "orb" => one(ORB, OrbFacts::FIELDS, rest),
        "statuses" => listed(STATUSES, StatusFacts::FIELDS, rest),
        "blind" if rest.is_empty() => pack(BLIND, 0, 0, 0),
        "kill" => one(KILL, KillFacts::FIELDS, rest),
        "ult" => one(ULT, UltFacts::FIELDS, rest),
        "title" => menu_screen(look, Shown::Title, rest),
        "pause" => menu_screen(look, Shown::Pause, rest),
        "match_end" => menu_screen(look, Shown::MatchEnd, rest),
        "select" => menu_screen(look, Shown::Select, rest),
        "loading" => menu_screen(look, Shown::Loading, rest),
        "settings" => menu_screen(look, Shown::Settings, rest),
        _ => None,
    }
}

pub(super) fn card(slot: Slot) -> Option<(CardList, usize)> {
    let (group, a, b, _) = unpack(slot);
    match group {
        BUY_CARDS => Some((
            [CardList::Weapons, CardList::Armor, CardList::Abilities].get(a).copied()?,
            b,
        )),
        BUY_TEAM => Some((CardList::Team, a)),
        _ => None,
    }
}

fn item_of<'a, T>(
    list: &'a [T],
    index: usize,
    field: u32,
    read: impl FnOnce(&'a T, u32) -> Val<'a>,
) -> Val<'a> {
    match list.get(index) {
        None => Val::Missing,
        Some(_) if field == WHOLE => Val::Bool(true),
        Some(found) => read(found, field),
    }
}

fn card_of(card: &Card, field: u32) -> Val<'_> {
    if field == WHOLE {
        return Val::Bool(!card.name.is_empty());
    }
    card.field(field)
}

fn level_is(level: Option<usize>, index: usize) -> Val<'static> {
    Val::Bool(level == Some(index))
}

impl Facts for HudFacts {
    type Look = ScreenSource;
    const SCREENS: &'static [&'static str] = &Shown::KEYS;
    const PIECES: &'static [&'static str] = &pieces::PIECES;

    fn slot(look: &ScreenSource, path: &[Seg]) -> Option<Slot> {
        resolve(look, path)
    }

    fn spot(look: &ScreenSource, id: &str) -> Option<Slot> {
        menu::spot(look, id)
    }

    fn version(&self) -> u64 {
        self.version
    }

    fn screen(&self) -> Option<usize> {
        self.screen.map(|shown| shown as usize)
    }

    fn points(&self, slot: Slot, each: &mut dyn FnMut(Point<'_>)) {
        let (group, list, _, _) = unpack(slot);
        if group != LISTS {
            return;
        }
        for item in self.minimap.list(list) {
            each(Point {
                pos: item.pos,
                yaw_degrees: item.yaw_degrees,
                color: item.color,
                fade: item.fade,
                radius_m: item.radius_m,
                icon: &item.icon,
            });
        }
    }

    fn piece(&self, look: &ScreenSource, piece: usize, area: Area, pen: &mut Pen<'_>) {
        pieces::piece(self, look, piece, area, pen);
    }

    fn read(&self, slot: Slot) -> Val<'_> {
        let (group, a, b, field) = unpack(slot);
        let menu = &self.menu;
        match group {
            ROUND => self.round.field(field),
            KEYS => self
                .keys
                .get(a)
                .map_or(Val::Missing, |spoken| Val::Text(spoken)),
            SCORE => self.score.get(a).map_or(Val::Missing, |score| whole(*score)),
            TEAMS => self.teams.get(a).map_or(Val::Missing, |team| team.field(field)),
            PLAYERS => self.players.get(a).map_or(Val::Missing, |team| {
                item_of(team, b, field, PlateFacts::field)
            }),
            FEED => item_of(&self.feed, a, field, FeedLine::field),
            PLAYER => self.player.field(field),
            ABILITIES => self
                .abilities
                .get(a)
                .map_or(Val::Missing, |ability| ability.field(field)),
            WEAPON => self.weapon.field(field),
            ACTION => self.action.field(field),
            BUY => self.buy.field(field),
            BUY_CARDS => {
                let lists = [&self.buy.weapons, &self.buy.armor, &self.buy.abilities];
                lists
                    .get(a)
                    .and_then(|cards| cards.get(b))
                    .map_or(Val::Missing, |card| card_of(card, field))
            }
            BUY_TEAM => item_of(&self.buy.team, a, field, Mate::field),
            BUY_SIGNATURE => card_of(&self.buy.signature, field),
            BOARD => self
                .board
                .get(a)
                .map_or(Val::Missing, |rows| item_of(rows, b, field, BoardRow::field)),
            RESULT => self.result.field(field),
            LISTS => whole(self.minimap.list(a).len() as u32),
            FOCUS => Val::Bool(
                menu.screen.map(|shown| shown as usize) == Some(a) && menu.focus == Some(b),
            ),
            HOVER => Val::Bool(
                menu.screen.map(|shown| shown as usize) == Some(a) && menu.hover == Some(b),
            ),
            PRESS => Val::Bool(
                menu.screen.map(|shown| shown as usize) == Some(a) && menu.press == Some(b),
            ),
            CAN_QUIT => Val::Bool(menu.can_quit),
            SELECT_CARDS => menu
                .select
                .cards
                .get(a)
                .map_or(Val::Missing, |card| card.field(field)),
            SELECT_ABILITIES => menu
                .select
                .cards
                .get(a)
                .and_then(|card| card.abilities.get(b))
                .map_or(Val::Missing, |ability| ability.field(field)),
            LEVEL if a == 0 => level_is(menu.select.level, b),
            LEVEL => level_is(menu.settings.level, b),
            LEVEL_HOVER if a == 0 => level_is(menu.select.hover_level, b),
            LEVEL_HOVER => level_is(menu.settings.hover_level, b),
            LEVEL_PRESS if a == 0 => level_is(menu.select.press_level, b),
            LEVEL_PRESS => level_is(menu.settings.press_level, b),
            CROSSHAIR_HOVER => Val::Bool(menu.settings.hover_crosshair == Some(a)),
            LOADING => menu.loading.field(field),
            TIPS => menu
                .loading
                .tips
                .get(a)
                .map_or(Val::Missing, |tip| Val::Text(tip)),
            SETTINGS => menu.settings.field(field),
            SETTINGS_KEYS if field == WHOLE => Val::Bool(a < menu.settings.keys.len()),
            SETTINGS_KEYS => menu
                .settings
                .keys
                .get(a)
                .map_or(Val::Missing, |row| row.field(field)),
            CROSSHAIR => Val::Bool(menu.settings.crosshair == a),
            SLIDER => menu
                .settings
                .sliders
                .get(a)
                .map_or(Val::Missing, |slider| slider.field(field)),
            SPECTATE => self.spectate.field(field),
            PLACE => self.place.field(field),
            SCOPE => whole(self.scope),
            PICKUP => self.pickup.field(field),
            POINT => self.point.field(field),
            CAST => self.cast.field(field),
            ORB => self.orb.field(field),
            STATUSES => item_of(&self.statuses, a, field, StatusFacts::field),
            BLIND => Val::Num(self.blind),
            KILL => self.kill.field(field),
            ULT => self.ult.field(field),
            _ => Val::Unknown,
        }
    }
}
