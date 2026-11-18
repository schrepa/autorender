use crate::play::screen::{
    AbilityCard, AbilityFacts, ActionFacts, BoardRow, BuyFacts, Card, CastFacts, FeedLine,
    HudFacts, KeyRow, LoadingFacts, MapItem, Mate, MenuFacts, MinimapFacts, PickupFacts,
    PlateFacts, PlayerFacts, ResultFacts, RoundFacts, SelectCard, SelectFacts, SettingsFacts,
    Shown, Slider, TeamFacts, WeaponFacts,
};

use super::layout::Seg;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Val<'a> {
    Unknown,
    Missing,
    Bool(bool),
    Num(f32),
    Text(&'a str),
}

impl<'a> Val<'a> {
    pub(super) fn truthy(self) -> bool {
        match self {
            Val::Unknown | Val::Missing => false,
            Val::Bool(value) => value,
            Val::Num(value) => value != 0.0,
            Val::Text(text) => !text.is_empty(),
        }
    }

    pub(super) fn number(self) -> Option<f32> {
        match self {
            Val::Num(value) => Some(value),
            Val::Bool(value) => Some(f32::from(u8::from(value))),
            Val::Unknown | Val::Missing | Val::Text(_) => None,
        }
    }

    pub(super) fn text(self) -> Option<&'a str> {
        match self {
            Val::Text(text) => Some(text),
            Val::Unknown | Val::Missing | Val::Bool(_) | Val::Num(_) => None,
        }
    }
}

fn whole(value: u32) -> Val<'static> {
    Val::Num(value as f32)
}

fn maybe(value: Option<u32>) -> Val<'static> {
    value.map_or(Val::Missing, whole)
}

fn key(path: &[Seg]) -> Option<(&str, &[Seg])> {
    match path {
        [Seg::Key(name), rest @ ..] => Some((name.as_str(), rest)),
        [Seg::Index(_), ..] | [] => None,
    }
}

fn side_index(seg: &Seg) -> Option<usize> {
    match seg {
        Seg::Index(index) => Some(*index),
        Seg::Key(name) => match name.as_str() {
            "left" => Some(0),
            "right" => Some(1),
            _ => None,
        },
    }
}

fn picked<'a, T>(
    list: &'a [T],
    path: &'a [Seg],
) -> Result<(Option<&'a T>, &'a [Seg]), Val<'static>> {
    match path {
        [seg, rest @ ..] => match side_index(seg) {
            Some(index) => Ok((list.get(index), rest)),
            None => Err(Val::Unknown),
        },
        [] => Err(Val::Unknown),
    }
}

fn within<'a, T>(
    list: &'a [T],
    path: &'a [Seg],
    field: impl Fn(&'a T, &'a [Seg]) -> Val<'a>,
) -> Val<'a> {
    match picked(list, path) {
        Ok((Some(item), rest)) => field(item, rest),
        Ok((None, _)) => Val::Missing,
        Err(unknown) => unknown,
    }
}

fn leaf<'a>(path: &[Seg], field: impl Fn(&str) -> Val<'a>) -> Val<'a> {
    match key(path) {
        Some((name, [])) => field(name),
        Some((_, [_, ..])) | None => Val::Unknown,
    }
}

impl RoundFacts {
    fn value(&self, path: &[Seg]) -> Val<'_> {
        leaf(path, |name| match name {
            "time" => Val::Num(self.time),
            "number" => whole(self.number),
            "planted" => Val::Bool(self.planted),
            "fuse" => Val::Num(self.fuse),
            "site" => Val::Text(&self.site),
            "time_low" => Val::Bool(self.time_low),
            "fuse_low" => Val::Bool(self.fuse_low),
            _ => Val::Unknown,
        })
    }
}

impl TeamFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "color" => Val::Text(self.color.key()),
            "name" => Val::Text(&self.name),
            _ => Val::Unknown,
        })
    }
}

impl PlateFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "icon" => Val::Text(&self.icon),
            "alive" => Val::Bool(self.alive),
            "health" => Val::Num(self.health),
            "ult_ready" => Val::Bool(self.ult_ready),
            _ => Val::Unknown,
        })
    }
}

impl PickupFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "active" => Val::Bool(self.active),
            "key" => Val::Text(&self.key),
            "weapon" => Val::Text(&self.weapon),
            "icon" => Val::Text(&self.icon),
            _ => Val::Unknown,
        })
    }
}

impl CastFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "active" => Val::Bool(self.active),
            "key" => Val::Text(&self.key),
            "icon" => Val::Text(&self.icon),
            _ => Val::Unknown,
        })
    }
}

impl FeedLine {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        if path.is_empty() {
            return Val::Bool(true);
        }
        leaf(path, |name| match name {
            "killer" => Val::Text(&self.killer),
            "killer_color" => Val::Text(self.killer_color.key()),
            "weapon" => Val::Text(&self.weapon),
            "headshot" => Val::Bool(self.headshot),
            "victim" => Val::Text(&self.victim),
            "victim_color" => Val::Text(self.victim_color.key()),
            "mine" => Val::Bool(self.mine),
            "ability" => Val::Text(&self.ability),
            "world" => Val::Bool(self.world),
            _ => Val::Unknown,
        })
    }
}

impl PlayerFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "health" => Val::Num(self.health),
            "health_low" => Val::Bool(self.health_low),
            "armor" => Val::Num(self.armor),
            "armor_icon" => Val::Text(&self.armor_icon),
            "credits" => whole(self.credits),
            "char_name" => Val::Text(&self.char_name),
            "role" => Val::Text(&self.role),
            _ => Val::Unknown,
        })
    }
}

impl AbilityFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "icon" => Val::Text(&self.icon),
            "max" => whole(self.max),
            "charges" => whole(self.charges),
            "empty" => Val::Bool(self.empty),
            "cooling" => Val::Bool(self.cooling),
            "cooldown" => Val::Num(self.cooldown),
            "points" => whole(self.points),
            "points_needed" => whole(self.points_needed),
            "ready" => Val::Bool(self.ready),
            _ => Val::Unknown,
        })
    }
}

impl WeaponFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "name" => Val::Text(&self.name),
            "mag" => maybe(self.mag),
            "mag_size" => maybe(self.mag_size),
            "reserve" => maybe(self.reserve),
            "low" => Val::Bool(self.low),
            "reload" => self.reload.map_or(Val::Missing, Val::Num),
            _ => Val::Unknown,
        })
    }
}

impl ActionFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "active" => Val::Bool(self.active),
            "label" => Val::Text(&self.label),
            "left" => Val::Num(self.left),
            "progress" => Val::Num(self.progress),
            "color" => Val::Text(self.color.key()),
            "checkpoint" => Val::Bool(self.checkpoint),
            _ => Val::Unknown,
        })
    }
}

impl Card {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        if let Some(("stats", rest)) = key(path) {
            let stats = self.stats;
            return leaf(rest, |name| match name {
                "damage" => Val::Num(stats.damage),
                "rate" => Val::Num(stats.rate),
                "range" => Val::Num(stats.range),
                _ => Val::Unknown,
            });
        }
        leaf(path, |name| match name {
            "points" => whole(self.points),
            "kind" => Val::Text(&self.kind),
            "name" => Val::Text(&self.name),
            "icon" => Val::Text(&self.icon),
            "price" => Val::Text(&self.price),
            "owned" => Val::Bool(self.owned),
            "afford" => Val::Bool(self.afford),
            "max" => whole(self.max),
            "charges" => whole(self.charges),
            _ => Val::Unknown,
        })
    }
}

impl BuyFacts {
    fn value<'a>(&'a self, path: &'a [Seg]) -> Val<'a> {
        match key(path) {
            Some(("time", [])) => Val::Num(self.time),
            Some(("time_low", [])) => Val::Bool(self.time_low),
            Some(("weapons", rest)) => within(&self.weapons, rest, Card::value),
            Some(("armor", rest)) => within(&self.armor, rest, Card::value),
            Some(("abilities", rest)) => within(&self.abilities, rest, Card::value),
            Some(("reward_win", [])) => whole(self.reward_win),
            Some(("reward_loss", [])) => whole(self.reward_loss),
            Some(("team", rest)) => within(&self.team, rest, Mate::value),
            Some(_) | None => Val::Unknown,
        }
    }
}

impl Mate {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "name" => Val::Text(&self.name),
            "icon" => Val::Text(&self.icon),
            "credits" => whole(self.credits),
            "can_rifle" => Val::Bool(self.can_rifle),
            _ => Val::Unknown,
        })
    }
}

impl MinimapFacts {
    fn value(&self, path: &[Seg]) -> Val<'static> {
        leaf(path, |name| match name {
            "me" => whole(self.me.len() as u32),
            "players" => whole(self.players.len() as u32),
            "device" => whole(self.device.len() as u32),
            _ => Val::Unknown,
        })
    }
}

impl BoardRow {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "name" => Val::Text(&self.name),
            "icon" => Val::Text(&self.icon),
            "k" => whole(self.k),
            "d" => whole(self.d),
            "a" => whole(self.a),
            "adr" => whole(self.adr),
            "loadout" => Val::Text(&self.loadout),
            "credits" => Val::Text(&self.credits),
            "ult" => whole(self.ult),
            "ult_need" => whole(self.ult_need),
            "alive" => Val::Bool(self.alive),
            "me" => Val::Bool(self.me),
            _ => Val::Unknown,
        })
    }
}

impl ResultFacts {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "title" => Val::Text(&self.title),
            "reason" => Val::Text(&self.reason),
            "color" => Val::Text(self.color.key()),
            "mvp" => Val::Text(&self.mvp),
            "mvp_icon" => Val::Text(&self.mvp_icon),
            "mvp_color" => Val::Text(self.mvp_color.key()),
            "mvp_portrait" => Val::Text(&self.mvp_portrait),
            "mvp_name" => Val::Text(&self.mvp_name),
            "mvp_stats" => Val::Text(&self.mvp_stats),
            "mvp_char" => Val::Text(&self.mvp_char),
            _ => Val::Unknown,
        })
    }
}

fn listed<'a, T>(
    list: &'a [T],
    path: &'a [Seg],
    field: impl Fn(&'a T, &'a [Seg]) -> Val<'a>,
) -> Val<'a> {
    match path {
        [Seg::Index(index)] => Val::Bool(*index < list.len()),
        [Seg::Index(_), _, ..] | [Seg::Key(_), ..] | [] => within(list, path, field),
    }
}

const SLIDERS: [&str; 5] = ["sensitivity", "fov", "master", "effects", "interface"];

impl Slider {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "frac" => Val::Num(self.frac),
            "text" => Val::Text(&self.text),
            _ => Val::Unknown,
        })
    }
}

impl KeyRow {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "action" => Val::Text(&self.action),
            "binding" => Val::Text(&self.binding),
            _ => Val::Unknown,
        })
    }
}

impl SettingsFacts {
    fn value<'a>(&'a self, path: &'a [Seg]) -> Val<'a> {
        match key(path) {
            Some(("crosshair_color", [])) => Val::Text(&self.crosshair_color),
            Some(("invert", [])) => Val::Bool(self.invert),
            Some(("keys", rest)) => listed(&self.keys, rest, KeyRow::value),
            Some((name, rest)) => {
                if let Some(index) = name
                    .strip_prefix("crosshair_")
                    .and_then(|at| at.parse::<usize>().ok())
                {
                    return if rest.is_empty() {
                        Val::Bool(self.crosshair == index)
                    } else {
                        Val::Unknown
                    };
                }
                match SLIDERS.iter().position(|slider| *slider == name) {
                    Some(index) => self.sliders[index].value(rest),
                    None => Val::Unknown,
                }
            }
            None => Val::Unknown,
        }
    }
}

impl AbilityCard {
    fn value<'a>(&'a self, path: &[Seg]) -> Val<'a> {
        leaf(path, |name| match name {
            "icon" => Val::Text(&self.icon),
            "key" => Val::Text(&self.key),
            "name" => Val::Text(&self.name),
            "desc" => Val::Text(&self.desc),
            "ultimate" => Val::Bool(self.ultimate),
            _ => Val::Unknown,
        })
    }
}

impl SelectCard {
    fn value<'a>(&'a self, path: &'a [Seg]) -> Val<'a> {
        if let Some(("abilities", rest)) = key(path) {
            return within(&self.abilities, rest, AbilityCard::value);
        }
        leaf(path, |name| match name {
            "image" => Val::Text(&self.image),
            "name" => Val::Text(&self.name),
            "role" => Val::Text(&self.role),
            "selected" => Val::Bool(self.selected),
            _ => Val::Unknown,
        })
    }
}

impl SelectFacts {
    fn value<'a>(&'a self, path: &'a [Seg]) -> Val<'a> {
        match key(path) {
            Some(("cards", rest)) => within(&self.cards, rest, SelectCard::value),
            Some((name, [])) => match name.strip_prefix("bots_") {
                Some(level) => Val::Bool(level == self.level),
                None => Val::Unknown,
            },
            Some(_) | None => Val::Unknown,
        }
    }
}

impl LoadingFacts {
    fn value<'a>(&'a self, path: &'a [Seg]) -> Val<'a> {
        match key(path) {
            Some(("tips", rest)) => within(&self.tips, rest, |tip, rest| {
                if rest.is_empty() {
                    Val::Text(tip)
                } else {
                    Val::Unknown
                }
            }),
            Some(_) | None => leaf(path, |name| match name {
                "map" => Val::Text(&self.map),
                "mode" => Val::Text(&self.mode),
                "percent" => whole(self.percent),
                "progress" => Val::Num(self.progress),
                "site_a_note" => Val::Text(&self.site_a_note),
                "site_b_note" => Val::Text(&self.site_b_note),
                _ => Val::Unknown,
            }),
        }
    }
}

impl MenuFacts {
    fn focused(&self, shown: Shown, path: &[Seg]) -> Option<Val<'static>> {
        match path {
            [Seg::Key(name)] => {
                let button = name.strip_prefix("focus_")?;
                Some(Val::Bool(
                    self.screen == Some(shown) && self.focus == button,
                ))
            }
            [Seg::Key(_), _, ..] | [Seg::Index(_), ..] | [] => None,
        }
    }

    fn value<'a>(&'a self, shown: Shown, path: &'a [Seg]) -> Val<'a> {
        if let Some(found) = self.focused(shown, path) {
            return found;
        }
        match shown {
            Shown::Select => self.select.value(path),
            Shown::Loading => self.loading.value(path),
            Shown::Settings => self.settings.value(path),
            Shown::Title => match path {
                [Seg::Key(name)] if name == "can_quit" => Val::Bool(self.can_quit),
                [Seg::Key(_), ..] | [Seg::Index(_), ..] | [] => Val::Unknown,
            },
            Shown::Pause
            | Shown::MatchEnd
            | Shown::Hud
            | Shown::Buy
            | Shown::Scoreboard
            | Shown::RoundEnd => Val::Unknown,
        }
    }
}

fn ability_slot(name: &str) -> Option<usize> {
    match name {
        "c" => Some(0),
        "q" => Some(1),
        "e" => Some(2),
        "x" => Some(3),
        _ => None,
    }
}

impl HudFacts {
    pub(super) fn map_items(&self, path: &[Seg]) -> Option<&[MapItem]> {
        let minimap = &self.minimap;
        match key(path) {
            Some(("minimap", [Seg::Key(name)])) => match name.as_str() {
                "me" => Some(&minimap.me),
                "players" => Some(&minimap.players),
                "device" => Some(&minimap.device),
                _ => None,
            },
            Some(_) | None => None,
        }
    }

    pub(super) fn value<'a>(&'a self, path: &'a [Seg]) -> Val<'a> {
        let Some((name, rest)) = key(path) else {
            return Val::Unknown;
        };
        match name {
            "round" => self.round.value(rest),
            "score" => match picked(&self.score, rest) {
                Ok((Some(score), [])) => whole(*score),
                Ok((None, [])) => Val::Missing,
                Ok((_, [_, ..])) | Err(_) => Val::Unknown,
            },
            "teams" => within(&self.teams, rest, |team, rest| team.value(rest)),
            "players" => match picked(&self.players, rest) {
                Ok((Some(team), rest)) => within(team, rest, |plate, rest| plate.value(rest)),
                Ok((None, _)) => Val::Missing,
                Err(unknown) => unknown,
            },
            "feed" => within(&self.feed, rest, |line, rest| line.value(rest)),
            "player" => self.player.value(rest),
            "abilities" => match key(rest) {
                Some((slot, rest)) => match ability_slot(slot) {
                    Some(index) => self.abilities[index].value(rest),
                    None => Val::Unknown,
                },
                None => Val::Unknown,
            },
            "weapon" => self.weapon.value(rest),
            "action" => self.action.value(rest),
            "buy" => self.buy.value(rest),
            "board" => match picked(&self.board, rest) {
                Ok((Some(team), rest)) => within(team, rest, |row, rest| row.value(rest)),
                Ok((None, _)) => Val::Missing,
                Err(unknown) => unknown,
            },
            "result" => self.result.value(rest),
            "minimap" => self.minimap.value(rest),
            "title" => self.menu.value(Shown::Title, rest),
            "pause" => self.menu.value(Shown::Pause, rest),
            "match_end" => self.menu.value(Shown::MatchEnd, rest),
            "select" => self.menu.value(Shown::Select, rest),
            "loading" => self.menu.value(Shown::Loading, rest),
            "settings" => self.menu.value(Shown::Settings, rest),
            "spectate" => leaf(rest, |name| match name {
                "active" => Val::Bool(self.spectate.active),
                "name" => Val::Text(&self.spectate.name),
                "icon" => Val::Text(&self.spectate.icon),
                "color" => Val::Text(&self.spectate.color),
                _ => Val::Unknown,
            }),
            "place" => leaf(rest, |name| match name {
                "name" => Val::Text(&self.place.name),
                "callout" => Val::Text(&self.place.callout),
                _ => Val::Unknown,
            }),
            "scope" if rest.is_empty() => whole(self.scope),
            "pickup" => self.pickup.value(rest),
            "cast" => self.cast.value(rest),
            _ => Val::Unknown,
        }
    }
}
