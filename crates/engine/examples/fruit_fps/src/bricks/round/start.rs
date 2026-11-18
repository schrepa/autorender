use engine::prelude::*;
use serde::Deserialize;

use crate::play::actors::Actor;
use crate::play::round::Seat;
use crate::rules::{Control, Slot, Team};

#[derive(Clone, Debug, Deserialize)]
struct StartFile {
    #[serde(rename = "match")]
    start: StartNumbers,
    player: Footing,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub(super) struct StartNumbers {
    pub roster: Vec<SeatNumbers>,
    pub next_match_seconds: f64,
    pub device_reach_m: f32,
    pub device_set_down: SetDown,
    #[serde(skip)]
    pub footing: Footing,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SetDown {
    pub right_m: f32,
    pub ahead_m: f32,
    pub lift_m: f32,
    pub clear_m: f32,
    pub pull_step_m: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
pub(super) struct Footing {
    pub step_height_m: f32,
    pub slope_limit_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub(super) struct SeatNumbers {
    pub team: Team,
    pub character: String,
    pub control: Control,
    pub level: String,
}

pub(super) fn start_numbers(text: &str) -> Result<StartNumbers, String> {
    serde_json::from_str::<StartFile>(text)
        .map(|file| StartNumbers {
            footing: file.player,
            ..file.start
        })
        .map_err(|error| format!("numbers.json match: {error}"))
}

pub(super) fn seats<'a>(
    start: &StartNumbers,
    actors: impl Iterator<Item = (Entity, &'a Actor, bool)>,
) -> Result<Vec<Seat>, String> {
    let mut seats: Vec<Seat> = start
        .roster
        .iter()
        .map(|seat| Seat {
            entity: None,
            slot: Slot {
                team: seat.team,
                character: seat.character.clone(),
                control: seat.control,
            },
            level: seat.level.clone(),
        })
        .collect();
    let mut found: Vec<(u32, Entity, &Actor, bool)> = actors
        .map(|(entity, actor, human)| (actor.slot, entity, actor, human))
        .collect();
    found.sort_by_key(|(slot, entity, _, _)| (*slot, *entity));
    for (slot, entity, actor, human) in found {
        let Some(seat) = seats.get_mut(slot as usize) else {
            return Err(format!(
                "actor slot {slot} is past the roster's {} seats",
                start.roster.len()
            ));
        };
        if seat.entity.is_some() {
            return Err(format!("two actors hold slot {slot}"));
        }
        if seat.slot.team != actor.team || seat.slot.character != actor.character {
            return Err(format!(
                "actor slot {slot} is {:?} {} but the roster says {:?} {}",
                actor.team, actor.character, seat.slot.team, seat.slot.character
            ));
        }
        if human != (seat.slot.control == Control::Human) {
            return Err(format!(
                "actor slot {slot}: the human marker and the roster's control differ"
            ));
        }
        seat.entity = Some(entity);
    }
    Ok(seats)
}
