use engine::Node;
use engine::prelude::*;
use serde::Deserialize;

use crate::play::actors::{Actor, Human};
use crate::play::flow::SeatChoice;
use crate::rules::{Control, Team};

#[derive(Deserialize)]
struct RosterFile {
    #[serde(rename = "match")]
    game: RosterMatch,
}

#[derive(Deserialize)]
struct RosterMatch {
    roster: Vec<RosterSeat>,
}

#[derive(Deserialize)]
struct RosterSeat {
    team: Team,
    character: String,
    control: Control,
    level: String,
}

pub(super) fn default_roster(text: &str) -> Result<Vec<SeatChoice>, String> {
    let file: RosterFile = serde_json::from_str(text)
        .map_err(|error| format!("numbers.json match.roster: {error}"))?;
    Ok(file
        .game
        .roster
        .into_iter()
        .map(|seat| SeatChoice {
            team: seat.team,
            character: seat.character,
            control: seat.control,
            level: seat.level,
        })
        .collect())
}

pub(super) fn chosen(default: &[SeatChoice], character: &str, level: &str) -> Vec<SeatChoice> {
    let team = default
        .iter()
        .find(|seat| seat.control == Control::Human)
        .map_or(Team::First, |seat| seat.team);
    default
        .iter()
        .map(|seat| {
            let human = seat.team == team && seat.character == character;
            SeatChoice {
                team: seat.team,
                character: seat.character.clone(),
                control: if human { Control::Human } else { Control::Bot },
                level: if human {
                    seat.level.clone()
                } else {
                    level.to_owned()
                },
            }
        })
        .collect()
}

pub(super) type Seated<'a> = (Entity, &'a Actor, Has<Human>, Option<&'a Node>);

pub(super) fn seat_human(
    commands: &mut Commands<'_, '_>,
    roster: &[SeatChoice],
    actors: &Query<Seated<'_>>,
) -> bool {
    let Some(wanted) = roster.iter().find(|seat| seat.control == Control::Human) else {
        return false;
    };
    let mut moved = false;
    for (entity, actor, human, node) in actors {
        let chosen = actor.team == wanted.team && actor.character == wanted.character;
        let node = node.copied().unwrap_or_default();
        if chosen && !human {
            commands.entity(entity).insert((
                Human {},
                Node {
                    visible: false,
                    ..node
                },
            ));
            moved = true;
        } else if !chosen && human {
            commands.entity(entity).remove::<Human>().insert(Node {
                visible: true,
                ..node
            });
            moved = true;
        }
    }
    moved
}
