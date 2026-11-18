engine::brick!(arms);

mod fire;
mod pickup;
mod test;

use engine::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::actors::{Actor, Alive, Health, Human, SpeedFactor};
use crate::play::arms::{Arms, Drops, HandAction};
use crate::play::events::arms::{ArmsEvents, Weapon};
use crate::play::numbers::Numbers;
use crate::play::round::Round;
use crate::weapons::{ArmsNumbers, DamageLedger, Loadout, Slot, WeaponNumbers, actor_rng};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WeaponStats {
    pub slot: String,
    pub name: String,
    pub magazine: Option<u32>,
    pub reserve: Option<u32>,
    pub reloading: Option<f32>,
    pub spread_deg: f32,
    pub scope_level: u32,
    pub ready: bool,
    pub held: String,
    pub action: Option<String>,
    pub action_progress: f32,
    pub run_factor: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ActorArms {
    pub slot: u32,
    pub health: f32,
    pub armor: f32,
    pub alive: bool,
    pub magazine: Option<u32>,
    pub stance: String,
    pub speed_factor: f32,
    pub run_factor: f32,
    pub held: String,
    pub action: Option<String>,
    pub pickable: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DropStats {
    pub id: u32,
    pub weapon: String,
    pub position: [f32; 3],
    pub magazine: u32,
    pub reserve: u32,
}

stats! {
    #[section("arms")]
    pub struct ArmsStats {
        pub refusal: Option<String>,
        pub weapon: Option<WeaponStats>,
        pub recoil_pitch: f32,
        pub recoil_yaw: f32,
        pub shots_tick: u32,
        pub hits_tick: u32,
        pub heads_tick: u32,
        pub shots: u64,
        pub hits: u64,
        pub heads: u64,
        pub damage: f64,
        pub actors: Vec<ActorArms>,
        pub drops: Vec<DropStats>,
        pub pickups_tick: u32,
        pub drops_tick: u32,
    }
}

#[derive(Resource, Default)]
struct Setup {
    revision: Option<u64>,
    numbers: Option<ArmsNumbers>,
    refusal: Option<String>,
}

#[derive(Resource, Default)]
struct Book {
    ledger: DamageLedger,
    shots: u64,
    hits: u64,
    heads: u64,
    damage: f64,
}

pub struct ArmsBrick;

impl Brick for ArmsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<ArmsEvents>()
            .resource::<Setup>()
            .resource::<Book>()
            .stats::<ArmsStats>()
            .settings::<test::ArmsTest>()
            .resource::<Drops>()
            .systems(Schedule::FixedSimulation, setup)
            .systems(Schedule::FixedSimulation, arm)
            .systems(Schedule::FixedSimulation, fire::act)
            .systems(Schedule::FixedSimulation, pickup::pick)
            .systems(Schedule::FixedSimulation, test::door)
            .systems(Schedule::FixedSimulation, report);
    }
}

fn setup(numbers: Res<Numbers>, mut setup: ResMut<Setup>) {
    let Some(game) = numbers.get() else {
        return;
    };
    if setup.revision == Some(numbers.revision()) {
        return;
    }
    setup.revision = Some(numbers.revision());
    match serde_json::from_str::<ArmsNumbers>(&game.text) {
        Ok(parsed) => {
            setup.numbers = Some(parsed);
            setup.refusal = None;
        }
        Err(error) => {
            setup.numbers = None;
            setup.refusal = Some(error.to_string());
        }
    }
}

fn arm(
    time: Res<FixedTime>,
    setup: Res<Setup>,
    round: Option<Res<Round>>,
    actors: Query<(Entity, &Actor, Has<SpeedFactor>), Without<Arms>>,
    mut spawner: Spawner<ArmsBrick>,
) {
    let Some(numbers) = setup.numbers.as_ref() else {
        return;
    };
    let Some(tps) = ticks_per_second(&time) else {
        return;
    };
    let matched = round.as_deref().filter(|round| round.game.is_some());
    let (primary, seed) = match matched {
        Some(round) => (None, round.seed),
        None => (numbers.combat.loadout_primary, numbers.combat.seed),
    };
    for (entity, actor, has_speed) in &actors {
        let loadout = Loadout::new(&numbers.weapons, primary, actor_rng(seed, actor.slot), tps);
        let speed = loadout.speed_factor(&numbers.weapons);
        spawner.insert(entity, Arms::new(loadout, speed));
        if !has_speed {
            spawner.insert(entity, SpeedFactor(speed));
        }
    }
}

fn drop_weapon(
    loadout: &mut Loadout,
    weapons: &WeaponNumbers,
    tps: u32,
    at: glam::Vec3,
    drops: &mut Drops,
) {
    match loadout.drop_primary(weapons, tps) {
        Some(primary) => drops.push(at, &primary),
        None => drops.push(at, &loadout.sidearm),
    };
}

fn ticks_per_second(time: &FixedTime) -> Option<u32> {
    (time.delta_seconds > 0.0).then(|| (1.0 / time.delta_seconds).round() as u32)
}

fn action_name(action: HandAction) -> String {
    match action {
        HandAction::Equip => "equip",
        HandAction::Swing { heavy: false } => "swing_light",
        HandAction::Swing { heavy: true } => "swing_heavy",
        HandAction::Plant => "plant",
        HandAction::Defuse => "defuse",
    }
    .to_owned()
}

fn slot_name(slot: Slot) -> &'static str {
    match slot {
        Slot::Primary => "primary",
        Slot::Sidearm => "sidearm",
        Slot::Melee => "melee",
    }
}

type Reported<'a> = (
    &'a Actor,
    &'a Arms,
    &'a Health,
    &'a SpeedFactor,
    Has<Alive>,
    Has<Human>,
);

fn report(
    setup: Res<Setup>,
    events: Res<ArmsEvents>,
    book: Res<Book>,
    drops: Res<Drops>,
    actors: Query<Reported<'_>>,
    mut stats: ResMut<ArmsStats>,
) {
    let mut next = ArmsStats {
        refusal: setup.refusal.clone(),
        shots_tick: events.shots.len() as u32,
        hits_tick: events.hits.len() as u32,
        heads_tick: events.hits.iter().filter(|hit| hit.headshot).count() as u32,
        pickups_tick: events.pickups.len() as u32,
        drops_tick: events.drops.len() as u32,
        drops: drops
            .items
            .iter()
            .map(|item| DropStats {
                id: item.id,
                weapon: Weapon::Gun(item.weapon).key().to_owned(),
                position: item.position.to_array(),
                magazine: item.magazine,
                reserve: item.reserve,
            })
            .collect(),
        shots: book.shots,
        hits: book.hits,
        heads: book.heads,
        damage: book.damage,
        ..ArmsStats::default()
    };
    let mut rows: Vec<_> = actors.iter().collect();
    rows.sort_by_key(|(actor, ..)| actor.slot);
    for (actor, arms, health, speed, alive, human) in rows {
        let magazine = arms.loadout.active_gun().map(|gun| gun.magazine());
        next.actors.push(ActorArms {
            slot: actor.slot,
            health: health.health,
            armor: health.armour,
            alive,
            magazine,
            stance: format!("{:?}", arms.stance),
            speed_factor: speed.0,
            run_factor: arms.run_factor,
            held: arms.hand.held.key().to_owned(),
            action: arms.hand.action.map(action_name),
            pickable: arms.pickable,
        });
        if !human {
            continue;
        }
        let Some(numbers) = setup.numbers.as_ref() else {
            continue;
        };
        let view = arms.loadout.view(&numbers.weapons, arms.stance);
        let recoil = arms.loadout.recoil();
        next.recoil_pitch = recoil.pitch_radians;
        next.recoil_yaw = recoil.yaw_radians;
        next.weapon = Some(WeaponStats {
            slot: slot_name(view.slot).to_owned(),
            name: view.name.to_owned(),
            magazine: view.magazine,
            reserve: view.reserve,
            reloading: view.reload_progress,
            spread_deg: view.spread_deg,
            scope_level: view.scope_level as u32,
            ready: view.ready,
            held: arms.hand.held.key().to_owned(),
            action: arms.hand.action.map(action_name),
            action_progress: arms.hand.progress,
            run_factor: arms.run_factor,
        });
    }
    stats.set_if_neq(next);
}
