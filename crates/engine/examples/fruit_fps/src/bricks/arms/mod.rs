engine::brick!(arms);

mod fire;
mod pickup;
mod test;

use engine::prelude::*;
use engine_core::Stats;
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::actors::{Actor, Alive, Health, Human, SpeedFactor};
use crate::play::arms::{Arms, Drops, HandAction};
use crate::play::events::arms::{ArmsEvents, PickupEvent, Weapon};
use crate::play::numbers::Numbers;
use crate::play::round::Round;
use crate::play::world::Arena;
use crate::weapons::{ArmsNumbers, DamageLedger, KeyDrop, Loadout, Slot, actor_rng, forward};

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
    pub swing_step: u32,
    pub zoom: f32,
    pub scope_raised: bool,
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
    pub inspecting: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KnifeStats {
    pub slot: u32,
    pub heavy: bool,
    pub step: u32,
    pub swing_tick: u64,
    pub strike_due: u64,
    pub strike_tick: Option<u64>,
    pub victim: Option<u32>,
    pub killed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PickupStats {
    pub slot: u32,
    pub weapon: String,
    pub left: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DropStats {
    pub id: u32,
    pub by: Option<u32>,
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
        pub picked_up: u64,
        pub dropped: u64,
        pub last_pickup: Option<PickupStats>,
        pub unknown_surfaces: Vec<String>,
        pub knife: Option<KnifeStats>,
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
    picked_up: u64,
    dropped: u64,
    last_pickup: Option<PickupEvent>,
    unknown_surfaces: Vec<engine_solids::Kind>,
    knife: Option<KnifeStats>,
}

pub struct ArmsBrick;

impl Brick for ArmsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<ArmsEvents>()
            .resource::<Setup>()
            .resource::<Book>()
            .resource::<Drops>()
            .systems(Schedule::FixedSimulation, setup)
            .systems(Schedule::FixedSimulation, arm)
            .systems(Schedule::FixedSimulation, fire::act)
            .systems(Schedule::FixedSimulation, pickup::pick)
            .dev(|dev| {
                dev.settings::<test::ArmsTest>()
                    .stats::<ArmsStats>()
                    .systems(Schedule::FixedSimulation, test::door)
                    .systems(Schedule::FixedSimulation, report);
            });
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
    match ArmsNumbers::deserialize(&game.whole) {
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
    let tps = time.rate;
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

fn drop_weapon(loadout: &mut Loadout, at: glam::Vec3, drops: &mut Drops) {
    if let Some(gun) = loadout.drop_on_death() {
        drops.push(at, &gun);
    }
}

fn thrown_to(arena: &Arena, throw: &KeyDrop, feet: Vec3, yaw_radians: f32) -> Vec3 {
    let ahead = forward(yaw_radians, 0.0);
    let lift = feet + Vec3::Y * throw.lift_m;
    let travel = arena
        .solids()
        .sphere_sweep(lift, throw.radius_m, ahead * throw.throw_m)
        .map_or(throw.throw_m, |hit| hit.distance);
    let steps = throw.back_steps.max(1);
    (0..=steps)
        .map(|step| lift + ahead * (travel * (steps - step) as f32 / steps as f32))
        .find_map(|from| arena.ground_under(from, throw.lift_m + throw.floor_drop_m))
        .map_or(feet, |hit| hit.point)
}

fn action_name(action: HandAction) -> String {
    match action {
        HandAction::Equip => "equip",
        HandAction::Swing { heavy: false, .. } => "swing_light",
        HandAction::Swing { heavy: true, .. } => "swing_heavy",
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

fn knife_watch(book: &mut Book, events: &ArmsEvents) {
    if let Some(swing) = events.swings.last() {
        book.knife = Some(KnifeStats {
            slot: swing.slot,
            heavy: swing.heavy,
            step: u32::from(swing.step),
            swing_tick: events.tick,
            strike_due: swing.strike_tick,
            ..KnifeStats::default()
        });
    }
    let Some(knife) = book.knife.as_mut() else {
        return;
    };
    let struck = (events.shots.iter())
        .find(|shot| shot.shooter == knife.slot && shot.weapon == Weapon::Melee);
    if let Some(shot) = struck {
        knife.strike_tick = Some(events.tick);
        knife.victim = match shot.target {
            crate::play::events::arms::Target::Actor { slot, .. } => Some(slot),
            crate::play::events::arms::Target::Nothing
            | crate::play::events::arms::Target::World => None,
        };
    }
    knife.killed |= (events.hits.iter())
        .any(|hit| hit.shooter == knife.slot && hit.weapon == Weapon::Melee && hit.killed);
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
    mut book: ResMut<Book>,
    drops: Res<Drops>,
    arena: Res<Arena>,
    actors: Query<Reported<'_>>,
    mut stats: Stats<ArmsStats>,
) {
    book.picked_up += events.pickups.len() as u64;
    book.dropped += events.drops.len() as u64;
    if let Some(last) = events.pickups.last() {
        book.last_pickup = Some(*last);
    }
    knife_watch(&mut book, &events);
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
                by: item.by,
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
        unknown_surfaces: (book.unknown_surfaces.iter())
            .map(|kind| {
                arena
                    .kind_name(*kind)
                    .map_or_else(|| format!("{:08x}", kind.0), str::to_owned)
            })
            .collect(),
        picked_up: book.picked_up,
        dropped: book.dropped,
        last_pickup: book.last_pickup.map(|pickup| PickupStats {
            slot: pickup.slot,
            weapon: Weapon::Gun(pickup.weapon).key().to_owned(),
            left: pickup.left.map(|gun| Weapon::Gun(gun).key().to_owned()),
        }),
        knife: book.knife,
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
            inspecting: (arms.loadout.inspecting()).map(|gun| Weapon::Gun(gun).key().to_owned()),
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
            spread_deg: arms.live_spread_deg,
            scope_level: view.scope_level as u32,
            ready: view.ready,
            held: arms.hand.held.key().to_owned(),
            action: arms.hand.action.map(action_name),
            action_progress: arms.hand.progress,
            swing_step: match arms.hand.action {
                Some(HandAction::Swing { step, .. }) => u32::from(step),
                _ => 0,
            },
            zoom: arms.zoom,
            scope_raised: (arms.loadout.active_gun())
                .is_some_and(crate::weapons::Gun::scope_raised),
            run_factor: arms.run_factor,
        });
    }
    stats.show(|| next);
}
