use engine::prelude::*;
use glam::Vec3;

use super::pickup::{self, Ground};
use super::senses::{self, Events, Noise, ReadTicks, Sight, Who};
use super::shop::{Counter, Shopping};
use super::world::{BotWorld, Thrower};
use super::{Book, BotsTest, LevelChoice, Loaded, Reach, level_of};
use crate::abilities::{self as powers_maths, ActorId, DrawKind, DrawShape, OrbState};
use crate::bots::{
    self, Brain, BuySense, Call, Gait as BotGait, Heard, Intent, Level, MatchSense, Mate, Member,
    ObjectKind, ObjectSense, OrbSense, Places, Purchase, Role, Seen, Sense, Smoke, SoundKind,
    Squad, SquadSense, Tactics, WeaponChoice,
};
use crate::play::actors::{Actor, Alive, Body, Equip, Gait, Health, Human, Wish};
use crate::play::arms::{Arms, Drops};
use crate::play::bots::{BotMind, BotMinds, TeamSight, WishSummary};
use crate::play::events::arms::ArmsEvents;
use crate::play::events::body::BodyEvents;
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::flow::Flow;
use crate::play::paths::{HazardKind, Hazards, Paths};
use crate::play::powers::Powers;
use crate::play::round::Round;
use crate::play::view::ViewOffsets;
use crate::play::world::Arena;
use crate::rules::{AbilitySlot, BuyRequest, Phase, Team, View};
use crate::weapons::forward;

const BRAIN_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
const SQUAD_MIX: u64 = 0xC2B2_AE3D_27D4_EB4F;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    matches: u32,
    seed: u64,
    numbers: Option<u64>,
    paths: Option<u64>,
    level: LevelChoice,
}

#[derive(Resource, Default)]
pub(super) struct Minds {
    key: Option<Key>,
    brains: Vec<Option<Brain>>,
    squads: [Option<Squad>; 2],
    calls: [Vec<Call>; 2],
    next_calls: [Vec<Call>; 2],
    fired: Vec<bool>,
    picking: Vec<bool>,
    drops: u32,
    gifts: u32,
    read: ReadTicks,
    noises: Vec<Noise>,
    whos: Vec<Who>,
    reaches: Vec<[Reach; 4]>,
    intents: Vec<(u32, Intent)>,
    wishes: Vec<(u32, WishSummary)>,
    tactics: Option<Tactics>,
    sight: Vec<Option<Seen>>,
    placed: Vec<(usize, ObjectSense)>,
    seen: Vec<Seen>,
    heard: Vec<Heard>,
    mates: Vec<Mate>,
    smokes: Vec<Smoke>,
    hazards: [Vec<bots::Hazard>; 2],
    orbs: Vec<OrbSense>,
    reserved: Vec<(u32, u32)>,
    objects: Vec<ObjectSense>,
    members: [Vec<Member>; 2],
    shopping: Shopping,
}

#[derive(SystemParam)]
pub(super) struct Facts<'w> {
    time: Res<'w, FixedTime>,
    test: DevSetting<'w, BotsTest>,
    book: Res<'w, Book>,
    round: Option<Res<'w, Round>>,
    paths: Res<'w, Paths>,
    arena: Res<'w, Arena>,
    powers: Option<Res<'w, Powers>>,
    hazards: Option<Res<'w, Hazards>>,
    flow: Option<Res<'w, Flow>>,
    drops: Option<Res<'w, Drops>>,
}

#[derive(SystemParam)]
pub(super) struct Said<'w> {
    body: Option<Res<'w, BodyEvents>>,
    arms: Option<Res<'w, ArmsEvents>>,
    powers: Option<Res<'w, PowerEvents>>,
    round: Option<Res<'w, RoundEvents>>,
}

type Row<'a> = (
    &'a Actor,
    &'a Body,
    &'a Health,
    &'a mut Wish,
    Option<&'a Arms>,
    Option<&'a ViewOffsets>,
    Has<Alive>,
    Has<Human>,
);

type Seat<'a> = (&'a Actor, Has<Body>, Has<Health>, Has<Wish>, Has<Human>);

struct Ctx<'a> {
    loaded: &'a Loaded,
    places: &'a Places,
    round: &'a Round,
    view: &'a View<'a>,
    arena: &'a Arena,
    paths: &'a Paths,
    powers: Option<&'a Powers>,
    drops: Option<&'a Drops>,
    tick: u64,
}

fn rest(wish: &mut Wish) {
    *wish = Wish {
        yaw_radians: wish.yaw_radians,
        pitch_radians: wish.pitch_radians,
        ..Wish::default()
    };
}

fn rest_all(actors: &mut Query<Row<'_>>) {
    for (_, _, _, mut wish, _, _, _, human) in actors.iter_mut() {
        if !human {
            rest(&mut wish);
        }
    }
}

fn snapshot(
    actors: &Query<Row<'_>>,
    loaded: &Loaded,
    whos: &mut Vec<Who>,
    reaches: &mut Vec<[Reach; 4]>,
) {
    whos.clear();
    for (actor, body, health, wish, arms, offsets, alive, human) in actors.iter() {
        let recoil = offsets.map_or((0.0, 0.0), |held| {
            (held.pitch_radians(), held.yaw_radians())
        });
        whos.push(Who {
            slot: actor.slot,
            team: actor.team,
            alive,
            human,
            feet: body.feet,
            velocity: body.velocity,
            eye: body.eye,
            crouched: body.crouched,
            grounded: body.grounded,
            pickable: arms.and_then(|arms| arms.pickable),
            yaw: wish.yaw_radians,
            pitch: wish.pitch_radians,
            recoil_yaw: recoil.1,
            recoil_pitch: recoil.0,
            health: health.health,
            armour: health.armour,
            gun: senses::gun_sense(arms, &loaded.weapons, body.motion, recoil),
            character: loaded.book.character(actor.character.as_str()),
            interact: wish.interact_held,
        });
    }
    whos.sort_by_key(|who| who.slot);
    reaches.clear();
    for who in whos.iter() {
        let character = actors
            .iter()
            .find(|row| row.0.slot == who.slot)
            .map(|row| row.0.character.as_str());
        reaches.push(character.map_or([Reach::default(); 4], |key| loaded.reach(key)));
    }
}

fn reset(state: &mut Minds, ctx: &Ctx<'_>) {
    let seed = ctx.round.seed;
    let slots = state
        .whos
        .iter()
        .map(|who| who.slot as usize + 1)
        .max()
        .unwrap_or(0);
    state.brains = vec![None; slots];
    state.tactics = None;
    state.squads = [Team::First, Team::Second].map(|team| {
        let count = state.whos.iter().filter(|who| who.team == team).count();
        let mix = (team.index() as u64 + 1).wrapping_mul(SQUAD_MIX);
        Some(Squad::new(seed ^ mix, count, team == Team::Second))
    });
    state.calls = [Vec::new(), Vec::new()];
    state.next_calls = [Vec::new(), Vec::new()];
    state.fired = vec![false; slots];
}

fn ensure_brains(state: &mut Minds, round: &Round, choice: LevelChoice) {
    let slots = state
        .whos
        .iter()
        .map(|who| who.slot as usize + 1)
        .max()
        .unwrap_or(0);
    if state.brains.len() < slots {
        state.brains.resize(slots, None);
    }
    if state.fired.len() < slots {
        state.fired.resize(slots, false);
    }
    if state.picking.len() < slots {
        state.picking.resize(slots, false);
    }
    for who in state.whos.iter().filter(|who| !who.human) {
        let held = &mut state.brains[who.slot as usize];
        if held.is_none() {
            let level_name = round
                .seats
                .get(who.slot as usize)
                .map_or("", |seat| seat.level.as_str());
            let mix = (u64::from(who.slot) + 1).wrapping_mul(BRAIN_MIX);
            *held = Some(Brain::new(round.seed ^ mix, level_of(level_name, choice)));
        }
    }
}

fn member(who: &Who, view: &View<'_>, carrier: Option<u32>, tools: u16) -> Member {
    let player = view.players.get(who.slot as usize);
    let role = player
        .and_then(|player| view.catalog.characters.get(player.character))
        .and_then(|character| Role::from_data_name(&character.role))
        .unwrap_or(Role::Support);
    let gun_cost = player
        .and_then(|player| player.primary)
        .and_then(|weapon| view.catalog.weapon(weapon))
        .and_then(|weapon| weapon.cost)
        .unwrap_or(0);
    Member {
        id: who.slot,
        bot: !who.human,
        alive: who.alive,
        position: who.feet,
        health: who.health,
        role,
        carrying_device: carrier == Some(who.slot),
        gun_cost,
        tools,
    }
}

fn plan(state: &mut Minds, ctx: &Ctx<'_>, games: &[MatchSense; 2]) {
    let carrier = senses::carrier(ctx.round);
    let tactics = state.tactics.get_or_insert_with(|| {
        let world = BotWorld {
            arena: ctx.arena,
            paths: ctx.paths,
            team: 0,
            places: ctx.places,
            thrower: None,
        };
        bots::build_tactics(&world, &ctx.loaded.bots, &mut Vec::new())
    });
    for team in [Team::First, Team::Second] {
        let index = team.index();
        let members = &mut state.members[index];
        members.clear();
        for (who, reach) in state.whos.iter().zip(&state.reaches) {
            if who.team != team {
                continue;
            }
            let tools = bots::tools_of(&senses::abilities_of(ctx.powers, who.slot, *reach));
            members.push(member(who, ctx.view, carrier, tools));
        }
        let enemies_alive = state
            .whos
            .iter()
            .filter(|who| who.team != team && who.alive)
            .count() as u32;
        let sense = SquadSense {
            game: games[index],
            members,
            calls: &state.calls[index],
            enemies_alive,
        };
        let world = BotWorld {
            arena: ctx.arena,
            paths: ctx.paths,
            team: index,
            places: ctx.places,
            thrower: None,
        };
        if let Some(squad) = state.squads[index].as_mut() {
            squad.plan(&sense, &world, tactics, &ctx.loaded.bots, ctx.tick);
        }
    }
}

fn gather(state: &mut Minds, ctx: &Ctx<'_>, at: usize) {
    let who = &state.whos[at];
    let count = state.whos.len();
    state.seen.clear();
    state
        .seen
        .extend(state.sight[at * count..(at + 1) * count].iter().flatten());
    state.heard.clear();
    senses::heard(&state.noises, &state.whos, who, &mut state.heard);
    senses::revealed(ctx.powers, &state.whos, who, &mut state.heard);
    state.mates.clear();
    for other in state
        .whos
        .iter()
        .filter(|other| other.team == who.team && other.slot != who.slot)
    {
        state.mates.push(Mate {
            id: other.slot,
            position: other.feet,
            health: other.health,
            alive: other.alive,
            human: other.human,
            yaw: other.yaw,
        });
    }
}

fn think_one(
    state: &mut Minds,
    ctx: &Ctx<'_>,
    games: &[MatchSense; 2],
    at: usize,
) -> Option<Intent> {
    let who = state.whos[at];
    let reach = state.reaches[at];
    let index = who.team.index();
    let me = state.members[index]
        .iter()
        .position(|member| member.id == who.slot)?;
    gather(state, ctx, at);
    let order = state.squads[index].as_ref()?.order(me);
    let player = ctx.view.players.get(who.slot as usize);
    let level = state
        .brains
        .get(who.slot as usize)
        .and_then(Option::as_ref)
        .map(|brain| brain.debug().level);
    fill_objects(&mut state.objects, &state.placed, ctx, &who, level);
    let sense = Sense {
        me,
        alive: who.alive,
        position: who.feet,
        velocity: who.velocity,
        yaw: who.yaw,
        pitch: who.pitch,
        crouched: who.crouched,
        grounded: who.grounded,
        health: who.health,
        armor: who.armour,
        gun: who.gun,
        abilities: senses::abilities_of(ctx.powers, who.slot, reach),
        credits: player.map_or(0, |player| player.credits),
        carrying_device: senses::carrier(ctx.round) == Some(who.slot),
        blind: ctx.powers.map_or(0.0, |powers| powers.blindness(who.slot)),
        enemies: &state.seen,
        sounds: &state.heard,
        mates: &state.mates,
        calls: &state.calls[index],
        smokes: &state.smokes,
        hazards: &state.hazards[index],
        revealed: ctx.powers.is_some_and(|powers| {
            let enemy = if who.team == Team::First {
                Team::Second
            } else {
                Team::First
            };
            powers.revealed(who.slot, enemy)
        }),
        orbs: &state.orbs,
        objects: &state.objects,
        game: games[index],
        order,
        buy: BuySense {
            device_asked: asked_for_device(state, ctx, &who),
        },
    };
    let loaded = ctx.loaded;
    let thrower = who
        .character
        .and_then(|character| loaded.specs(character))
        .zip(ctx.powers)
        .map(|(specs, powers)| Thrower {
            actor: powers_maths::Actor {
                id: ActorId(who.slot),
                team: index,
                feet: who.feet,
                eye: who.eye,
                view: forward(who.yaw, who.pitch),
                radius: loaded.radius_m,
                height: if who.crouched {
                    loaded.crouch_height_m
                } else {
                    loaded.height_m
                },
            },
            specs,
            powers,
        });
    let world = BotWorld {
        arena: ctx.arena,
        paths: ctx.paths,
        team: index,
        places: ctx.places,
        thrower,
    };
    let quiet = state.seen.is_empty();
    let brain = state.brains.get_mut(who.slot as usize)?.as_mut()?;
    let mut intent = bots::think(brain, &sense, &world, &ctx.loaded.bots, ctx.tick);
    let ground = Ground {
        drops: ctx.drops,
        arena: ctx.arena,
        loaded: ctx.loaded,
        reserved: &state.reserved,
    };
    let slot = who.slot as usize;
    let before = state.picking.get(slot).copied().unwrap_or(false);
    let press = quiet && pickup::consider(&ground, &who, brain, before, &mut intent);
    if let Some(picking) = state.picking.get_mut(slot) {
        *picking = press;
    }
    if let Some(call) = intent.call {
        state.next_calls[index].push(call);
    }
    Some(intent)
}

fn read_powers(state: &mut Minds, ctx: &Ctx<'_>) {
    if let Some(powers) = ctx.powers {
        state.noises.retain(|noise| {
            noise.kind != SoundKind::Step || powers.modifiers(noise.slot).speed >= 1.0
        });
    }
    state.orbs.clear();
    if let Some(powers) = ctx.powers {
        state.orbs.extend(powers.orbs().iter().map(|orb| OrbSense {
            at: orb.at,
            waiting: orb.state == OrbState::Waiting,
        }));
    }
}

fn read_hazards(state: &mut Minds, hazards: Option<&Hazards>) {
    state.smokes.clear();
    state.hazards.iter_mut().for_each(Vec::clear);
    let Some(hazards) = hazards else {
        return;
    };
    state
        .smokes
        .extend(hazards.smokes.iter().map(|smoke| Smoke {
            center: smoke.centre,
            radius: smoke.radius,
        }));
    for (out, team) in state.hazards.iter_mut().zip(&hazards.teams) {
        out.extend(
            team.iter()
                .filter(|hazard| {
                    matches!(
                        hazard.kind,
                        HazardKind::Damage | HazardKind::Strike | HazardKind::Slow
                    )
                })
                .map(|hazard| bots::Hazard {
                    center: hazard.centre,
                    radius: hazard.radius,
                }),
        );
    }
}

fn reserve_gifts(state: &mut Minds, ctx: &Ctx<'_>) {
    state.reserved.clear();
    let Some(drops) = ctx.drops else {
        return;
    };
    let players = ctx.view.players;
    for item in &drops.items {
        let Some(by) = item.by else {
            continue;
        };
        let Some(giver) = state.whos.iter().find(|w| w.slot == by && w.human) else {
            continue;
        };
        let receiver = state
            .whos
            .iter()
            .filter(|w| w.alive && w.team == giver.team && w.slot != by && !w.gun.has_primary)
            .filter_map(|w| players.get(w.slot as usize).map(|p| (p.credits, w.slot)))
            .min()
            .map(|(_, slot)| slot);
        if let Some(receiver) = receiver {
            state.reserved.push((item.id, receiver));
        }
    }
}

fn sight_table(state: &mut Minds, ctx: &Ctx<'_>) {
    let sight = Sight {
        arena: ctx.arena,
        powers: ctx.powers,
        loaded: ctx.loaded,
    };
    let carrier = senses::carrier(ctx.round);
    let count = state.whos.len();
    state.sight.clear();
    state.sight.resize(count * count, None);
    for (at, viewer) in state.whos.iter().enumerate().filter(|(_, w)| w.alive) {
        for (to, target) in state.whos.iter().enumerate() {
            if target.alive && target.team != viewer.team {
                state.sight[at * count + to] = senses::seen(&sight, viewer, target, carrier);
            }
        }
    }
}

fn team_sight(state: &Minds, tick: u64, sight_out: &mut TeamSight) {
    let count = state.whos.len();
    for team in 0..2 {
        sight_out.seen[team].clear();
        for (at, viewer) in state.whos.iter().enumerate() {
            if viewer.team.index() != team {
                continue;
            }
            for (to, target) in state.whos.iter().enumerate() {
                if state.sight[at * count + to].is_some() {
                    sight_out.seen[team].push((target.slot, viewer.slot));
                    let last = &mut sight_out.last[team];
                    match last.iter_mut().find(|l| l.0 == target.slot) {
                        Some(entry) => *entry = (target.slot, tick, target.feet),
                        None => last.push((target.slot, tick, target.feet)),
                    }
                }
            }
        }
    }
}

fn read_objects(out: &mut Vec<(usize, ObjectSense)>, powers: Option<&Powers>) {
    out.clear();
    let Some(powers) = powers else {
        return;
    };
    for drawn in powers.drawables() {
        let kind = match drawn.kind {
            DrawKind::Trap => ObjectKind::Trap,
            DrawKind::Wire => ObjectKind::Wire,
            DrawKind::Sensor => ObjectKind::Sensor,
            DrawKind::Wall => ObjectKind::Wall,
            DrawKind::Projectile
            | DrawKind::Smoke
            | DrawKind::Zone
            | DrawKind::Curtain
            | DrawKind::StrikeMark => continue,
        };
        let at = match drawn.shape {
            DrawShape::Sphere { centre, .. } | DrawShape::Box { centre, .. } => centre,
            DrawShape::Segment { a, .. } => a,
        };
        let object = ObjectSense {
            id: drawn.effect.0,
            kind,
            at,
            enemy: false,
        };
        out.push((drawn.team, object));
    }
}

fn fill_objects(
    out: &mut Vec<ObjectSense>,
    placed: &[(usize, ObjectSense)],
    ctx: &Ctx<'_>,
    who: &Who,
    level: Option<Level>,
) {
    out.clear();
    let Some(level) = level else {
        return;
    };
    let play = &ctx.loaded.bots.play;
    let notice = ctx.loaded.bots.level(level).notice_wire;
    let team = who.team.index();
    for (owner, object) in placed {
        let enemy = *owner != team;
        if (object.at - who.eye).length() > play.object_shoot_m
            || !ctx.arena.sees(who.eye, object.at)
        {
            continue;
        }
        if object.kind == ObjectKind::Wire && enemy {
            let mix = u64::from(who.slot)
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(u64::from(object.id).wrapping_mul(0x85EB_CA6B));
            if (mix % 1000) as f32 / 1000.0 >= notice {
                continue;
            }
        }
        out.push(ObjectSense { enemy, ..*object });
    }
}

fn asked_for_device(state: &Minds, ctx: &Ctx<'_>, carrier: &Who) -> bool {
    let play = &ctx.loaded.bots.play;
    if ctx.view.phase != Phase::Buy || senses::carrier(ctx.round) != Some(carrier.slot) {
        return false;
    }
    let chest = (carrier.feet + carrier.eye) * 0.5;
    let limit = engine::math::cos(play.ask_device_deg.to_radians());
    state.whos.iter().any(|human| {
        let to = chest - human.eye;
        let flat = Vec3::new(to.x, 0.0, to.z).length();
        human.human
            && human.alive
            && human.interact
            && human.team == carrier.team
            && flat <= play.ask_device_m
            && forward(human.yaw, human.pitch).dot(to.normalize_or_zero()) >= limit
    })
}

fn equip_of(choice: WeaponChoice) -> Equip {
    match choice {
        WeaponChoice::Primary => Equip::Primary,
        WeaponChoice::Sidearm => Equip::Sidearm,
        WeaponChoice::Melee => Equip::Melee,
    }
}

fn cast_of(slot: usize) -> Option<AbilitySlot> {
    AbilitySlot::ALL.get(slot).copied()
}

fn apply(wish: &mut Wish, intent: &Intent, fired_before: bool, buys: Vec<BuyRequest>) {
    let still = intent.gait == BotGait::Still;
    let flat = Vec3::new(intent.wish.x, 0.0, intent.wish.z);
    *wish = Wish {
        direction: if still {
            Vec3::ZERO
        } else {
            flat.clamp_length_max(1.0)
        },
        gait: if intent.gait == BotGait::Walk {
            Gait::Walk
        } else {
            Gait::Run
        },
        crouch: intent.crouch,
        jump: intent.jump,
        yaw_radians: intent.yaw,
        pitch_radians: intent.pitch,
        fire_held: intent.fire,
        fire_pressed: intent.fire && !fired_before,
        reload: intent.reload,
        inspect: false,
        drop: intent.drop_device,
        cycle: 0,
        ready: false,
        scope: intent.scope,
        equip: if intent.drop_device {
            Some(Equip::Device)
        } else {
            intent.switch.map(equip_of)
        },
        interact_held: intent.plant || intent.defuse || intent.interact,
        cast: intent.ability.and_then(|ability| cast_of(ability.slot)),
        buys,
        buy_menu: false,
        scoreboard: false,
    };
}

fn summary(wish: &Wish) -> WishSummary {
    WishSummary {
        direction: wish.direction,
        walk: wish.gait == Gait::Walk,
        fire: wish.fire_held,
        yaw_radians: wish.yaw_radians,
        pitch_radians: wish.pitch_radians,
    }
}

fn write(state: &mut Minds, actors: &mut Query<Row<'_>>) {
    state.wishes.clear();
    for (actor, _, _, mut wish, _, _, _, human) in actors.iter_mut() {
        if human {
            continue;
        }
        let slot = actor.slot as usize;
        match state.intents.iter().find(|(at, _)| *at == actor.slot) {
            Some((_, intent)) => {
                let before = state.fired.get(slot).copied().unwrap_or(false);
                let buys = state.shopping.requests(slot, intent.buy == Purchase::Own);
                state.gifts += buys
                    .iter()
                    .filter(|buy| matches!(buy, BuyRequest::BuyFor(..)))
                    .count() as u32;
                apply(&mut wish, intent, before, buys);
                if let Some(fired) = state.fired.get_mut(slot) {
                    *fired = intent.fire;
                }
                if state.picking.get(slot).copied().unwrap_or(false) {
                    wish.interact_held = true;
                }
                state.drops += u32::from(intent.drop_device);
                state.wishes.push((actor.slot, summary(&wish)));
            }
            None => rest(&mut wish),
        }
    }
}

fn record(state: &Minds, minds: &mut BotMinds) {
    minds.bots.clear();
    for (slot, intent) in &state.intents {
        if let Some(brain) = state.brains.get(*slot as usize).and_then(Option::as_ref) {
            minds.bots.push(BotMind {
                slot: *slot,
                mind: brain.debug(),
                throws: brain.throws(),
                line_up: brain.last_throw(),
                goal: intent.goal,
                wish: state
                    .wishes
                    .iter()
                    .find(|(at, _)| at == slot)
                    .map_or(WishSummary::default(), |(_, wish)| *wish),
            });
        }
    }
    minds.squads = [0, 1].map(|team| state.squads[team].as_ref().map(Squad::debug));
    minds.drops = state.drops;
    minds.gifts = state.gifts;
    minds.buy_calls = state.shopping.calls.map(|call| call.map(|c| c.key()));
}

fn gate(facts: &Facts<'_>, has_match: bool) -> Option<String> {
    if let Some(problem) = facts.book.problem() {
        return Some(problem);
    }
    if facts.book.loaded.is_none() {
        return Some("numbers.json not read yet".to_owned());
    }
    if facts.book.places.is_none() {
        return Some(format!(
            "no map places: paths ready {}, layout {}",
            facts.paths.ready(),
            facts.paths.layout().is_some()
        ));
    }
    if !has_match {
        return Some("no match running (Round.game is empty)".to_owned());
    }
    if let Some(flow) = facts.flow.as_deref().filter(|flow| !flow.stage.moving()) {
        return Some(format!("flow stage {}: bots rest", flow.stage.key()));
    }
    let frozen = facts.test.get().is_some_and(|test| test.frozen);
    frozen.then(|| "frozen by settings.bots_test".to_owned())
}

fn unthought(seats: &Query<Seat<'_>>, state: &Minds) -> Option<String> {
    let mut rows: Vec<(u32, bool, bool)> = seats
        .iter()
        .filter(|(_, _, _, _, human)| !human)
        .map(|(actor, body, health, wish, _)| {
            let brain = state
                .brains
                .get(actor.slot as usize)
                .is_some_and(Option::is_some);
            (actor.slot, body && health && wish, brain)
        })
        .collect();
    rows.sort_by_key(|row| row.0);
    let reasons: Vec<String> = rows
        .into_iter()
        .filter(|(slot, _, _)| !state.intents.iter().any(|(at, _)| at == slot))
        .map(|(slot, set_up, brain)| {
            let reason = if !set_up {
                "no Body, Health or Wish yet"
            } else if !brain {
                "no brain"
            } else {
                "not a member of its team's squad"
            };
            format!("slot {slot}: {reason}")
        })
        .collect();
    (!reasons.is_empty()).then(|| reasons.join("; "))
}

pub(super) fn think(
    facts: Facts<'_>,
    said: Said<'_>,
    mut actors: Query<Row<'_>>,
    seats: Query<Seat<'_>>,
    mut state: ResMut<Minds>,
    mut minds: ResMut<BotMinds>,
) {
    let tick = facts.time.tick;
    let override_level = facts
        .test
        .get()
        .map_or(LevelChoice::Seat, |test| test.level);
    let level = match override_level {
        LevelChoice::Seat => match facts.flow.as_deref().map(|flow| flow.level.as_str()) {
            Some("easy") => LevelChoice::Easy,
            Some("normal") => LevelChoice::Normal,
            Some("hard") => LevelChoice::Hard,
            _ => LevelChoice::Seat,
        },
        LevelChoice::Easy => LevelChoice::Easy,
        LevelChoice::Normal => LevelChoice::Normal,
        LevelChoice::Hard => LevelChoice::Hard,
    };
    minds.tick = tick;
    minds.bots.clear();
    let round = facts.round.as_deref();
    let view = round.and_then(Round::view);
    minds.problem = gate(&facts, view.is_some());
    let (None, Some(loaded), Some(places), Some(round), Some(view)) = (
        &minds.problem,
        facts.book.loaded.as_deref(),
        facts.book.places.as_deref(),
        round,
        view.as_ref(),
    ) else {
        rest_all(&mut actors);
        state.intents.clear();
        return;
    };
    let ctx = Ctx {
        loaded,
        places,
        round,
        view,
        arena: &facts.arena,
        paths: &facts.paths,
        powers: facts.powers.as_deref(),
        drops: facts.drops.as_deref(),
        tick,
    };
    let state = &mut *state;
    snapshot(&actors, loaded, &mut state.whos, &mut state.reaches);
    let key = Key {
        matches: round.matches,
        seed: round.seed,
        numbers: facts.book.numbers_revision,
        paths: facts.book.paths_revision,
        level,
    };
    if state.key != Some(key) {
        state.key = Some(key);
        reset(state, &ctx);
    }
    ensure_brains(state, round, level);
    let events = Events {
        body: said.body.as_deref(),
        arms: said.arms.as_deref(),
        powers: said.powers.as_deref(),
        round: said.round.as_deref(),
    };
    senses::noises(
        &events,
        &mut state.read,
        &state.whos,
        loaded,
        &mut state.noises,
    );
    read_powers(state, &ctx);
    read_hazards(state, facts.hazards.as_deref());
    read_objects(&mut state.placed, ctx.powers);
    let games = [0, 1].map(|team| senses::match_sense(round, view, places, view.sides[team]));
    sight_table(state, &ctx);
    team_sight(state, tick, &mut minds.sight);
    reserve_gifts(state, &ctx);
    plan(state, &ctx, &games);
    let roles = |slot: u32| {
        state
            .members
            .iter()
            .flatten()
            .find(|m| m.id == slot)
            .map_or(Role::Support, |m| m.role)
    };
    let counter = Counter {
        view,
        powers: ctx.powers,
        whos: &state.whos,
        reaches: &state.reaches,
        roles: &roles,
        carrier: senses::carrier(round),
        play: &loaded.bots.play,
    };
    let mut shopping = std::mem::take(&mut state.shopping);
    shopping.refresh(&counter);
    state.shopping = shopping;
    state.intents.clear();
    for at in 0..state.whos.len() {
        if state.whos[at].human {
            continue;
        }
        if let Some(intent) = think_one(state, &ctx, &games, at) {
            state.intents.push((state.whos[at].slot, intent));
        }
    }
    for team in 0..2 {
        state.calls[team] = std::mem::take(&mut state.next_calls[team]);
    }
    write(state, &mut actors);
    record(state, &mut minds);
    minds.problem = unthought(&seats, state);
}
