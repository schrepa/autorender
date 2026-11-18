engine::brick!(flow);

mod keep;
mod roster;

use std::sync::Arc;

use engine::prelude::*;
use engine::{Pause, PauseReason};
use engine_core::Readiness;

use crate::bricks::screen::ScreenSource;
use crate::play::flow::{Flow, SeatChoice, Stage};
use crate::play::numbers::Numbers;
use crate::play::round::Round;
use crate::play::sound::SoundLoad;
use crate::rules::{Control, Phase};

use roster::Seated;

const PAUSED_BY: PauseReason = PauseReason::Brick("fruit_fps_flow");
const HOST_AWAY: [PauseReason; 2] = [PauseReason::Focus, PauseReason::Hidden];

settings! {
    #[section("flow")]
    #[serde(deny_unknown_fields)]
    pub struct FlowSetup {
        pub play_now: bool = false => any,
    }
}

type DefaultSeats = Result<Arc<[SeatChoice]>, String>;

#[derive(Resource, Default)]
struct FlowBook {
    defaults: Option<(u64, DefaultSeats)>,
    jumped: bool,
    seated_game: Option<u64>,
    loading_since: f64,
    saw_live: bool,
    waited: f64,
    match_end_since: f64,
    next_match_seconds: f64,
    waiting: Vec<String>,
    problem: Option<String>,
}

stats! {
    #[section("flow_state")]
    pub struct FlowStats {
        pub stage: String,
        pub settings_open: bool,
        pub game: u64,
        pub character: String,
        pub level: String,
        pub progress: f32,
        pub paused: bool,
        pub human_slot: Option<u32>,
        pub loading_seconds: f64,
        pub waiting_for: Vec<String>,
        pub settings_restored: bool,
        pub settings_writes: u32,
        pub settings_problem: Option<String>,
        pub problem: Option<String>,
    }
}

pub struct FlowBrick;

impl Brick for FlowBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<FlowSetup>()
            .stats::<FlowStats>()
            .resource::<Flow>()
            .resource::<FlowBook>()
            .resource::<keep::KeepBook>()
            .systems(Schedule::Presentation, keep::keep_settings)
            .systems(Schedule::Presentation, steer)
            .systems(Schedule::Presentation, report);
    }
}

fn defaults(book: &mut FlowBook, numbers: &Numbers) -> Option<Arc<[SeatChoice]>> {
    let game = numbers.get()?;
    let fresh = book
        .defaults
        .as_ref()
        .is_none_or(|(revision, _)| *revision != numbers.revision());
    if fresh {
        book.defaults = Some((
            numbers.revision(),
            roster::default_roster(&game.text).map(Arc::from),
        ));
    }
    match book.defaults.as_ref().map(|(_, read)| read) {
        Some(Ok(seats)) => {
            let seats = Arc::clone(seats);
            book.problem = None;
            Some(seats)
        }
        Some(Err(problem)) => {
            book.problem = Some(problem.clone());
            None
        }
        None => None,
    }
}

fn choose_defaults(flow: &mut Flow, seats: &[SeatChoice]) {
    let human = seats.iter().find(|seat| seat.control == Control::Human);
    let known = |character: &str| {
        seats.iter().any(|seat| {
            Some(seat.team) == human.map(|human| human.team) && seat.character == character
        })
    };
    if !known(&flow.character)
        && let Some(seat) = human
    {
        flow.character.clone_from(&seat.character);
    }
    if !seats.iter().any(|seat| seat.level == flow.level)
        && let Some(seat) = seats.iter().find(|seat| seat.control == Control::Bot)
    {
        flow.level.clone_from(&seat.level);
    }
}
fn loaded_share(readiness: Option<&Readiness>) -> f32 {
    match readiness {
        Some(readiness) if readiness.loading.total > 0 => {
            readiness.loading.done as f32 / readiness.loading.total as f32
        }
        Some(_) | None => 1.0,
    }
}

fn loading(
    flow: &mut Flow,
    book: &mut FlowBook,
    (pause, sound, readiness): (&Pause, Option<&SoundLoad>, Option<&Readiness>),
    (now, least): (f64, f64),
) {
    let waited = now - book.loading_since;
    book.waited = waited;
    let time_share = if least > 0.0 {
        (waited / least) as f32
    } else {
        1.0
    };
    let host_ready = !pause.holds(PauseReason::Preparing);
    let sound_ready = sound.is_none_or(|sound| sound.ready);
    let seated = book.seated_game == Some(flow.game);
    let share = if host_ready {
        time_share
    } else {
        engine::math::min(loaded_share(readiness), time_share)
    };
    flow.progress = engine::math::clamp(share, 0.0, 1.0);
    let waiting = &mut book.waiting;
    waiting.clear();
    if !host_ready {
        waiting.push("host preparing (files, pipelines)".to_owned());
    }
    if let Some(sound) = sound.filter(|sound| !sound.ready) {
        waiting.push(format!(
            "sound {}/{} loaded, {} failed",
            sound.loaded, sound.asked, sound.failed
        ));
    }
    if waited < least {
        waiting.push(format!(
            "min time {waited:.2}/{least:.2} s of FrameTime.real_elapsed"
        ));
    }
    if !seated {
        waiting.push("seating the roster".to_owned());
    }
    if host_ready && sound_ready && waited >= least && seated {
        flow.stage = Stage::Playing;
    }
}
type Reads<'w, 's> = (
    Option<Res<'w, Numbers>>,
    Option<Res<'w, Round>>,
    Option<Res<'w, Readiness>>,
    Res<'w, FrameTime>,
    Res<'w, FlowSetup>,
    Query<'w, 's, &'static ScreenSource>,
    Option<Res<'w, SoundLoad>>,
);

fn steer(
    (numbers, round, readiness, time, setup, sources, sound): Reads,
    actors: Query<Seated<'_>>,
    mut commands: Commands,
    mut flow: ResMut<Flow>,
    mut pause: ResMut<Pause>,
    mut book: ResMut<FlowBook>,
) {
    let book = &mut *book;
    let flow = &mut *flow;
    let now = time.real_elapsed;
    let seats = numbers
        .as_deref()
        .and_then(|numbers| defaults(book, numbers));
    if let Some(seats) = seats.as_deref().filter(|_| flow.restored) {
        choose_defaults(flow, seats);
        if setup.play_now && !book.jumped {
            flow.roster.clear();
            roster::seat_human(&mut commands, seats, &actors);
            flow.settings_open = false;
            flow.stage = Stage::Playing;
            flow.game += 1;
            book.seated_game = Some(flow.game);
            book.saw_live = false;
        }
        book.jumped = setup.play_now;
        if flow.stage == Stage::Loading && book.seated_game != Some(flow.game) {
            flow.roster = roster::chosen(seats, &flow.character, &flow.level);
            roster::seat_human(&mut commands, &flow.roster, &actors);
            book.seated_game = Some(flow.game);
            book.loading_since = now;
            book.saw_live = false;
        }
    }
    if flow.stage == Stage::Loading {
        let least = sources
            .iter()
            .next()
            .map_or(0.0, |source| source.menu.loading.min_seconds);
        loading(
            flow,
            book,
            (&pause, sound.as_deref(), readiness.as_deref()),
            (now, least),
        );
    } else {
        book.waiting.clear();
    }
    let away = HOST_AWAY.iter().any(|reason| pause.holds(*reason));
    let asked = std::mem::take(&mut flow.pause_asked);
    if (asked || away) && flow.stage == Stage::Playing {
        flow.stage = Stage::Paused;
    }
    if flow.stage == Stage::Playing {
        let phase = round.as_deref().and_then(Round::phase);
        if phase.is_some_and(|phase| phase != Phase::Over) {
            book.saw_live = true;
        }
        if book.saw_live && phase == Some(Phase::Over) {
            flow.stage = Stage::MatchEnd;
            book.match_end_since = now;
            book.next_match_seconds = round
                .as_deref()
                .and_then(|round| {
                    round.view().map(|view| {
                        f64::from(round.restart_ticks)
                            / f64::from(view.catalog.ticks_per_second.max(1))
                    })
                })
                .unwrap_or(0.0);
        }
    }
    flow.next_match_left = 0.0;
    if flow.stage == Stage::MatchEnd && book.next_match_seconds > 0.0 {
        let left = book.next_match_seconds - (now - book.match_end_since);
        flow.next_match_left = engine::math::max(left as f32, 0.0);
        if left <= 0.0 && !flow.settings_open {
            flow.game += 1;
            flow.stage = Stage::Playing;
            book.seated_game = Some(flow.game);
            book.saw_live = false;
            book.next_match_seconds = 0.0;
        }
    }
    let stopped = !flow.stage.lobby() && (flow.stage != Stage::Playing || flow.settings_open);
    if stopped != pause.holds(PAUSED_BY) {
        if stopped {
            pause.add(PAUSED_BY);
        } else {
            pause.remove(PAUSED_BY);
        }
    }
}

fn report(
    flow: Res<Flow>,
    pause: Res<Pause>,
    (book, kept): (Res<FlowBook>, Res<keep::KeepBook>),
    humans: Query<&crate::play::actors::Actor, With<crate::play::actors::Human>>,
    mut stats: ResMut<FlowStats>,
) {
    let next = FlowStats {
        stage: flow.stage.key().to_owned(),
        settings_open: flow.settings_open,
        game: flow.game,
        character: flow.character.clone(),
        level: flow.level.clone(),
        progress: flow.progress,
        paused: pause.holds(PAUSED_BY),
        human_slot: humans.iter().next().map(|actor| actor.slot),
        loading_seconds: book.waited,
        waiting_for: book.waiting.clone(),
        settings_restored: kept.restored,
        settings_writes: kept.writes,
        settings_problem: kept.problem.clone(),
        problem: book.problem.clone(),
    };
    stats.set_if_neq(next);
}
