use engine_core::smoke::{SmokeResult, SmokeSession};
use serde_json::json;

use super::requests::{FRAME_MS, control, frame, frames, grid, idle, number, spawn, status, tick};

pub(crate) fn frame_loop(session: &mut dyn SmokeSession) -> SmokeResult {
    spawn(session, &grid(8, false))?;
    control(session, "pause", json!({}), "pauses")?;
    let before = status(session, "status_answers")?;
    idle(300.0);
    let still = status(session, "status_answers_idle")?;
    engine_core::smoke_eq!(
        json!([frame(&still), tick(&still)]),
        json!([frame(&before), tick(&before)]),
        "idle_runs_no_frames"
    );
    let step_seconds = number(&before["result"]["fixed_delta_seconds"]);
    idle(500.0);
    let ran = frames(session, 10, "frames_run")?;
    let drawn = frame(&ran).saturating_sub(frame(&still));
    engine_core::smoke_check!(drawn >= 10, "area_runs_again", json!(drawn), json!(10));
    let ticks = tick(&ran).saturating_sub(tick(&still)) as f64;
    let allowed = 9.0 * FRAME_MS / 1000.0 / step_seconds + 3.0;
    engine_core::smoke_check!(
        ticks <= allowed,
        "idle_time_dropped",
        json!(ticks),
        json!(allowed)
    );
    control(session, "resume", json!({}), "resumes")?;
    let resumed = status(session, "status_answers_resumed")?;
    let shown = (0..10).filter(|_| session.frame()).count();
    engine_core::smoke_eq!(json!(shown), json!(10), "prepares_then_plays");
    let running = status(session, "status_answers_running")?;
    let played = frame(&running).saturating_sub(frame(&resumed));
    engine_core::smoke_check!(
        played >= 10,
        "frames_play_while_resumed",
        json!(played),
        json!(10)
    );
    control(session, "pause", json!({}), "pauses_again")?;
    Ok(())
}
