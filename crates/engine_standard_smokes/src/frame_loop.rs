use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::json;

use super::requests::{FRAME_MS, control, frame, frames, grid, number, seen, status, tick};

pub(crate) fn frame_loop(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&grid(8, false).to_string());
    control(session, "pause", json!({}), "pauses")?;
    let before = status(session, "status_answers")?;
    session.idle(300.0);
    let still = status(session, "status_answers_idle")?;
    smoke_eq!(
        json!([frame(&still), tick(&still)]),
        json!([frame(&before), tick(&before)]),
        "idle_runs_no_frames"
    );
    let step_seconds = number(&before["result"]["fixed_delta_seconds"]);
    session.idle(500.0);
    let ran = frames(session, 10, "frames_run")?;
    let drawn = frame(&ran).saturating_sub(frame(&still));
    let wanted = json!(10);
    smoke_eq!(
        seen(drawn >= 10, json!(drawn), &wanted),
        wanted,
        "area_runs_again"
    );
    let ticks = tick(&ran).saturating_sub(tick(&still)) as f64;
    let allowed = 9.0 * FRAME_MS / 1000.0 / step_seconds + 3.0;
    let wanted = json!(allowed);
    smoke_eq!(
        seen(ticks <= allowed, json!(ticks), &wanted),
        wanted,
        "idle_time_dropped"
    );
    control(session, "resume", json!({}), "resumes")?;
    let resumed = status(session, "status_answers_resumed")?;
    let shown = (0..10).filter(|_| session.frame()).count();
    smoke_eq!(json!(shown), json!(10), "prepares_then_plays");
    let running = status(session, "status_answers_running")?;
    let played = frame(&running).saturating_sub(frame(&resumed));
    let wanted = json!(10);
    smoke_eq!(
        seen(played >= 10, json!(played), &wanted),
        wanted,
        "frames_play_while_resumed"
    );
    control(session, "pause", json!({}), "pauses_again")?;
    Ok(())
}
