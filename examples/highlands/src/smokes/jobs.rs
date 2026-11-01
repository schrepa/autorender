use engine_core::smoke::{Checked, camera, inspect, number, render};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::{height_at, heights, open};

const SETTLE_FRAMES: u32 = 20_000;

const WALK_STEPS: u32 = 2400;

const EYE: [f64; 2] = [-120.0, 260.0];

const AIM: [f64; 2] = [380.0, -340.0];

struct Looked {
    samples: Vec<Value>,
    drawn: Value,
    jobs: Value,
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let workers = look(session, false)?;
    let jobs = &workers.jobs;
    smoke_eq!(number(&jobs["workers"]) > 0.0, true, "workers_run_jobs");
    smoke_eq!(
        number(&jobs["completed"]) > 0.0 && jobs["failed"] == 0,
        true,
        "the_bakes_and_tiles_run_there"
    );
    smoke_eq!(
        number(&jobs["shares"]) > 0.0 && number(&jobs["shared_bytes"]) > 0.0,
        true,
        "the_terrain_layers_are_shared_with_them"
    );
    smoke_eq!(
        &jobs["in_place"],
        &json!(0),
        "with_workers_no_job_runs_in_place"
    );
    let page = look(session, true)?;
    smoke_eq!(&page.jobs["workers"], &json!(0), "in_place_runs_no_workers");
    smoke_eq!(
        number(&page.jobs["in_place"]) > 0.0 && page.jobs["failed"] == 0,
        true,
        "without_workers_jobs_run_in_place"
    );
    alike(&workers, &page)
}

fn alike(workers: &Looked, page: &Looked) -> Checked<()> {
    let answered = workers
        .samples
        .iter()
        .filter(|sample| !sample.is_null())
        .count();
    smoke_eq!(
        answered > workers.samples.len() / 2,
        true,
        "most_points_have_ground"
    );
    smoke_eq!(
        &without_standing(&workers.samples),
        &without_standing(&page.samples),
        "samples_match_bit_for_bit"
    );
    standing_alike(&workers.samples, &page.samples)?;
    smoke_eq!(
        number(&workers.drawn["placed_objects"]) > 0.0,
        true,
        "objects_stand_on_the_tiles"
    );
    smoke_eq!(
        &workers.drawn,
        &page.drawn,
        "an_exact_frame_draws_the_same_tiles_water_and_objects"
    );
    Ok(())
}

fn look(session: &mut dyn SmokeSession, in_place: bool) -> Checked<Looked> {
    open(session, in_place)?;
    if !in_place {
        walks(session)?;
    }
    settle(session)?;
    let ground = height_at(session, EYE)?;
    let floor = number(&ground["height"]);
    smoke_eq!(floor.is_finite(), true, "the_eye_has_ground");
    camera(
        session,
        json!({
            "position": [EYE[0], floor + 30.0, EYE[1]],
            "target": [AIM[0], floor - 60.0, AIM[1]],
            "vertical_fov_radians": 1.0,
            "near": 0.2,
            "far": 40000.0,
        }),
    )?;
    settle(session)?;
    let points: Vec<[f64; 2]> = (0..16_i32)
        .flat_map(|row| {
            (0..16_i32)
                .map(move |column| [f64::from(-160 + column * 40), f64::from(300 - row * 45)])
        })
        .collect();
    let samples = heights(session, &points)?;
    let rendering = render(session)?;
    Ok(Looked {
        samples,
        drawn: drawn(session)?,
        jobs: rendering["jobs"].clone(),
    })
}

fn walks(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut agents = Value::Null;
    for _ in 0..WALK_STEPS {
        session.step(1);
        agents = inspect(session, "navigation_stats")?;
        if number(&agents["walking"]) > 0.0 {
            break;
        }
    }
    smoke_eq!(&agents["walking"], &json!(1), "the_hiker_walks");
    smoke_eq!(
        number(&agents["portals"]) > 0.0 && number(&agents["grounds"]) > 0.0,
        true,
        "the_walk_read_its_tiles_portals_and_grounds"
    );
    let rendering = render(session)?;
    smoke_eq!(
        &rendering["jobs"]["failed"],
        &json!(0),
        "no_record_a_walk_read_failed"
    );
    Ok(())
}

fn drawn(session: &mut dyn SmokeSession) -> Checked<Value> {
    let shading = inspect(session, "terrain_shading_stats")?;
    let scatter = inspect(session, "scatter_stats")?;
    let water = inspect(session, "water_stats")?;
    Ok(json!({
        "tiles": shading["tiles"],
        "leaves": shading["leaves"]["kept"],
        "triangles": shading["triangles"],
        "shadow_patches": shading["shadow_patches"],
        "placed_objects": scatter["placed_objects"],
        "placed_far_objects": scatter["placed_far_objects"],
        "water_leaves": water["leaf_cut"],
    }))
}

fn settle(session: &mut dyn SmokeSession) -> Checked<Value> {
    let (mut previous, mut quiet) = (Value::Null, 0);
    let mut terrain = Value::Null;
    let mut jobs = Value::Null;
    for _ in 0..SETTLE_FRAMES {
        let rendering = render(session)?;
        let frame = drawn(session)?;
        terrain = inspect(session, "terrain_stats")?;
        jobs = rendering["jobs"].clone();
        let settled = number(&terrain["resident_tiles"]) > 0.0
            && terrain["baking"] == 0
            && terrain["loading_tiles"] == 0
            && terrain["fallback_tiles"] == 0
            && jobs["running"] == 0
            && frame == previous;
        previous = frame;
        quiet = if settled { quiet + 1 } else { 0 };
        if quiet >= 3 {
            break;
        }
    }
    smoke_eq!(quiet >= 3, true, "the_terrain_settles");
    smoke_eq!(&jobs["failed"], &json!(0), "no_terrain_job_failed");
    Ok(terrain)
}

fn without_standing(samples: &[Value]) -> Vec<Value> {
    samples
        .iter()
        .map(|sample| {
            let mut sample = sample.clone();
            if let Some(fields) = sample.as_object_mut() {
                fields.remove("standing_water");
            }
            sample
        })
        .collect()
}

fn standing_alike(workers: &[Value], page: &[Value]) -> Checked<()> {
    let (mut apart, mut wetness) = (0_usize, 0_usize);
    for (a, b) in workers.iter().zip(page) {
        let surface = |sample: &Value| sample["standing_water"]["surface"].as_f64();
        match (surface(a), surface(b)) {
            (Some(a), Some(b)) => apart += usize::from((a - b).abs() > 0.05),
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => wetness += 1,
        }
    }
    smoke_eq!(
        wetness * 50 <= workers.len() && apart * 100 <= workers.len(),
        true,
        "standing_water_settles_alike"
    );
    Ok(())
}
