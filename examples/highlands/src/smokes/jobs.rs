use engine::smoke::{SmokeResult, SmokeSession};
use engine::terrain::data::smokes::{
    Checked, above, camera, height_at, heights, inspect, number, render,
};
use engine::{smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::open;

const SETTLE_FRAMES: u32 = 20_000;

const WALK_STEPS: u32 = 2400;

const EYE: [f64; 2] = [-120.0, 260.0];

const AIM: [f64; 2] = [380.0, -340.0];

struct Looked {
    samples: Vec<Value>,
    drawn: Value,
    jobs: Value,
}

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let workers = look(session, false)?;
    let jobs = &workers.jobs;
    smoke_check!(
        number(&jobs["workers"]) > 0.0,
        "workers_run_jobs",
        jobs,
        &json!({"workers": above(0)})
    );
    smoke_check!(
        number(&jobs["completed"]) > 0.0 && jobs["failed"] == 0,
        "the_bakes_and_tiles_run_there",
        jobs,
        &json!({"completed": above(0), "failed": 0})
    );
    smoke_check!(
        number(&jobs["shares"]) > 0.0 && number(&jobs["shared_bytes"]) > 0.0,
        "the_terrain_layers_are_shared_with_them",
        jobs,
        &json!({"shares": above(0), "shared_bytes": above(0)})
    );
    smoke_eq!(
        &jobs["in_place"],
        &json!(0),
        "with_workers_no_job_runs_in_place"
    );
    let page = look(session, true)?;
    smoke_eq!(&page.jobs["workers"], &json!(0), "in_place_runs_no_workers");
    smoke_check!(
        number(&page.jobs["in_place"]) > 0.0 && page.jobs["failed"] == 0,
        "without_workers_jobs_run_in_place",
        &page.jobs,
        &json!({"in_place": above(0), "failed": 0})
    );
    alike(&workers, &page)
}

fn alike(workers: &Looked, page: &Looked) -> Checked<()> {
    let answered = workers
        .samples
        .iter()
        .filter(|sample| !sample.is_null())
        .count();
    smoke_check!(
        answered > workers.samples.len() / 2,
        "most_points_have_ground",
        &answered,
        &above(workers.samples.len() / 2)
    );
    smoke_eq!(
        &without_standing(&workers.samples),
        &without_standing(&page.samples),
        "samples_match_bit_for_bit"
    );
    standing_alike(&workers.samples, &page.samples)?;
    smoke_check!(
        number(&workers.drawn["placed_objects"]) > 0.0,
        "objects_stand_on_the_tiles",
        &workers.drawn,
        &json!({"placed_objects": above(0)})
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
    smoke_check!(
        floor.is_finite(),
        "the_eye_has_ground",
        &ground,
        &json!({"height": "finite"})
    );
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
    let jobs = inspect(session, "Jobs")?;
    Ok(Looked {
        samples,
        drawn: drawn(&rendering),
        jobs,
    })
}

fn walks(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut agents = Value::Null;
    for _ in 0..WALK_STEPS {
        session.step(1);
        agents = inspect(session, "NavAgents")?;
        if number(&agents["walking"]) > 0.0 {
            break;
        }
    }
    smoke_eq!(&agents["walking"], &json!(1), "the_hiker_walks");
    smoke_check!(
        number(&agents["portals"]) > 0.0 && number(&agents["grounds"]) > 0.0,
        "the_walk_read_its_tiles_portals_and_grounds",
        &agents,
        &json!({"portals": above(0), "grounds": above(0)})
    );
    smoke_eq!(
        &agents["failed_jobs"],
        &json!(0),
        "no_record_a_walk_read_failed"
    );
    Ok(())
}

fn drawn(rendering: &Value) -> Value {
    let terrain = &rendering["terrain"];
    json!({
        "tiles": terrain["tiles"],
        "leaves": terrain["leaves"],
        "triangles": terrain["triangles"],
        "shadow_patches": terrain["shadow_patches"],
        "placed_objects": terrain["placed_objects"],
        "placed_far_objects": terrain["placed_far_objects"],
        "water_patches": rendering["water"]["patches"],
    })
}

fn settle(session: &mut dyn SmokeSession) -> Checked<Value> {
    let (mut previous, mut quiet) = (Value::Null, 0);
    let mut terrains = Value::Null;
    for _ in 0..SETTLE_FRAMES {
        let frame = drawn(&render(session)?);
        terrains = inspect(session, "Terrains")?;
        let running = inspect(session, "Jobs")?["running"].clone();
        let settled = number(&terrains["resident_tiles"]) > 0.0
            && terrains["baking"] == 0
            && terrains["loading_tiles"] == 0
            && terrains["fallback_tiles"] == 0
            && running == 0
            && frame == previous;
        previous = frame;
        quiet = if settled { quiet + 1 } else { 0 };
        if quiet >= 3 {
            break;
        }
    }
    smoke_check!(
        quiet >= 3,
        "the_terrain_settles",
        &terrains,
        &json!({"resident_tiles": above(0), "baking": 0, "loading_tiles": 0, "fallback_tiles": 0})
    );
    smoke_eq!(&terrains["failed_jobs"], &json!(0), "no_terrain_job_failed");
    Ok(terrains)
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
    let (mut wet, mut apart, mut wetness) = (0_usize, 0_usize, 0_usize);
    for (a, b) in workers.iter().zip(page) {
        let surface = |sample: &Value| sample["standing_water"]["surface"].as_f64();
        match (surface(a), surface(b)) {
            (Some(a), Some(b)) => {
                wet += 1;
                apart += usize::from((a - b).abs() > 0.05);
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => wetness += 1,
        }
    }
    smoke_check!(
        wetness * 50 <= workers.len() && apart * 100 <= wet.max(1),
        "standing_water_settles_alike",
        &json!({"samples": workers.len(), "wet": wet, "apart": apart, "wetness": wetness}),
        &json!({"apart_at_most_percent": 1, "wetness_at_most_percent": 2})
    );
    Ok(())
}
