use engine_core::smoke::{Checked, control, environment, number, render};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::{aim, open};

const STREAMS: [&str; 3] = ["torrent", "rapids", "meadow_river"];

const WATER_VIEWS: [&str; 9] = [
    "water_depth",
    "water_speed",
    "water_froude",
    "water_turbulence",
    "water_white",
    "water_reach",
    "water_span",
    "water_eddies",
    "water_roughness",
];

const REACHES: [&str; 5] = ["pool", "plane", "steps", "cascade", "falls"];

const NOT_WORK: [&str; 4] = ["frame", "tick", "read_back_age", "bricks_cost.frame"];

const QUIET_FRAMES: u32 = 20_000;

const PROBES_ACROSS: u32 = 7;

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    open(session)?;
    for stream in STREAMS {
        aim(session, stream)?;
        environment(session, json!({"debug_view": {"view": "off"}}))?;
        quiet(session)?;
        for view in WATER_VIEWS {
            environment(session, json!({"debug_view": {"view": view}}))?;
            let shown = quiet(session)?;
            let passes = &shown["render_passes"];
            smoke_eq!(
                passes
                    .as_array()
                    .is_some_and(|passes| passes.iter().any(|pass| pass == "debug_view")),
                true,
                "the_water_views_draw"
            );
        }
    }
    probed(session)?;
    environment(session, json!({"debug_view": {"view": "off"}}))?;
    Ok(())
}

fn probed(session: &mut dyn SmokeSession) -> Checked<()> {
    aim(session, "torrent")?;
    environment(session, json!({"debug_view": {"view": "water_depth"}}))?;
    let rendering = quiet(session)?;
    let width = number(&rendering["resolution"]["display_width"]);
    let height = number(&rendering["resolution"]["display_height"]);
    let mut pixels: Vec<[f64; 2]> = (1..PROBES_ACROSS)
        .flat_map(|row| {
            (1..PROBES_ACROSS).map(move |column| {
                let across = f64::from(PROBES_ACROSS);
                [
                    (width * f64::from(column) / across).floor(),
                    (height * f64::from(row) / across).floor(),
                ]
            })
        })
        .collect();
    let off_center = |pixel: &[f64; 2]| (pixel[0] - width / 2.0).hypot(pixel[1] - height / 2.0);
    pixels.sort_by(|a, b| off_center(a).total_cmp(&off_center(b)));
    let mut probe = Value::Null;
    for [x, y] in pixels {
        probe = control(session, "probe", json!({"x": x as u32, "y": y as u32}))?;
        if probe["kind"] == "water" {
            break;
        }
    }
    smoke_eq!(
        &probe["kind"],
        &json!("water"),
        "the_torrent_view_holds_water"
    );
    smoke_eq!(&probe["terrain"], &Value::Null, "water_is_probed_as_water");
    let water = &probe["water"];
    let depth = number(&water["depth"]);
    smoke_eq!(
        depth > 0.0 && depth.is_finite(),
        true,
        "the_probe_reads_a_depth"
    );
    smoke_eq!(
        number(&water["speed"]) >= 0.0 && number(&water["froude"]) >= 0.0,
        true,
        "the_probe_reads_the_flow"
    );
    let share = |field: &str| (0.0..=1.0).contains(&number(&water[field]));
    smoke_eq!(
        share("turbulence") && share("white") && share("roughness"),
        true,
        "the_probe_reads_its_shares"
    );
    smoke_eq!(
        water["reach"].is_null() || REACHES.iter().any(|reach| water["reach"] == *reach),
        true,
        "the_probe_names_its_reach"
    );
    smoke_eq!(
        number(&water["span"]) > 0.0 && number(&water["eddies"]) >= 0.0,
        true,
        "the_probe_reads_span_and_eddies"
    );
    Ok(())
}

fn quiet(session: &mut dyn SmokeSession) -> Checked<Value> {
    let (mut previous, mut calm): (Vec<(String, Value)>, u32) = (Vec::new(), 0);
    let mut rendering = Value::Null;
    let mut changing: Vec<String> = Vec::new();
    for _ in 0..QUIET_FRAMES {
        rendering = render(session)?;
        let busy = rendering["loading"] == true || rendering["pipelines_compiling"] == true;
        let mut counters = Vec::new();
        work(&rendering, "", &mut counters);
        changing = counters
            .iter()
            .filter(|counter| !previous.contains(*counter))
            .map(|(path, _)| path.clone())
            .collect();
        calm = if !busy && counters == previous {
            calm + 1
        } else {
            0
        };
        previous = counters;
        if calm >= 3 {
            break;
        }
    }
    let unsettled = json!({
        "loading": rendering["loading"],
        "pipelines_compiling": rendering["pipelines_compiling"],
        "changing": changing,
    });
    smoke_eq!(
        if calm >= 3 { Value::Null } else { unsettled },
        Value::Null,
        "the_view_settles"
    );
    for _ in 0..30 {
        rendering = render(session)?;
    }
    Ok(rendering)
}

fn work(stats: &Value, prefix: &str, out: &mut Vec<(String, Value)>) {
    let Some(fields) = stats.as_object() else {
        return;
    };
    for (key, value) in fields {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            Value::Array(items) => out.push((path, json!(items.len()))),
            Value::Object(_) => work(value, &path, out),
            Value::Number(_) | Value::Bool(_) => {
                if !NOT_WORK.contains(&path.as_str()) && !path.ends_with("_ms") {
                    out.push((path, value.clone()));
                }
            }
            Value::Null | Value::String(_) => {}
        }
    }
}
