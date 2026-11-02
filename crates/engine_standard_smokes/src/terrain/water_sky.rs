use engine_core::smoke::{
    COMPILE_FRAMES, Checked, above, camera, capture, control, environment, inspect, not, number,
    query, refuse, render, renders_until, with,
};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::{
    SETTLE_FRAMES, WAIT_FRAMES, fast, height_at, heights, open, overview, settle, view, viewed,
};
use crate::requests::seen;
use crate::stats::terrain;

const SIDE: usize = 513;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    open(session, false)?;
    hydrology(session)?;
    lakes(session)?;
    water(session)?;
    clouds(session)?;
    clouds_around(session)?;
    let starry = sky(session)?;
    exposure(session, &starry)?;
    sky_rules(session)?;
    settle(session)?;
    let current = horizons(session)?;
    bounce(session)?;
    aerial(session, &current)?;
    captures(session)
}

fn captures(session: &mut dyn SmokeSession) -> Checked<()> {
    control(session, "resume", json!({}))?;
    let mut rendering = Value::Null;
    for _ in 0..COMPILE_FRAMES {
        if !session.frame() {
            break;
        }
        let mut status = query(session, "status", json!({}))?;
        rendering = status["rendering"].take();
        if number(&rendering["reflection_probes_stats"]["ready"]) > 0.0 {
            break;
        }
    }
    control(session, "pause", json!({}))?;
    let wanted = json!({"ready": above(0)});
    smoke_eq!(
        seen(
            number(&rendering["reflection_probes_stats"]["ready"]) > 0.0,
            rendering["reflection_probes_stats"].clone(),
            &wanted,
        ),
        wanted,
        "a_probe_captures"
    );
    Ok(())
}

fn hydrology(session: &mut dyn SmokeSession) -> Checked<()> {
    let bakes = viewed(session)?;
    smoke_eq!(
        &bakes["baking"],
        &json!(false),
        "the_bake_is_done_once_tiles_load"
    );
    let water = &bakes["hydrology"];
    let wanted = json!({ "samples": SIDE * SIDE });
    smoke_eq!(
        seen(
            water["samples"].as_u64() == Some((SIDE * SIDE) as u64),
            water.clone(),
            &wanted,
        ),
        wanted,
        "the_bake_reads_its_level"
    );
    let wanted = json!({"channel_samples": above(0), "max_area": above(20_000)});
    smoke_eq!(
        seen(
            number(&water["channel_samples"]) > 0.0 && number(&water["max_area"]) > 20_000.0,
            water.clone(),
            &wanted,
        ),
        wanted,
        "water_gathers_into_channels"
    );
    let watered = height_at(session, [-300.0, -300.0])?;
    let wanted = json!({"water": {"flow": above(0), "wetness": "finite"}});
    smoke_eq!(
        seen(
            number(&watered["water"]["flow"]) > 0.0
                && number(&watered["water"]["wetness"]).is_finite(),
            watered.clone(),
            &wanted,
        ),
        wanted,
        "queries_report_water"
    );
    let falls = inspect(session, "water_stats")?;
    smoke_eq!(&falls["falls"], &water["falls"], "falls_spawn_as_found");
    Ok(())
}

fn lakes(session: &mut dyn SmokeSession) -> Checked<()> {
    let grid: Vec<[f64; 2]> = (0..25)
        .flat_map(|row| {
            (0..25).map(move |column| [f64::from(-480 + column * 40), f64::from(-480 + row * 40)])
        })
        .collect();
    let found = heights(session, &grid)?;
    let lakes: Vec<&Value> = found
        .iter()
        .filter(|sample| sample["standing_water"].is_object())
        .collect();
    let standing = |sample: &&Value| {
        let water = &sample["standing_water"];
        number(&water["depth"]) > 0.0
            && (number(&water["surface"]) - number(&water["depth"]) - number(&sample["height"]))
                .abs()
                < 1e-2
    };
    let wanted = json!({"standing_water": {"depth": above(0)}});
    smoke_eq!(
        seen(
            !lakes.is_empty() && lakes.iter().all(standing),
            json!(lakes),
            &wanted,
        ),
        wanted,
        "water_stands_in_lakes"
    );
    Ok(())
}

fn water(session: &mut dyn SmokeSession) -> Checked<()> {
    let settings = inspect(session, "water")?;
    camera(session, overview())?;
    let wet = environment(session, json!({ "water": settings }))?;
    let wanted = json!({"leaf_cut": {"kept": above(0)}});
    smoke_eq!(
        seen(
            number(&wet["rendering"]["water_stats"]["leaf_cut"]["kept"]) > 0.0,
            wet["rendering"]["water_stats"].clone(),
            &wanted,
        ),
        wanted,
        "water_draws"
    );
    let with_water = capture(session)?;
    environment(
        session,
        json!({"water": with(&settings, json!({"enabled": false}))}),
    )?;
    let without_water = capture(session)?;
    let wanted = not(&with_water);
    smoke_eq!(
        seen(with_water != without_water, without_water.clone(), &wanted),
        wanted,
        "water_shades_the_view"
    );
    let murky = refuse(
        session,
        "control",
        "environment",
        json!({"water": with(&settings, json!({"clarity": 0.0}))}),
    )?;
    smoke_eq!(
        &murky["path"],
        &json!("$.args.water.clarity"),
        "murky_water_is_refused"
    );
    environment(session, json!({ "water": settings }))?;
    Ok(())
}

fn layer() -> Value {
    json!({
        "enabled": true,
        "coverage": 0.6,
        "base": 600.0,
        "thickness": 900.0,
        "density": 1.0,
        "wind": [8.0, 3.0],
        "scale": 2000.0,
        "seed": 3,
        "shadows": 0.75,
        "resolution": 1024,
        "slices": 16,
    })
}

fn eye(position: [f64; 3], target: [f64; 3], far: f64) -> Value {
    json!({
        "position": position,
        "target": target,
        "vertical_fov_radians": 1.0,
        "near": 0.5,
        "far": far,
    })
}

fn clouds(session: &mut dyn SmokeSession) -> Checked<()> {
    camera(
        session,
        eye([0.0, 60.0, 0.0], [300.0, 400.0, 200.0], 4000.0),
    )?;
    let clear = capture(session)?;
    environment(session, json!({ "clouds": layer() }))?;
    let (marched, clouded) = renders_until(session, COMPILE_FRAMES, |rendering| {
        rendering["cloud_stats"]["rows"] != 0
    })?;
    let wanted = json!({"rows": above(0)});
    smoke_eq!(
        seen(marched, clouded["cloud_stats"].clone(), &wanted),
        wanted,
        "the_cloud_noise_arrives"
    );
    smoke_eq!(
        &clouded["cloud_stats"]["rows"],
        &json!(256),
        "exact_frames_march_the_whole_dome"
    );
    let read = inspect(session, "clouds")?;
    let wanted = json!(0.6);
    smoke_eq!(
        seen(
            (number(&read["coverage"]) - 0.6).abs() < 1e-6,
            read["coverage"].clone(),
            &wanted,
        ),
        wanted,
        "coverage_reads_back"
    );
    let cloudy = capture(session)?;
    let wanted = not(&clear);
    smoke_eq!(
        seen(clear != cloudy, cloudy.clone(), &wanted),
        wanted,
        "clouds_show_in_the_sky"
    );
    camera(session, eye([0.0, 400.0, 1.0], [0.0, 30.0, 0.0], 4000.0))?;
    let shaded = capture(session)?;
    environment(
        session,
        json!({"clouds": with(&layer(), json!({"shadows": 0.0}))}),
    )?;
    let unshaded = capture(session)?;
    let wanted = not(&shaded);
    smoke_eq!(
        seen(shaded != unshaded, unshaded.clone(), &wanted),
        wanted,
        "clouds_shade_the_ground"
    );
    Ok(())
}

fn clouds_around(session: &mut dyn SmokeSession) -> Checked<()> {
    camera(
        session,
        eye([0.0, 1900.0, 1.0], [400.0, 0.0, 300.0], 8000.0),
    )?;
    let over = environment(session, json!({ "clouds": layer() }))?;
    let wanted = json!({"view_pixels": above(0)});
    smoke_eq!(
        seen(
            number(&over["rendering"]["cloud_stats"]["view_pixels"]) > 0.0,
            over["rendering"]["cloud_stats"].clone(),
            &wanted,
        ),
        wanted,
        "the_view_marches_its_clouds"
    );
    let below_view = capture(session)?;
    let domed = environment(
        session,
        json!({"clouds": with(&layer(), json!({"in_view": false}))}),
    )?;
    smoke_eq!(
        &domed["rendering"]["cloud_stats"]["view_pixels"],
        &json!(0),
        "off_the_view_marches_none"
    );
    let dome_only = capture(session)?;
    let wanted = not(&below_view);
    smoke_eq!(
        seen(below_view != dome_only, dome_only.clone(), &wanted),
        wanted,
        "clouds_lie_below_the_camera"
    );
    let none = environment(
        session,
        json!({"clouds": with(&layer(), json!({"coverage": 0.0}))}),
    )?;
    smoke_eq!(
        &none["rendering"]["cloud_stats"]["rows"],
        &json!(0),
        "no_coverage_marches_nothing"
    );
    let odd = refuse(
        session,
        "control",
        "environment",
        json!({"clouds": with(&layer(), json!({"resolution": 1000}))}),
    )?;
    smoke_eq!(
        &odd["path"],
        &json!("$.args.clouds.resolution"),
        "an_odd_cloud_resolution_is_refused"
    );
    environment(
        session,
        json!({"clouds": with(&layer(), json!({"enabled": false}))}),
    )?;
    camera(session, overview())?;
    Ok(())
}

fn sky(session: &mut dyn SmokeSession) -> Checked<Value> {
    camera(
        session,
        eye([0.0, 60.0, 0.0], [300.0, 200.0, 200.0], 4000.0),
    )?;
    let authored = capture(session)?;
    let mut lit = environment(
        session,
        json!({"sky": {"enabled": true}, "time_of_day": {"enabled": true, "hours": 12.0}}),
    )?;
    let mut first = lit["rendering"].take();
    for _ in 0..COMPILE_FRAMES {
        if first["sky_stats"]["tables"] != 0 {
            break;
        }
        first = render(session)?;
    }
    smoke_eq!(
        &first["sky_stats"]["tables"],
        &json!(3),
        "the_air_tables_and_the_sky_view_build"
    );
    let unchanged = render(session)?;
    smoke_eq!(
        &unchanged["sky_stats"]["tables"],
        &json!(0),
        "unchanged_nothing_builds"
    );
    let noon = capture(session)?;
    let wanted = not(&authored);
    smoke_eq!(
        seen(authored != noon, noon.clone(), &wanted),
        wanted,
        "the_physical_sky_replaces_the_authored_one"
    );
    let step = environment(
        session,
        json!({"time_of_day": {"enabled": true, "hours": 19.3}}),
    )?;
    smoke_eq!(
        &step["rendering"]["sky_stats"]["tables"],
        &json!(1),
        "the_sun_step_builds_the_sky_view_alone"
    );
    environment(
        session,
        json!({"time_of_day": {"enabled": true, "hours": 23.0}}),
    )?;
    let starry = capture(session)?;
    let stars = inspect(session, "night_sky")?;
    smoke_eq!(
        &stars["stars"],
        &json!(1.0),
        "the_night_sky_shows_its_stars"
    );
    let haze = inspect(session, "haze")?;
    smoke_eq!(&haze["enabled"], &json!(true), "haze_is_on");
    environment(
        session,
        json!({"night_sky": {"stars": 0.0, "moon": 0.0, "cirrus": 0.0}}),
    )?;
    let night = capture(session)?;
    let wanted = not(&starry);
    smoke_eq!(
        seen(night != starry, night.clone(), &wanted),
        wanted,
        "the_night_sky_shows"
    );
    environment(session, json!({"night_sky": {}}))?;
    Ok(starry)
}

fn exposure(session: &mut dyn SmokeSession, starry: &Value) -> Checked<()> {
    let cirrus = refuse(
        session,
        "control",
        "environment",
        json!({"night_sky": {"cirrus": 2.0}}),
    )?;
    smoke_eq!(
        &cirrus["path"],
        &json!("$.args.night_sky.cirrus"),
        "thick_cirrus_is_refused"
    );
    let hazy = refuse(
        session,
        "control",
        "environment",
        json!({"haze": {"visibility": -1.0}}),
    )?;
    smoke_eq!(
        &hazy["path"],
        &json!("$.args.haze.visibility"),
        "negative_visibility_is_refused"
    );
    let mut metering = environment(session, json!({"auto_exposure": {"enabled": true}}))?;
    let mut exposure = metering["rendering"]["exposure_stats"].take();
    smoke_eq!(
        &exposure["active"],
        &json!(true),
        "frames_meter_their_exposure"
    );
    for _ in 0..WAIT_FRAMES {
        if number(&exposure["stops"]) > 0.0 {
            break;
        }
        let mut rendering = render(session)?;
        exposure = rendering["exposure_stats"].take();
    }
    let stops = number(&exposure["stops"]);
    let wanted = json!({"stops": {"above": 0, "at_most": 400}});
    smoke_eq!(
        seen(stops > 0.0 && stops <= 400.0, exposure.clone(), &wanted),
        wanted,
        "the_night_exposure_keeps_within_its_limits"
    );
    let exposed = capture(session)?;
    let wanted = not(starry);
    smoke_eq!(
        seen(exposed != *starry, exposed.clone(), &wanted),
        wanted,
        "the_meter_exposure_replaces_the_sky_exposure"
    );
    let strong = refuse(
        session,
        "control",
        "environment",
        json!({"auto_exposure": {"strength": 2.0}}),
    )?;
    smoke_eq!(
        &strong["path"],
        &json!("$.args.auto_exposure.strength"),
        "metering_past_full_strength_is_refused"
    );
    environment(session, json!({"auto_exposure": {"enabled": false}}))?;
    Ok(())
}

fn sky_rules(session: &mut dyn SmokeSession) -> Checked<()> {
    environment(
        session,
        json!({
            "night_sky": {"stars": 0.0, "moon": 0.0, "cirrus": 0.0},
            "haze": {"enabled": false},
        }),
    )?;
    let sky = inspect(session, "sky")?;
    smoke_eq!(&sky["enabled"], &json!(true), "the_sky_reads_back");
    let mie = refuse(
        session,
        "control",
        "environment",
        json!({"sky": {"mie_g": 2.0}}),
    )?;
    smoke_eq!(
        &mie["path"],
        &json!("$.args.sky.mie_g"),
        "an_impossible_mie_g_is_refused"
    );
    let hours = refuse(
        session,
        "control",
        "environment",
        json!({"time_of_day": {"hours": 25.0}}),
    )?;
    smoke_eq!(
        &hours["path"],
        &json!("$.args.time_of_day.hours"),
        "hours_past_a_day_are_refused"
    );
    let off = environment(
        session,
        json!({"sky": {"enabled": false}, "time_of_day": {"enabled": false}}),
    )?;
    smoke_eq!(
        &off["rendering"]["sky_stats"]["tables"],
        &json!(0),
        "without_the_sky_nothing_builds"
    );
    camera(session, overview())?;
    Ok(())
}

fn horizons(session: &mut dyn SmokeSession) -> Checked<Value> {
    let current = inspect(session, "Environment")?;
    let sun = with(&current["sun"], json!({"sun_direction": [1.0, 0.22, 1.0]}));
    let shadows = with(
        &current["shadows"],
        json!({"levels": 1, "finest_texel": 0.005}),
    );
    let moved = environment(session, json!({"sun": sun, "shadows": shadows}))?;
    smoke_eq!(
        terrain(&moved["rendering"])["horizons"].clone(),
        json!(2),
        "the_terrain_casts_two_horizons"
    );
    smoke_eq!(
        terrain(&moved["rendering"])["lighting_uploads"].clone(),
        json!(0),
        "a_new_sun_uploads_nothing"
    );
    camera(session, view([-60.0, 320.0, 220.0], [0.0, 30.0, 0.0]))?;
    settle(session)?;
    let long = capture(session)?;
    environment(
        session,
        json!({"terrain": fast(), "terrain_shading": {"sun_shadows": false}}),
    )?;
    let flat = render(session)?;
    smoke_eq!(
        terrain(&flat)["horizons"].clone(),
        json!(0),
        "without_sun_shadows_no_horizons"
    );
    settle(session)?;
    let short = capture(session)?;
    let wanted = not(&long);
    smoke_eq!(
        seen(long != short, short.clone(), &wanted),
        wanted,
        "the_dome_casts_a_long_shadow"
    );
    Ok(current)
}

fn bounce(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut changed = environment(session, json!({ "terrain": fast(), "terrain_shading": {} }))?;
    let mut bounced = changed["rendering"].take();
    for _ in 0..SETTLE_FRAMES {
        if terrain(&bounced)["bounce"]["relit"] != 0 {
            break;
        }
        bounced = render(session)?;
    }
    smoke_eq!(
        terrain(&bounced)["bounce"]["relit"].clone(),
        json!(1),
        "the_terrain_probes_relight"
    );
    let wanted = json!({"probes": above(0)});
    smoke_eq!(
        seen(
            number(&terrain(&bounced)["bounce"]["probes"]) > 0.0,
            terrain(&bounced)["bounce"].clone(),
            &wanted,
        ),
        wanted,
        "probes_relight_around_the_camera"
    );
    settle(session)?;
    let relit = capture(session)?;
    environment(
        session,
        json!({"terrain": fast(), "terrain_shading": {"bounce": false}}),
    )?;
    let unlit = render(session)?;
    smoke_eq!(
        terrain(&unlit)["bounce"]["relit"].clone(),
        json!(0),
        "without_bounce_light_nothing_relights"
    );
    settle(session)?;
    let estimated = capture(session)?;
    let wanted = not(&relit);
    smoke_eq!(
        seen(relit != estimated, estimated.clone(), &wanted),
        wanted,
        "bounce_light_shows"
    );
    Ok(())
}

fn aerial(session: &mut dyn SmokeSession, current: &Value) -> Checked<()> {
    let clear_air = with(
        &current["sky_light"],
        json!({"aerial_start": 0.0, "aerial_density": 0.0}),
    );
    environment(
        session,
        json!({"terrain": fast(), "terrain_shading": {}, "sky_light": clear_air}),
    )?;
    settle(session)?;
    let clear = capture(session)?;
    let thick_air = with(
        &current["sky_light"],
        json!({"aerial_start": 0.0, "aerial_density": 0.004}),
    );
    environment(session, json!({ "sky_light": thick_air }))?;
    settle(session)?;
    let hazed = capture(session)?;
    let wanted = not(&clear);
    smoke_eq!(
        seen(clear != hazed, hazed.clone(), &wanted),
        wanted,
        "air_hazes_the_view"
    );
    let restored = json!({
        "sun": current["sun"],
        "sky_light": current["sky_light"],
        "shadows": current["shadows"],
    });
    environment(session, restored)?;
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}
