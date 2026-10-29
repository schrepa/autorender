use bevy_ecs::prelude::Entity;
use engine_core::smoke::{SmokeResult, SmokeSession};
use engine_core::{smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::{
    Checked, SETTLE_FRAMES, WAIT_FRAMES, above, camera, capture, environment, fast, height_at,
    heights, inspect, not, number, open, overview, refuse, render, renders_until, seen, settle,
    view, with,
};

const SIDE: usize = 513;

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let (fixture, _) = open(session)?;
    hydrology(session, fixture.ground)?;
    lakes(session)?;
    water(session)?;
    clouds(session)?;
    clouds_around(session)?;
    let starry = sky(session)?;
    exposure(session, &starry)?;
    sky_rules(session)?;
    settle(session)?;
    let (current, atmosphere) = horizons(session)?;
    bounce(session)?;
    aerial(session, &current, &atmosphere)?;
    let (captured, rendering) = renders_until(session, SETTLE_FRAMES, |rendering| {
        number(&rendering["reflection_probes"]["ready"]) > 0.0
    })?;
    smoke_check!(
        captured,
        "a_probe_captures",
        &rendering["reflection_probes"],
        &json!({"ready": above(0)})
    );
    Ok(())
}

fn hydrology(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let bakes = seen(session, ground).bakes;
    smoke_check!(
        !bakes.baking,
        "the_bake_is_done_once_tiles_load",
        &bakes.baking,
        &false
    );
    let water = bakes.hydrology;
    smoke_check!(
        water.is_some_and(|water| water.samples == SIDE * SIDE),
        "the_bake_reads_its_level",
        &water,
        &json!({ "samples": SIDE * SIDE })
    );
    smoke_check!(
        water.is_some_and(|water| water.channel_samples > 0 && water.max_area > 20_000.0),
        "water_gathers_into_channels",
        &water,
        &json!({"channel_samples": above(0), "max_area": above(20_000)})
    );
    let watered = height_at(session, [-300.0, -300.0])?;
    smoke_check!(
        number(&watered["water"]["flow"]) > 0.0 && number(&watered["water"]["wetness"]).is_finite(),
        "queries_report_water",
        &watered,
        &json!({"water": {"flow": above(0), "wetness": "finite"}})
    );
    let terrains = inspect(session, "Terrains")?;
    smoke_eq!(
        &terrains["falls"],
        &json!(water.map(|water| water.falls)),
        "falls_spawn_as_found"
    );
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
    smoke_check!(
        !lakes.is_empty() && lakes.iter().all(standing),
        "water_stands_in_lakes",
        &lakes,
        &json!({"standing_water": {"depth": above(0)}})
    );
    Ok(())
}

fn water(session: &mut dyn SmokeSession) -> Checked<()> {
    let settings = inspect(session, "Water")?;
    camera(session, overview())?;
    let wet = environment(session, json!({ "water": settings }))?;
    smoke_check!(
        number(&wet["rendering"]["water"]["leaves"]) > 0.0,
        "water_draws",
        &wet["rendering"]["water"],
        &json!({"leaves": above(0)})
    );
    let with_water = capture(session)?;
    environment(
        session,
        json!({"water": with(&settings, json!({"enabled": false}))}),
    )?;
    let without_water = capture(session)?;
    smoke_check!(
        with_water != without_water,
        "water_shades_the_view",
        &without_water,
        &not(&with_water)
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
    let clouded = environment(session, json!({ "clouds": layer() }))?;
    smoke_eq!(
        &clouded["rendering"]["clouds"]["rows"],
        &json!(256),
        "exact_frames_march_the_whole_dome"
    );
    let read = inspect(session, "Clouds")?;
    smoke_check!(
        (number(&read["coverage"]) - 0.6).abs() < 1e-6,
        "coverage_reads_back",
        &read["coverage"],
        &json!(0.6)
    );
    let cloudy = capture(session)?;
    smoke_check!(
        clear != cloudy,
        "clouds_show_in_the_sky",
        &cloudy,
        &not(&clear)
    );
    camera(session, eye([0.0, 400.0, 1.0], [0.0, 30.0, 0.0], 4000.0))?;
    let shaded = capture(session)?;
    environment(
        session,
        json!({"clouds": with(&layer(), json!({"shadows": 0.0}))}),
    )?;
    let unshaded = capture(session)?;
    smoke_check!(
        shaded != unshaded,
        "clouds_shade_the_ground",
        &unshaded,
        &not(&shaded)
    );
    Ok(())
}

fn clouds_around(session: &mut dyn SmokeSession) -> Checked<()> {
    camera(
        session,
        eye([0.0, 1900.0, 1.0], [400.0, 0.0, 300.0], 8000.0),
    )?;
    let over = environment(session, json!({ "clouds": layer() }))?;
    smoke_check!(
        number(&over["rendering"]["clouds"]["view_pixels"]) > 0.0,
        "the_view_marches_its_clouds",
        &over["rendering"]["clouds"],
        &json!({"view_pixels": above(0)})
    );
    let below_view = capture(session)?;
    let domed = environment(
        session,
        json!({"clouds": with(&layer(), json!({"in_view": false}))}),
    )?;
    smoke_eq!(
        &domed["rendering"]["clouds"]["view_pixels"],
        &json!(0),
        "off_the_view_marches_none"
    );
    let dome_only = capture(session)?;
    smoke_check!(
        below_view != dome_only,
        "clouds_lie_below_the_camera",
        &dome_only,
        &not(&below_view)
    );
    let none = environment(
        session,
        json!({"clouds": with(&layer(), json!({"coverage": 0.0}))}),
    )?;
    smoke_eq!(
        &none["rendering"]["clouds"]["rows"],
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
    let first = environment(session, json!({"sky": {}, "time_of_day": {"hours": 12.0}}))?;
    smoke_eq!(
        &first["rendering"]["sky"]["tables"],
        &json!(3),
        "the_air_tables_and_the_sky_view_build"
    );
    let unchanged = render(session)?;
    smoke_eq!(
        &unchanged["sky"]["tables"],
        &json!(0),
        "unchanged_nothing_builds"
    );
    let noon = capture(session)?;
    smoke_check!(
        authored != noon,
        "the_physical_sky_replaces_the_authored_one",
        &noon,
        &not(&authored)
    );
    let step = environment(session, json!({"time_of_day": {"hours": 19.3}}))?;
    smoke_eq!(
        &step["rendering"]["sky"]["tables"],
        &json!(1),
        "the_sun_step_builds_the_sky_view_alone"
    );
    environment(session, json!({"time_of_day": {"hours": 23.0}}))?;
    let starry = capture(session)?;
    let stars = inspect(session, "NightSky")?;
    smoke_eq!(
        &stars["stars"],
        &json!(1.0),
        "the_night_sky_shows_its_stars"
    );
    let haze = inspect(session, "Haze")?;
    smoke_eq!(&haze["enabled"], &json!(true), "haze_is_on");
    environment(
        session,
        json!({"night_sky": {"stars": 0.0, "moon": 0.0, "cirrus": 0.0}}),
    )?;
    let night = capture(session)?;
    smoke_check!(
        night != starry,
        "the_night_sky_shows",
        &night,
        &not(&starry)
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
    let mut metering = environment(session, json!({"auto_exposure": {}}))?;
    let mut exposure = metering["rendering"]["sky"]["exposure"].take();
    smoke_eq!(
        &exposure["active"],
        &json!(true),
        "frames_meter_their_exposure"
    );
    for _ in 0..WAIT_FRAMES {
        if exposure["metered"] != 0 {
            break;
        }
        let mut rendering = render(session)?;
        exposure = rendering["sky"]["exposure"].take();
    }
    let stops = number(&exposure["stops"]);
    smoke_check!(
        stops > 0.0 && stops <= 400.0,
        "the_night_exposure_keeps_within_its_limits",
        &exposure,
        &json!({"stops": {"above": 0, "at_most": 400}})
    );
    let exposed = capture(session)?;
    smoke_check!(
        exposed != *starry,
        "the_meter_exposure_replaces_the_sky_exposure",
        &exposed,
        &not(starry)
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
    let sky = inspect(session, "Sky")?;
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
        &off["rendering"]["sky"]["tables"],
        &json!(0),
        "without_the_sky_nothing_builds"
    );
    camera(session, overview())?;
    Ok(())
}

fn horizons(session: &mut dyn SmokeSession) -> Checked<(Value, Value)> {
    let current = inspect(session, "Environment")?;
    let atmosphere = with(
        &current["atmosphere"],
        json!({"sun_direction": [1.0, 0.22, 1.0]}),
    );
    let shadows = with(
        &current["shadows"],
        json!({"levels": 1, "finest_texel": 0.005}),
    );
    let moved = environment(
        session,
        json!({"atmosphere": atmosphere, "shadows": shadows}),
    )?;
    smoke_eq!(
        &moved["rendering"]["terrain"]["horizons"],
        &json!(2),
        "the_terrain_casts_two_horizons"
    );
    smoke_eq!(
        &moved["rendering"]["terrain"]["lighting_uploads"],
        &json!(0),
        "a_new_sun_uploads_nothing"
    );
    camera(session, view([-60.0, 320.0, 220.0], [0.0, 30.0, 0.0]))?;
    settle(session)?;
    let long = capture(session)?;
    let flat = environment(
        session,
        json!({"terrain": with(&fast(), json!({"sun_shadows": false}))}),
    )?;
    smoke_eq!(
        &flat["rendering"]["terrain"]["horizons"],
        &json!(0),
        "without_sun_shadows_no_horizons"
    );
    settle(session)?;
    let short = capture(session)?;
    smoke_check!(
        long != short,
        "the_dome_casts_a_long_shadow",
        &short,
        &not(&long)
    );
    Ok((current, atmosphere))
}

fn bounce(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut changed = environment(session, json!({ "terrain": fast() }))?;
    let mut bounced = changed["rendering"].take();
    for _ in 0..SETTLE_FRAMES {
        if bounced["terrain"]["bounce"]["relit"] != 0 {
            break;
        }
        bounced = render(session)?;
    }
    smoke_eq!(
        &bounced["terrain"]["bounce"]["relit"],
        &json!(1),
        "the_terrain_probes_relight"
    );
    smoke_check!(
        number(&bounced["terrain"]["bounce"]["probes"]) > 0.0,
        "probes_relight_around_the_camera",
        &bounced["terrain"]["bounce"],
        &json!({"probes": above(0)})
    );
    settle(session)?;
    let relit = capture(session)?;
    let unlit = environment(
        session,
        json!({"terrain": with(&fast(), json!({"bounce": false}))}),
    )?;
    smoke_eq!(
        &unlit["rendering"]["terrain"]["bounce"]["relit"],
        &json!(0),
        "without_bounce_light_nothing_relights"
    );
    settle(session)?;
    let estimated = capture(session)?;
    smoke_check!(
        relit != estimated,
        "bounce_light_shows",
        &estimated,
        &not(&relit)
    );
    Ok(())
}

fn aerial(session: &mut dyn SmokeSession, current: &Value, atmosphere: &Value) -> Checked<()> {
    let clear_air = with(
        atmosphere,
        json!({"aerial_start": 0.0, "aerial_density": 0.0}),
    );
    environment(session, json!({"terrain": fast(), "atmosphere": clear_air}))?;
    settle(session)?;
    let clear = capture(session)?;
    let thick_air = with(
        atmosphere,
        json!({"aerial_start": 0.0, "aerial_density": 0.004}),
    );
    environment(session, json!({ "atmosphere": thick_air }))?;
    settle(session)?;
    let hazed = capture(session)?;
    smoke_check!(clear != hazed, "air_hazes_the_view", &hazed, &not(&clear));
    environment(
        session,
        json!({"atmosphere": current["atmosphere"], "shadows": current["shadows"]}),
    )?;
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}
