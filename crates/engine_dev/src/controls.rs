use crate::protocol::{
    BenchmarkInput, BenchmarkMode, CaptureRequest, ExplainQuestion, MAX_WATCHED, ProfileRequest,
    WatchRequest, bounds,
};
use crate::results::{DebugMissingAssetResult, EnvironmentResult, ParentResult, result};
use crate::{BrowserControl, EnvironmentControl, bounded};
use bevy_ecs::prelude::World;
use engine_core::{AssetLibrary, Camera, Engine, Mesh3d, Projection, glam::Vec3};
use engine_registry::{OperationScene, Registry};
use engine_scene::{SceneEnvironment, SceneInstance};
use engine_settings::{Error, ErrorCode, Expected};
use serde_json::Value;

impl BrowserControl {
    pub fn parse(name: &str, args: Value, paused: bool) -> Result<Self, Error> {
        let control: Self = crate::protocol::parse_named(name, args)?;
        control.validate().map_err(|error| error.within("$.args"))?;
        if control.needs_pause() && !paused {
            return Err(Error::new(ErrorCode::PauseRequired, "$")
                .expected(Expected::Paused)
                .got(name));
        }
        Ok(control)
    }

    pub fn validate(&self) -> Result<(), Error> {
        match self {
            Self::Benchmark {
                warmup,
                samples,
                mode,
                script,
            } => {
                bounded(
                    *warmup,
                    bounds::WARMUP,
                    ErrorCode::InvalidBenchmark,
                    "$.warmup",
                )?;
                bounded(
                    *samples,
                    bounds::SAMPLES,
                    ErrorCode::InvalidBenchmark,
                    "$.samples",
                )?;
                validated_script(mode, script, warmup + samples)?;
            }
            Self::Frames {
                count, interval_ms, ..
            } => {
                bounded(*count, bounds::FRAMES, ErrorCode::OutOfRange, "$.count")?;
                bounded(
                    *interval_ms,
                    bounds::FRAME_INTERVAL,
                    ErrorCode::OutOfRange,
                    "$.interval_ms",
                )?;
            }
            Self::Probe {
                alpha: Some(alpha), ..
            } => {
                validated_presentation_alpha(*alpha)?;
            }
            Self::Probe { alpha: None, .. } => {}
            Self::Game { name, .. } => {
                let valid = name.len() <= 64
                    && name.starts_with(|first: char| first.is_ascii_lowercase())
                    && name.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'
                    });
                if !valid {
                    return Err(Error::new(ErrorCode::InvalidName, "$.name")
                        .expected(Expected::Id)
                        .at_most(64)
                        .got(name));
                }
            }
            Self::Reload {
                expected_revision,
                revision,
                ..
            } => {
                for (path, value) in [
                    ("$.expected_revision", expected_revision),
                    ("$.revision", revision),
                ] {
                    let decimal = !value.is_empty()
                        && value.bytes().all(|byte| byte.is_ascii_digit())
                        && (value == "0" || !value.starts_with('0'))
                        && value.parse::<u64>().is_ok();
                    if !decimal {
                        return Err(Error::new(ErrorCode::InvalidRevision, path)
                            .expected(Expected::Integer)
                            .got(value));
                    }
                }
            }
            Self::Present { .. }
            | Self::Culling { .. }
            | Self::Environment(_)
            | Self::Camera { .. }
            | Self::Step { .. }
            | Self::Input { .. }
            | Self::Render { .. }
            | Self::Pause { .. }
            | Self::Resume { .. }
            | Self::DebugPanic { .. }
            | Self::DebugFault { .. }
            | Self::DebugDeviceLoss { .. }
            | Self::DebugMissingAsset { .. }
            | Self::Parent { .. }
            | Self::Jobs { .. } => {}
        }
        Ok(())
    }
}

fn validated_script(
    mode: &BenchmarkMode,
    script: &[BenchmarkInput],
    frames: u32,
) -> Result<(), Error> {
    if matches!(mode, BenchmarkMode::Render | BenchmarkMode::Present) && !script.is_empty() {
        return Err(Error::new(ErrorCode::InvalidBenchmark, "$.script").expected(Expected::Absent));
    }
    for (index, entry) in script.iter().enumerate() {
        if entry.at >= frames {
            return Err(
                Error::new(ErrorCode::InvalidBenchmark, format!("$.script[{index}].at"))
                    .at_most(u64::from(frames - 1))
                    .got(entry.at),
            );
        }
        BrowserControl::parse("input", Value::Object(entry.input.clone()), true).map_err(
            |error| Error {
                path: format!("$.script[{index}].input"),
                ..error
            },
        )?;
    }
    Ok(())
}

impl CaptureRequest {
    pub fn validate(&self) -> Result<(), Error> {
        if let Some(alpha) = self.alpha {
            validated_presentation_alpha(alpha)?;
        }
        Ok(())
    }
}

impl ProfileRequest {
    pub fn validate(&self) -> Result<(), Error> {
        bounded(
            self.frames,
            bounds::FRAMES,
            ErrorCode::OutOfRange,
            "$.frames",
        )?;
        bounded(
            self.interval_ms,
            bounds::FRAME_INTERVAL,
            ErrorCode::OutOfRange,
            "$.interval_ms",
        )?;
        page_limit(self.top, "$.top")
    }
}

impl ExplainQuestion {
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            Self::Frame { top, .. } => page_limit(*top, "$.top"),
            Self::Entity { .. } | Self::Scene {} => Ok(()),
        }
    }
}

impl WatchRequest {
    pub fn validate(&self) -> Result<(), Error> {
        bounded(
            self.frames,
            bounds::FRAMES,
            ErrorCode::OutOfRange,
            "$.frames",
        )?;
        bounded(
            self.interval_ms,
            bounds::FRAME_INTERVAL,
            ErrorCode::OutOfRange,
            "$.interval_ms",
        )?;
        if self.stats.is_empty() && self.entities.is_empty() {
            return Err(Error::new(ErrorCode::InvalidRequest, "$")
                .expected(Expected::NonEmpty)
                .variants(["stats", "entities"]));
        }
        for (field, count) in [
            ("$.stats", self.stats.len()),
            ("$.entities", self.entities.len()),
        ] {
            if count > MAX_WATCHED {
                return Err(Error::new(ErrorCode::OutOfRange, field)
                    .expected(Expected::Array)
                    .at_most(MAX_WATCHED as u64)
                    .got(count));
            }
        }
        for (index, stat) in self.stats.iter().enumerate() {
            let valid = !stat.is_empty()
                && stat.len() <= 128
                && stat.bytes().all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.".contains(&byte)
                });
            if !valid {
                return Err(
                    Error::new(ErrorCode::InvalidName, format!("$.stats[{index}]"))
                        .expected(Expected::Path)
                        .at_most(128)
                        .got(stat),
                );
            }
        }
        Ok(())
    }
}

pub(crate) fn page_limit(value: usize, path: &str) -> Result<(), Error> {
    if (1..=bounds::MAX_PAGE_SIZE).contains(&value) {
        return Ok(());
    }
    Err(Error::new(ErrorCode::InvalidPage, path)
        .range(1.0, bounds::MAX_PAGE_SIZE as f64)
        .got(value))
}

pub fn validated_presentation_alpha(alpha: f64) -> Result<f32, Error> {
    bounded(
        alpha,
        bounds::ALPHA,
        ErrorCode::InvalidPresentationAlpha,
        "$.alpha",
    )
    .map(|alpha| alpha as f32)
}

pub fn validated_camera(
    position: [f32; 3],
    target: [f32; 3],
    vertical_fov_radians: f32,
    near: f32,
    far: f32,
) -> Result<Camera, Error> {
    let camera = Camera {
        position: Vec3::from_array(position),
        target: Vec3::from_array(target),
        near,
        far,
        projection: Projection::Perspective {
            vertical_fov_radians,
        },
    };
    camera.check().map_err(|invalid| Error {
        code: ErrorCode::InvalidCamera,
        ..invalid
    })?;
    if !camera.view_projection(1.0).is_finite() {
        return Err(Error::new(ErrorCode::InvalidCamera, "$").expected(Expected::Finite));
    }
    Ok(camera)
}

pub enum OperationCall {
    Brick { name: &'static str, request: Value },
    Environment(EnvironmentControl),
    MissingAsset { entity: String },
    Parent { entity: String, parent: String },
}

impl OperationCall {
    pub fn apply(
        &self,
        engine: &mut Engine,
        scene: Option<&SceneInstance>,
    ) -> Result<Value, Error> {
        match self {
            Self::Brick { name, request } => {
                crate::bricks::run(engine.app.world_mut(), name, request.clone())
            }
            Self::Environment(control) => {
                environment(engine.app.world_mut(), control).map_err(|error| {
                    if error.path.starts_with("$.") || error.path.starts_with("$[") {
                        error.within("$.args")
                    } else {
                        error
                    }
                })
            }
            Self::MissingAsset { entity } => {
                missing_asset(engine, scene, entity).map_err(|error| error.within("$.args"))
            }
            Self::Parent { entity, parent } => parented(engine, scene, (entity, parent)),
        }
    }
}

fn environment(world: &mut World, control: &EnvironmentControl) -> Result<Value, Error> {
    let current = |world: &World| {
        serde_json::to_value(SceneEnvironment::current(world)).expect("environment_serializes")
    };
    let before = current(world);
    let environment = if control.merge {
        let mut merged = serde_json::Map::new();
        for (key, value) in &control.sections {
            let mut section = before[key.as_str()].clone();
            overlay(&mut section, value.clone());
            merged.insert(key.clone(), section);
        }
        engine_settings::parse_json::<SceneEnvironment>(&Value::Object(merged).to_string())?
    } else {
        stays_on(world, control, &before)?;
        control.sections()
    };
    let skipped = environment.apply_sections(world, "$")?;
    let after = current(world);
    let named = environment.ordered_keys(world);
    let pick = |all: &Value| {
        named
            .iter()
            .map(|key| (key.clone(), all[key.as_str()].clone()))
            .collect::<serde_json::Map<_, _>>()
    };
    Ok(result::environment(EnvironmentResult {
        temporary: true,
        environment_changed: named.clone(),
        before: pick(&before),
        after: pick(&after),
        skipped,
    }))
}

fn stays_on(world: &World, control: &EnvironmentControl, before: &Value) -> Result<(), Error> {
    let Some(registry) = world.get_resource::<Registry>() else {
        return Ok(());
    };
    for (key, value) in &control.sections {
        let unnamed = value
            .as_object()
            .is_some_and(|fields| !fields.contains_key("enabled"));
        let off_by_default = registry
            .section(key)
            .is_some_and(|section| section.default_value()["enabled"] == false);
        if unnamed && off_by_default && before[key.as_str()]["enabled"] == true {
            return Err(
                Error::new(ErrorCode::MissingField, format!("$.{key}.enabled"))
                    .expected(Expected::Present),
            );
        }
    }
    Ok(())
}

fn overlay(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(base), Value::Object(patch)) => {
            for (key, value) in patch {
                overlay(base.entry(key).or_insert(Value::Null), value);
            }
        }
        (base, patch) => *base = patch,
    }
}

fn parented(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    (entity, parent): (&str, &str),
) -> Result<Value, Error> {
    let refused = || Error::new(ErrorCode::CollisionInvalidTransform, "$");
    let world = engine.app.world_mut();
    let scene = scene
        .map(SceneInstance::operation_scene)
        .or_else(|| world.get_resource::<OperationScene>().cloned())
        .ok_or_else(refused)?;
    let (Some(child), Some(parent)) = (scene.entity(world, entity), scene.entity(world, parent))
    else {
        return Err(refused());
    };
    engine_core::set_parent(world, child, Some(parent)).map_err(|_| refused())?;
    Ok(result::parent(ParentResult { temporary: true }))
}

fn missing_asset(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    entity: &str,
) -> Result<Value, Error> {
    let scene = scene.ok_or_else(|| {
        Error::new(ErrorCode::NoScene, "$.entity")
            .expected(Expected::Entity)
            .got(entity)
    })?;
    let world = engine.app.world_mut();
    let mesh = crate::inspect::authored_entity(world, scene, entity)
        .and_then(|entity| world.get::<Mesh3d>(entity))
        .map(|mesh3d| mesh3d.mesh)
        .ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Entity)
                .got(entity)
        })?;
    let entities = world.try_query::<&Mesh3d>().map_or(0, |mut meshes| {
        meshes
            .iter(world)
            .filter(|mesh3d| mesh3d.mesh == mesh)
            .count()
    });
    let mut assets = world.get_resource_mut::<AssetLibrary>().ok_or_else(|| {
        Error::new(ErrorCode::MissingResource, "$")
            .expected(Expected::Present)
            .got("AssetLibrary")
    })?;
    assets.remove_mesh(mesh);
    Ok(result::debug_missing_asset(DebugMissingAssetResult {
        temporary: true,
        entities,
    }))
}
