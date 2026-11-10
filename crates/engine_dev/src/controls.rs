use crate::protocol::{BenchmarkInput, BenchmarkMode, CaptureRequest, ProfileRequest, bounds};
use crate::results::{EnvironmentResult, ParentResult, result};
use crate::{BrowserControl, EnvironmentControl, bounded};
use bevy_ecs::prelude::World;
use engine_core::{Camera, Engine, Projection, Transform, glam::Vec3};
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
            Self::Present { alpha } => {
                validated_presentation_alpha(*alpha)?;
            }
            Self::Environment(_)
            | Self::Camera { .. }
            | Self::Step { .. }
            | Self::Input { .. }
            | Self::Render { .. }
            | Self::Pause { .. }
            | Self::Resume { .. }
            | Self::Parent { .. } => {}
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
) -> Result<(Transform, Camera), Error> {
    let lens = Camera {
        near,
        far,
        projection: Projection::Perspective {
            vertical_fov_radians,
        },
    };
    lens.check()?;
    let (position, target) = (Vec3::from_array(position), Vec3::from_array(target));
    let reach = Camera::REACH;
    for (name, point) in [("position", position), ("target", target)] {
        if !point.is_finite() || point.abs().max_element() > reach {
            return Err(Error::new(ErrorCode::InvalidCamera, format!("$.{name}")));
        }
    }
    if (target - position).length_squared() < Camera::MIN_DEPTH * Camera::MIN_DEPTH {
        return Err(Error::new(ErrorCode::InvalidCamera, "$.target"));
    }
    if !lens
        .view_projection(position, target - position, 1.0)
        .is_finite()
    {
        return Err(Error::new(ErrorCode::InvalidCamera, "$").expected(Expected::Finite));
    }
    Ok((Transform::looking_at(position, target), lens))
}

pub enum OperationCall {
    Brick { name: &'static str, request: Value },
    Environment(EnvironmentControl),
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
                engine_registry::run_operation(engine.app.world_mut(), name, request.clone())
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
            Self::Parent { entity, parent } => parented(engine, scene, (entity, parent)),
        }
    }
}

fn environment(world: &mut World, control: &EnvironmentControl) -> Result<Value, Error> {
    let current = |world: &World| -> Result<Value, Error> {
        let environment = SceneEnvironment::current(world)?;
        serde_json::to_value(environment).map_err(|error| {
            Error::new(ErrorCode::InvalidValue, "$.environment").got(error.to_string())
        })
    };
    let before = current(world)?;
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
    let after = current(world)?;
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
    let world = engine.app.world_mut();
    let scene = scene
        .map(SceneInstance::operation_scene)
        .or_else(|| world.get_resource::<OperationScene>().cloned())
        .ok_or_else(|| Error::new(ErrorCode::NoScene, "$"))?;
    let found = |id: &str, path: &str| {
        scene.entity(id).ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, path)
                .expected(Expected::Entity)
                .got(id)
        })
    };
    let (child, parent) = (found(entity, "$.entity")?, found(parent, "$.parent")?);
    engine_core::set_parent(world, child, Some(parent))?;
    Ok(result::parent(ParentResult { temporary: true }))
}
