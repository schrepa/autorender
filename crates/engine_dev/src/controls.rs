use crate::protocol::{
    CaptureRequest, ExplainQuestion, MAX_WATCHED, ProfileRequest, WatchRequest, bounds,
};
use crate::results::{DebugMissingAssetResult, result};
use crate::{BrowserControl, DevError, bounded};
use engine_core::{AssetLibrary, Camera, Engine, Mesh3d, glam::Vec3};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
use serde_json::Value;

impl BrowserControl {
    pub fn parse(name: &str, args: Value, paused: bool) -> Result<Self, DevError> {
        let control: Self = crate::protocol::parse_named(name, args)?;
        control.validate().map_err(DevError::in_args)?;
        if control.needs_pause() && !paused {
            return Err(DevError::new(
                ErrorCode::PauseRequired,
                "$",
                format!("pause the host before {name}"),
            ));
        }
        Ok(control)
    }

    pub fn validate(&self) -> Result<(), DevError> {
        match self {
            Self::Benchmark {
                warmup, samples, ..
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
                    return Err(DevError::new(
                        ErrorCode::InvalidName,
                        "$.name",
                        "game control names are 1 to 64 lowercase letters, digits and underscores",
                    ));
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
                        return Err(DevError::new(
                            ErrorCode::InvalidRevision,
                            path,
                            "revisions are unsigned decimal strings without leading zeros",
                        ));
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
            | Self::DebugMissingAsset { .. } => {}
        }
        Ok(())
    }
}

impl CaptureRequest {
    pub fn validate(&self) -> Result<(), DevError> {
        if let Some(alpha) = self.alpha {
            validated_presentation_alpha(alpha)?;
        }
        Ok(())
    }
}

impl ProfileRequest {
    pub fn validate(&self) -> Result<(), DevError> {
        bounded(self.frames, bounds::FRAMES, ErrorCode::OutOfRange, "$.frames")?;
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
    pub fn validate(&self) -> Result<(), DevError> {
        match self {
            Self::Frame { top, .. } => page_limit(*top, "$.top"),
            Self::Entity { .. } | Self::Scene {} => Ok(()),
        }
    }
}

impl WatchRequest {
    pub fn validate(&self) -> Result<(), DevError> {
        bounded(self.frames, bounds::FRAMES, ErrorCode::OutOfRange, "$.frames")?;
        bounded(
            self.interval_ms,
            bounds::FRAME_INTERVAL,
            ErrorCode::OutOfRange,
            "$.interval_ms",
        )?;
        if self.stats.is_empty() && self.entities.is_empty() {
            return Err(DevError::new(
                ErrorCode::InvalidRequest,
                "$",
                "watch names stats or entities",
            )
            .allowed(["stats", "entities"]));
        }
        for (field, count) in [
            ("$.stats", self.stats.len()),
            ("$.entities", self.entities.len()),
        ] {
            if count > MAX_WATCHED {
                return Err(
                    DevError::new(ErrorCode::OutOfRange, field, format!("{count} watched"))
                        .expected(format!("at most {MAX_WATCHED}")),
                );
            }
        }
        for (index, stat) in self.stats.iter().enumerate() {
            let valid = !stat.is_empty()
                && stat.len() <= 128
                && stat
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.".contains(&byte));
            if !valid {
                return Err(DevError::new(
                    ErrorCode::InvalidName,
                    format!("$.stats[{index}]"),
                    stat.clone(),
                )
                .expected("a dot path of lowercase stat names, at most 128 bytes"));
            }
        }
        Ok(())
    }
}

pub(crate) fn page_limit(value: usize, path: &str) -> Result<(), DevError> {
    if (1..=bounds::MAX_PAGE_SIZE).contains(&value) {
        return Ok(());
    }
    Err(
        DevError::new(ErrorCode::InvalidPage, path, value.to_string())
            .expected(format!("1..{}", bounds::MAX_PAGE_SIZE)),
    )
}

pub fn validated_presentation_alpha(alpha: f64) -> Result<f32, DevError> {
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
) -> Result<Camera, DevError> {
    let camera = Camera {
        position: Vec3::from_array(position),
        target: Vec3::from_array(target),
        vertical_fov_radians,
        near,
        far,
    };
    camera.validate().map_err(|invalid| {
        DevError::new(
            ErrorCode::InvalidCamera,
            format!("$.{}", invalid.field),
            invalid.rule,
        )
    })?;
    if !camera.view_projection(1.0).is_finite() {
        return Err(DevError::new(
            ErrorCode::InvalidCamera,
            "$",
            "camera must produce a finite projection",
        ));
    }
    Ok(camera)
}

pub enum OperationCall {
    Brick { name: &'static str, request: Value },
    MissingAsset { entity: String },
}

impl OperationCall {
    pub fn changed(&self) -> String {
        match self {
            Self::Brick { name, .. } => format!("{name} changed"),
            Self::MissingAsset { .. } => "asset removed".to_owned(),
        }
    }

    pub fn apply(
        &self,
        engine: &mut Engine,
        scene: Option<&SceneInstance>,
    ) -> Result<Value, DevError> {
        match self {
            Self::Brick { name, request } => {
                crate::bricks::run(engine.app.world_mut(), name, request.clone())
            }
            Self::MissingAsset { entity } => {
                missing_asset(engine, scene, entity).map_err(DevError::in_args)
            }
        }
    }
}

fn missing_asset(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    entity: &str,
) -> Result<Value, DevError> {
    let scene = scene.ok_or_else(|| {
        DevError::new(
            ErrorCode::NoScene,
            "$.entity",
            "the meshes it removes are authored entities'",
        )
    })?;
    let world = engine.app.world_mut();
    let mesh = crate::inspect::authored_entity(world, scene, entity)
        .and_then(|entity| world.get::<Mesh3d>(entity))
        .map(|mesh3d| mesh3d.mesh)
        .ok_or_else(|| {
            DevError::new(
                ErrorCode::UnknownEntity,
                "$.entity",
                format!("no entity {entity} drawing a mesh in the scene"),
            )
        })?;
    let entities = world.try_query::<&Mesh3d>().map_or(0, |mut meshes| {
        meshes
            .iter(world)
            .filter(|mesh3d| mesh3d.mesh == mesh)
            .count()
    });
    let mut assets = world.get_resource_mut::<AssetLibrary>().ok_or_else(|| {
        DevError::new(
            ErrorCode::MissingResource,
            "$",
            "AssetLibrary is not installed",
        )
    })?;
    assets.remove_mesh(mesh);
    Ok(result::debug_missing_asset(DebugMissingAssetResult {
        temporary: true,
        entities,
    }))
}
