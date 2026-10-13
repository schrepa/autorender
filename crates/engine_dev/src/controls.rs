use crate::protocol::bounds;
use crate::{BrowserControl, DevError, bounded};
use engine_core::{Camera, glam::Vec3};
use engine_settings::ErrorCode;

impl BrowserControl {
    /// `source`, a request naming control `op`, as hosts take it on a host
    /// `paused` or not: parsed, checked ([`Self::validate`]) and refused
    /// where it needs the host paused and it is not.
    pub fn parse(source: &str, op: &str, paused: bool) -> Result<Self, DevError> {
        if op == "environment" {
            // Its sections parse as a scene environment's, whose errors
            // keep their whole path (a tagged request's lose it).
            let mut request: serde_json::Value = engine_scene::parse_json(source)?;
            if let Some(fields) = request.as_object_mut() {
                for key in ["op", "version", "merge"] {
                    fields.remove(key);
                }
            }
            engine_scene::parse_json::<engine_scene::SceneEnvironment>(&request.to_string())?;
        }
        let control: Self = crate::parse_request(source)?;
        control.validate()?;
        if control.needs_pause() && !paused {
            return Err(DevError::new(
                ErrorCode::PauseRequired,
                "$",
                format!("pause the host before {op}"),
            ));
        }
        Ok(control)
    }

    /// Checks what deserialization does not: the ranges and revision
    /// strings of the controls the browser host runs (benchmark, frames,
    /// capture and reload), which it receives back validated with their
    /// defaults. Controls the engine runs check their values where they
    /// apply.
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
            Self::Capture { alpha, gpu, .. } => match alpha {
                Some(alpha) => {
                    validated_presentation_alpha(*alpha)?;
                }
                None if *gpu => {
                    return Err(DevError::new(
                        ErrorCode::InvalidRequest,
                        "$.gpu",
                        "a GPU-presented capture needs an alpha",
                    ));
                }
                None => {}
            },
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
            // Their handlers check what the rest carry.
            Self::Present { .. }
            | Self::Culling { .. }
            | Self::Environment(_)
            | Self::Terrain { .. }
            | Self::TerrainEdit { .. }
            | Self::TerrainUndo { .. }
            | Self::TerrainRedo { .. }
            | Self::Deform { .. }
            | Self::NavGo { .. }
            | Self::Animation { .. }
            | Self::WorldChanges { .. }
            | Self::Camera { .. }
            | Self::Step { .. }
            | Self::Input { .. }
            | Self::Render { .. }
            | Self::Pause { .. }
            | Self::Resume { .. }
            | Self::DebugPanic { .. }
            | Self::DebugFault { .. } => {}
        }
        Ok(())
    }
}

/// Validates a temporary render fraction without mutating clocks or gameplay.
/// Fractions include both fixed-tick endpoints; no clamping hides bad requests.
pub fn validated_presentation_alpha(alpha: f64) -> Result<f32, DevError> {
    bounded(
        alpha,
        bounds::ALPHA,
        ErrorCode::InvalidPresentationAlpha,
        "$.alpha",
    )
    .map(|alpha| alpha as f32)
}

/// Validates a temporary camera without invoking fallback behavior or mutating
/// gameplay: the camera's rules, which scenes author too. Shared by browser
/// controls and native contract checks.
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
