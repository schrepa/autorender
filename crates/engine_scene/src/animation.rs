use engine_settings::ErrorCode;
use std::sync::Arc;

use engine_core::{
    AnimatedProperty, AnimationChannel, AnimationClip, KeyInterpolation, Keyframes, glam,
};
use serde::{Deserialize, Serialize};

use crate::{SceneError, SceneFiles, files::valid_uri};

pub(crate) struct PreparedPlayer {
    pub clip: usize,
    pub targets: Vec<usize>,
    pub speed: f32,
    pub looping: bool,
    pub time: f32,
}

pub(crate) struct PreparedSkin {
    pub skin: usize,
    pub joints: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneAnimation {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
    )]
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    pub targets: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<Vec<SceneChannel>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneChannel {
    pub target: u32,
    pub property: SceneProperty,
    #[serde(default)]
    pub interpolation: SceneInterpolation,
    pub times: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneProperty {
    Translation,
    Rotation,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneInterpolation {
    Step,
    #[default]
    Linear,
    CubicSpline,
}

impl engine_settings::Setting for ScenePlayer {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ScenePlayer {
    pub clip: String,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_core::AnimationPlayer::SPEED))
    )]
    pub speed: f32,
    #[serde(rename = "loop", default = "yes", skip_serializing_if = "is_yes")]
    pub looping: bool,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub time: f32,
}

fn one() -> f32 {
    1.0
}
fn is_one(value: &f32) -> bool {
    *value == 1.0
}
fn yes() -> bool {
    true
}
fn is_yes(value: &bool) -> bool {
    *value
}
fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneSkin {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
    )]
    pub id: String,
    pub joints: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inverse_binds: Vec<[f32; 16]>,
}

impl SceneSkin {
    pub(crate) fn inverse_binds(&self, path: &str) -> Result<Arc<[glam::Mat4]>, SceneError> {
        if self.joints.is_empty() {
            return Err(SceneError::new(
                ErrorCode::InvalidSkin,
                format!("{path}.joints"),
                "a skin needs at least one joint",
            ));
        }
        if self.inverse_binds.is_empty() {
            return Ok(vec![glam::Mat4::IDENTITY; self.joints.len()].into());
        }
        if self.inverse_binds.len() != self.joints.len() {
            return Err(SceneError::new(
                ErrorCode::InvalidSkin,
                format!("{path}.inverse_binds"),
                format!(
                    "a skin of {} joints needs as many inverse bind matrices, or none",
                    self.joints.len()
                ),
            ));
        }
        self.inverse_binds
            .iter()
            .enumerate()
            .map(|(index, values)| {
                let mut matrix = glam::Mat4::from_cols_array(values);
                let row = matrix.row(3);
                if !matrix.is_finite() || (row - glam::Vec4::W).abs().max_element() > 0.001 {
                    return Err(SceneError::new(
                        ErrorCode::InvalidSkin,
                        format!("{path}.inverse_binds[{index}]"),
                        "inverse bind matrices must be finite and affine",
                    ));
                }
                matrix.x_axis.w = 0.0;
                matrix.y_axis.w = 0.0;
                matrix.z_axis.w = 0.0;
                matrix.w_axis.w = 1.0;
                Ok(matrix)
            })
            .collect()
    }
}

impl SceneAnimation {
    pub(crate) fn clip(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
    ) -> Result<AnimationClip, SceneError> {
        let invalid = |field: &str, message: String| {
            SceneError::new(
                ErrorCode::InvalidAnimation,
                format!("{path}{field}"),
                message,
            )
        };
        let channels = match (&self.channels, &self.uri) {
            (Some(channels), None) => channels
                .iter()
                .enumerate()
                .map(|(index, channel)| {
                    channel
                        .channel()
                        .map_err(|message| invalid(&format!(".channels[{index}]"), message))
                })
                .collect::<Result<Vec<_>, _>>()?,
            (None, Some(uri)) => {
                let uri_path = format!("{path}.uri");
                valid_uri(uri, &uri_path)?;
                match files {
                    Some(files) => files.animation(uri, &uri_path)?,
                    None => vec![still_channel()],
                }
            }
            _ => {
                return Err(invalid(
                    "",
                    "an animation has either channels or a uri".into(),
                ));
            }
        };
        if let Some(channel) = channels
            .iter()
            .find(|channel| channel.target() as usize >= self.targets.len())
        {
            return Err(invalid(
                "",
                format!(
                    "channel target {} is beyond the {} targets",
                    channel.target(),
                    self.targets.len()
                ),
            ));
        }
        AnimationClip::new(channels).map_err(|error| invalid("", error.to_string()))
    }
}

fn still_channel() -> AnimationChannel {
    AnimationChannel::new(
        0,
        AnimatedProperty::Scale,
        KeyInterpolation::Step,
        vec![0.0],
        Keyframes::Vectors(vec![glam::Vec3::ONE].into_boxed_slice()),
    )
    .expect("a still key is valid")
}

impl SceneChannel {
    fn channel(&self) -> Result<AnimationChannel, String> {
        let property = match self.property {
            SceneProperty::Translation => AnimatedProperty::Translation,
            SceneProperty::Rotation => AnimatedProperty::Rotation,
            SceneProperty::Scale => AnimatedProperty::Scale,
        };
        let interpolation = match self.interpolation {
            SceneInterpolation::Step => KeyInterpolation::Step,
            SceneInterpolation::Linear => KeyInterpolation::Linear,
            SceneInterpolation::CubicSpline => KeyInterpolation::CubicSpline,
        };
        let keyframes = if property == AnimatedProperty::Rotation {
            let (values, rest) = self.values.as_chunks::<4>();
            if !rest.is_empty() {
                return Err("rotation values come in fours".into());
            }
            Keyframes::Rotations(
                values
                    .iter()
                    .map(|value| glam::Quat::from_array(*value))
                    .collect(),
            )
        } else {
            let (values, rest) = self.values.as_chunks::<3>();
            if !rest.is_empty() {
                return Err("translation and scale values come in threes".into());
            }
            Keyframes::Vectors(
                values
                    .iter()
                    .map(|value| glam::Vec3::from_array(*value))
                    .collect(),
            )
        };
        AnimationChannel::new(
            self.target,
            property,
            interpolation,
            self.times.clone(),
            keyframes,
        )
        .map_err(|error| error.to_string())
    }
}
