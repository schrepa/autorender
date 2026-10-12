//! Animation playback: players sample clips every fixed tick
//! and write their targets' local transforms, which interpolation then
//! presents smoothly between ticks.

mod skin;

pub use skin::*;

use bevy_ecs::prelude::{Component, Entity, Local, Query, Res, SystemSet};
use bevy_ecs::schedule::IntoScheduleConfigs;
use engine_settings::Rule;

use crate::{AnimationSample, AssetLibrary, ClipHandle, FixedTime, Transform};

/// Plays an animation clip on the entities its targets name.
///
/// Every fixed tick the player advances `time` by the tick times `speed`,
/// wrapping when `looping` and holding the last pose otherwise, and writes
/// each channel's sample into its target's local [`Transform`]. Target `i`
/// of the clip is `targets[i]`; `None` or a despawned entity skips its
/// channels. Give targets [`crate::InterpolateTransform`] for smooth
/// presentation between ticks (scene instantiation does).
#[derive(Component, Debug, Clone, PartialEq)]
pub struct AnimationPlayer {
    pub clip: ClipHandle,
    pub targets: Vec<Option<Entity>>,
    /// Seconds into the clip.
    pub time: f32,
    /// Playback rate: 1 is real time, 0 pauses, negative plays backward.
    pub speed: f32,
    pub looping: bool,
}

impl AnimationPlayer {
    /// The speeds players keep, as scenes author them and the development
    /// control sets them: up to a thousand times real time, either way.
    pub const SPEED: Rule = Rule::Range(-1000.0, 1000.0);

    /// A player of `clip` from its start at real time, looping.
    pub fn new(clip: ClipHandle, targets: Vec<Option<Entity>>) -> Self {
        Self {
            clip,
            targets,
            time: 0.0,
            speed: 1.0,
            looping: true,
        }
    }
}

/// Ordering of animation within [`crate::FixedSimulation`]: systems that
/// adjust animated poses run after [`AnimationSystems::Animate`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnimationSystems {
    Animate,
}

/// Advances every player one fixed tick and poses its targets.
pub(crate) fn animate(
    time: Res<FixedTime>,
    assets: Res<AssetLibrary>,
    mut players: Query<&mut AnimationPlayer>,
    mut transforms: Query<&mut Transform>,
    mut samples: Local<Vec<(u32, AnimationSample)>>,
) {
    for mut player in &mut players {
        let Some(clip) = assets.clip(player.clip) else {
            continue;
        };
        let duration = clip.duration();
        let mut next = player.time + time.delta_seconds * player.speed;
        if !next.is_finite() {
            next = 0.0;
        }
        next = if player.looping && duration > 0.0 {
            next.rem_euclid(duration)
        } else {
            crate::math::clamp(next, 0.0, duration)
        };
        if player.time != next {
            player.time = next;
        }
        samples.clear();
        clip.sample_channels(player.time, |channel, sample| {
            samples.push((channel.target(), sample));
        });
        // Channels of one target in a row share its lookup.
        let targets = &player.targets;
        for run in samples.chunk_by(|a, b| a.0 == b.0) {
            let Some(Some(target)) = targets.get(run[0].0 as usize) else {
                continue;
            };
            let Ok(mut transform) = transforms.get_mut(*target) else {
                continue;
            };
            for &(_, sample) in run {
                // Unchanged values leave the transform unchanged: held and
                // paused poses cost propagation nothing.
                match sample {
                    AnimationSample::Translation(translation) => {
                        if transform.translation != translation {
                            transform.translation = translation;
                        }
                    }
                    AnimationSample::Rotation(rotation) => {
                        if transform.rotation != rotation {
                            transform.rotation = rotation;
                        }
                    }
                    AnimationSample::Scale(scale) => {
                        if transform.scale != scale {
                            transform.scale = scale;
                        }
                    }
                }
            }
        }
    }
}

/// Clip players posing their targets, and skins:
/// [`crate::Feature::Animation`].
#[derive(Default)]
pub struct AnimationPlugin;

impl bevy_app::Plugin for AnimationPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        crate::install(app, crate::Feature::Animation);
        app.add_systems(
            crate::FixedSimulation,
            animate.in_set(AnimationSystems::Animate),
        )
        .configure_sets(
            crate::FixedSimulation,
            AnimationSystems::Animate.in_set(crate::FixedSimulationSystems::Pose),
        );
    }
}
