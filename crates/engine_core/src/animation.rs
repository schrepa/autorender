//! Animation playback (decision 035): players sample clips every fixed tick
//! and write their targets' local transforms, which interpolation then
//! presents smoothly between ticks.

use bevy_ecs::prelude::{Component, Entity, Query, Res, SystemSet};

use crate::{AnimationSample, AssetLibrary, ClipHandle, FixedTime, Transform};

/// Plays an animation clip on the entities its targets name (decision 035).
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
        let at = player.time;
        for channel in clip.channels() {
            let Some(Some(target)) = player.targets.get(channel.target() as usize) else {
                continue;
            };
            let Ok(mut transform) = transforms.get_mut(*target) else {
                continue;
            };
            // Unchanged values leave the transform unchanged: held and
            // paused poses cost propagation nothing.
            match channel.sample(at) {
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
