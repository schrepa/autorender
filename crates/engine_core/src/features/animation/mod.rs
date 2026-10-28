mod skin;

pub use skin::*;

use bevy_ecs::prelude::{Component, Entity, Local, Query, Res, SystemSet};
use bevy_ecs::schedule::IntoScheduleConfigs;
use engine_settings::Rule;

use crate::{AnimationSample, AssetLibrary, ClipHandle, FixedTime, Transform};

#[derive(Component, Debug, Clone, PartialEq)]
pub struct AnimationPlayer {
    pub clip: ClipHandle,
    pub targets: Vec<Option<Entity>>,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
}

impl AnimationPlayer {
    pub const SPEED: Rule = Rule::Range(-1000.0, 1000.0);

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

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnimationSystems {
    Animate,
}

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
        let targets = &player.targets;
        for run in samples.chunk_by(|a, b| a.0 == b.0) {
            let Some(Some(target)) = targets.get(run[0].0 as usize) else {
                continue;
            };
            let Ok(mut transform) = transforms.get_mut(*target) else {
                continue;
            };
            for &(_, sample) in run {
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
