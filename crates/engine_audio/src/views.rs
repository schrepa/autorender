use bevy_ecs::prelude::*;
use engine_core::{Error, Operation, SceneIds};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::mixer::Mixer;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeviceRequest {
    available: bool,
}

pub(crate) struct AudioDevice;

impl Operation for AudioDevice {
    const NAME: &'static str = "audio_device";
    type Params = ResMut<'static, Mixer>;
    type Request = DeviceRequest;

    fn run(
        mut mixer: SystemParamItem<'_, '_, Self::Params>,
        request: DeviceRequest,
    ) -> Result<Value, Error> {
        mixer.set_available(request.available);
        Ok(json!({"available": request.available}))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VoicesRequest {}

pub(crate) struct AudioVoices;

impl Operation for AudioVoices {
    const NAME: &'static str = "audio_voices";
    type Params = (Res<'static, Mixer>, SceneIds<'static, 'static>);
    type Request = VoicesRequest;

    fn run(
        (mixer, ids): SystemParamItem<'_, '_, Self::Params>,
        _: VoicesRequest,
    ) -> Result<Value, Error> {
        Ok(json!({"voices": mixer.listed(&ids)}))
    }
}
