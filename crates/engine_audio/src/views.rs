use bevy_ecs::prelude::*;
use engine_core::{Error, Operation, SceneIds};
use serde::{Deserialize, Serialize};

use crate::mixer::{Mixer, VoiceView};

#[derive(Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeviceRequest {
    available: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct VoicesReply {
    voices: Vec<VoiceView>,
}

pub(crate) struct AudioDevice;

impl Operation for AudioDevice {
    const NAME: &'static str = "audio_device";
    type Params = ResMut<'static, Mixer>;
    type Request = DeviceRequest;
    type Reply = DeviceRequest;

    fn run(
        mut mixer: SystemParamItem<'_, '_, Self::Params>,
        request: DeviceRequest,
    ) -> Result<DeviceRequest, Error> {
        mixer.set_available(request.available);
        Ok(request)
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct VoicesRequest {}

pub(crate) struct AudioVoices;

impl Operation for AudioVoices {
    const NAME: &'static str = "audio_voices";
    type Params = (Res<'static, Mixer>, SceneIds<'static, 'static>);
    type Request = VoicesRequest;
    type Reply = VoicesReply;

    fn run(
        (mixer, ids): SystemParamItem<'_, '_, Self::Params>,
        _: VoicesRequest,
    ) -> Result<VoicesReply, Error> {
        Ok(VoicesReply {
            voices: mixer.listed(&ids),
        })
    }
}
