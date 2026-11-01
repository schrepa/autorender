import package::generated::auto_exposure::ExposureState;

@group(0) @binding(0) var<uniform> exposure_state: ExposureState;

fn adapted(authored: f32) -> f32 {
    return exposure_state.exposure;
}
