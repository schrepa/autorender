struct ShadowPiece {
    light: array<vec4<f32>, 3>,
    state: vec4<f32>,
    level_state: vec4<f32>,
    atlas: vec4<f32>,
    windows: array<vec4<f32>, 20>,
    bands: array<vec4<f32>, 20>,
}
