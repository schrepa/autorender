const TOON_BANDS: f32 = 3.0;
const TOON_SHADOW_EDGE: f32 = 0.5;
const TOON_RIM: f32 = 0.15;
const TOON_SHARPEST: f32 = 1.0e-4;
const TOON_VISIBILITY: f32 = 0.25;
const TOON_SPECULAR: f32 = 1.0;
const TOON_NEAREST: f32 = 0.25;
const TOON_DARK: f32 = 1.0e-6;
const TOON_WIDEST: f32 = 0.5;
const TOON_UP: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
const TOON_HIGHLIGHTS: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 0.0);

fn banded(cosine: f32) -> f32 {
    return ceil(saturate(cosine) * TOON_BANDS) / TOON_BANDS;
}
