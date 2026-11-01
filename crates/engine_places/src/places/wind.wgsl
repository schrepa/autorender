const GUST_SLOW_SHARE: f32 = 0.6;
const GUST_SLOW_RATE: f32 = 0.9;
const GUST_SLOW_TRAVEL: f32 = 0.05;
const GUST_QUICK_SHARE: f32 = 0.4;
const GUST_QUICK_RATE: f32 = 2.3;
const GUST_QUICK_TRAVEL: f32 = 0.17;
const GUST_QUICK_ACROSS: f32 = 0.31;

fn wind_sway(wind: vec4<f32>, pixels: f32, root: vec3<f32>, height: f32,
        local: vec3<f32>, sway: f32) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn slot_sway(flags: u32) -> f32 {
    return 0.0;
}
