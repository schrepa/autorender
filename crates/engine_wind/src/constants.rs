use engine_core::gpu_consts;

gpu_consts! {
    pub const GUST_SLOW_SHARE: f32 = 0.6;
    pub const GUST_SLOW_RATE: f32 = 0.9;
    pub const GUST_SLOW_TRAVEL: f32 = 0.05;
    pub const GUST_QUICK_SHARE: f32 = 0.4;
    pub const GUST_QUICK_RATE: f32 = 2.3;
    pub const GUST_QUICK_TRAVEL: f32 = 0.17;
    pub const GUST_QUICK_ACROSS: f32 = 0.31;
    pub const WIND_FULL_SPEED: f32 = 12.0;
    pub const WIND_SPAN: f32 = 10.0;
    pub const WIND_TRUNK_FLUTTER: f32 = 0.12;
    pub const WIND_TRUNK_RATE: f32 = 4.7;
    pub const WIND_BRANCH: f32 = 4.0;
    pub const WIND_BRANCH_RATE: f32 = 3.1;
    pub const WIND_BRANCH_RISE: f32 = 2.1;
    pub const WIND_FLUTTER: f32 = 0.2;
    pub const WIND_FLUTTER_RATE: f32 = 13.0;
    pub const WIND_SHAKE_WHOLE: f32 = 1.0;
    pub const WIND_SHAKE_STILL: f32 = 0.5;
}
