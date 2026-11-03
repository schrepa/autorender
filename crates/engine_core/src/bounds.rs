use engine_settings::Rule;

pub const MAX_PAGE_SIZE: usize = 100;
pub const MAX_SCAN_ENTITIES: usize = 10_000;
pub const PAGE: Rule = Rule::Range(1.0, MAX_PAGE_SIZE as f64);
pub const MAX_POINTS: usize = 256;
pub const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);
pub const DISTANCE: Rule = Rule::Range(0.0, 2.0e6);
pub const EVENT_OFFSET: Rule = Rule::Range(0.0, 131_072.0);
pub const SWEEP_EXTENT: Rule = Rule::Range(0.0, 4096.0);
pub const SWEEP_SAMPLES: Rule = Rule::Range(2.0, 64.0);
pub const ALPHA: Rule = Rule::Range(0.0, 1.0);
pub const WARMUP: Rule = Rule::Range(0.0, 120.0);
pub const SAMPLES: Rule = Rule::Range(1.0, 240.0);
pub const TICKS: Rule = Rule::Range(1.0, crate::MAX_DEV_TICKS as f64);
pub const FRAMES: Rule = Rule::Range(1.0, 240.0);
pub const FRAME_INTERVAL: Rule = Rule::Range(1.0, 1000.0);
pub const AXIS: Rule = Rule::Range(-1.0, 1.0);
pub const TURN: Rule = Rule::Range(-1000.0, 1000.0);
pub const MAX_AXES: usize = crate::AxisId::COUNT as usize;
pub const MAX_CHANGES: usize = 256;
pub const MAX_VALUES: usize = 1024;
pub const PIXEL: Rule = Rule::Range(0.0, 16_384.0);

pub fn default_limit() -> usize {
    20
}
