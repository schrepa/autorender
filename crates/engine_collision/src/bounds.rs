use engine_core::Rule;

pub const MAX_PAGE_SIZE: usize = 100;
pub const MAX_SCAN_ENTITIES: usize = 10_000;
pub const PAGE: Rule = Rule::Range(1.0, MAX_PAGE_SIZE as f64);
pub const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);
pub const DISTANCE: Rule = Rule::Range(0.0, 2.0e6);

pub fn default_limit() -> usize {
    20
}
