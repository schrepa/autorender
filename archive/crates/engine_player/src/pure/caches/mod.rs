mod fresh;
mod random;
mod streaming;

use engine_core::glam::UVec2;
use engine_core::{FrameFacts, PureSession};

const FRAME: FrameFacts = FrameFacts {
    display_size: UVec2::new(1920, 1080),
    render_size: UVec2::new(1920, 1080),
    pixel_ratio: 1.0,
};

fn hash(bytes: &[u8]) -> String {
    format!("{:016x}", engine_core::content_hash(bytes) as u64)
}

pub fn run(session: &mut dyn PureSession) -> Result<(), String> {
    fresh::run(fresh::SEED, session)?;
    streaming::run(session)
}
