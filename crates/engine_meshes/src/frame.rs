use engine_render::parts::FrameKind;

#[derive(Default)]
pub struct SwitchFades {
    presented: f32,
    pub share: f32,
}

impl SwitchFades {
    fn advance(&mut self, kind: FrameKind, seconds: f32, fade_seconds: f32) {
        self.share = match (kind.gpu_mode, kind.display) {
            (false, _) => -1.0,
            (true, false) => 1.0,
            (true, true) => {
                let since = seconds - std::mem::replace(&mut self.presented, seconds);
                if fade_seconds > 0.0 && since < 1.0 {
                    engine_core::math::clamp(since / fade_seconds, 0.0, 1.0)
                } else {
                    1.0
                }
            }
        };
    }
}
