mod counts;
mod digest;

use engine::smokes::Smoke;

pub(crate) const SMOKES: [Smoke; 2] = [
    Smoke {
        name: "drawn_and_culled_counts_agree",
        run: counts::run,
    },
    Smoke {
        name: "same_ticks_same_digest",
        run: digest::run,
    },
];
