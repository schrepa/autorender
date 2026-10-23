//! Declares `SOURCE`, the hash of the code generating terrain records, and
//! in `sources` each kind's own: the modules making it and what they use,
//! with the jobs' own file, which reads every kind's input
//! ([`engine_sources`]).

use engine_sources::Set;

/// Each kind of terrain record and the modules making it.
const SETS: [(&str, &[&str]); 8] = [
    ("TILES", &["tile"]),
    ("BAKES", &["bake"]),
    ("PORTALS", &["nav"]),
    ("LIGHTING", &["lighting"]),
    ("BOUNCE", &["bounce"]),
    ("MACRO", &["macro_colors"]),
    ("WATER", &["steady_water", "shallow_water"]),
    ("SHADING", &["shading"]),
];

fn main() {
    let sets = SETS.map(|(name, roots)| Set {
        name,
        roots,
        stop: &[],
        files: &["src/jobs.rs"],
        crates: None,
    });
    engine_sources::emit_sets(&sets);
}
