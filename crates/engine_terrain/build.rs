use engine_sources::Set;

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
