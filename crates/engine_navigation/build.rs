use engine_sources::Set;

fn main() {
    engine_sources::emit_sets(&[Set {
        name: "PORTALS",
        roots: &["nav"],
        stop: &["agents", "dev"],
        files: &["src/jobs.rs"],
        crates: Some(&["engine_terrain", "engine_math", "engine_settings"]),
    }]);
}
