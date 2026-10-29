fn main() {
    let schemas = serde_json::json!({
        "PlayerWalk": schemars::schema_for!(engine_player::PlayerWalk),
    });
    engine::tooling::print_generated(&[(
        "crates/engine_player/src/generated/inspectors.json",
        serde_json::to_string_pretty(&schemas).expect("schema_serializes") + "\n",
    )]);
}
