use engine::bricks::AddBrick;

pub fn standard_engine() -> engine::Engine {
    let mut engine = engine::Engine::new(engine::EngineConfig::default());
    engine.app.add_bricks(engine::bricks::standard());
    engine
}

pub fn register_standard_kinds() {
    drop(standard_engine());
}
