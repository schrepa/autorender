use std::path::Path;

pub fn coverage(
    root: &Path,
    development: &engine_dev::schemas::Generated,
) -> Result<String, String> {
    engine_checks::coverage::coverage(
        &mut crate::bricks::standard_engine(),
        development,
        &crate::smoke::scripts(root)?,
    )
}

pub fn check(root: &Path) -> Result<(), String> {
    let engine = crate::bricks::standard_engine();
    let development = engine_dev::schemas::generated(engine.app.world())?;
    engine_checks::coverage::check(&coverage(root, &development)?)
}
