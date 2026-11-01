fn main() {
    engine_sources::emit_of(&[
        ("LODBUILD", "engine_lodbuild"),
        ("TEXTURES", "engine_textures"),
        ("RENDER", "engine_render"),
    ]);
}
