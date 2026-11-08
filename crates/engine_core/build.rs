fn main() {
    engine_sources::emit_of(&[
        ("LODBUILD", "engine_lodbuild"),
        ("TEXTURE_COOK", "engine_texture_cook"),
        ("RENDER", "engine_render"),
    ]);
}
