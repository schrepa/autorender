//! Declares the hash of the code making the records this crate asks for
//! without linking it, which keys them: the cook's levels of detail
//! (`engine_lodbuild`, [`engine_sources::emit_of`]).

fn main() {
    engine_sources::emit_of("engine_lodbuild");
}
