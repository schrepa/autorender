//! Declares `SOURCE`, the hash of the Stress Lab's code, which keys the
//! records its own code makes (the meshes and textures its worlds
//! generate) and which its scenes' content names ([`engine_sources`]).

fn main() {
    engine_sources::emit();
}
