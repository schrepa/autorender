//! Declares `SOURCE`, the hash of this crate's code, which keys the records
//! it makes: impostors' layers, the runtime virtual texture's far pages and
//! the scene's pipeline list ([`engine_sources`]).

fn main() {
    engine_sources::emit();
}
