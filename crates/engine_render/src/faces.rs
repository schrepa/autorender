pub const FACE_ROOT: &str = "face";

pub fn face_module(face: &str) -> String {
    format!("{FACE_ROOT}::{face}")
}

pub fn face_of(module: &str) -> Option<&str> {
    let face = module.strip_prefix(FACE_ROOT)?.strip_prefix("::")?;
    (!face.is_empty() && !face.contains("::")).then_some(face)
}
