use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Fault {
    pub name: &'static str,
    pub file: String,
    pub item: String,
    pub detail: String,
}

impl Fault {
    pub fn new(
        name: &'static str,
        file: &str,
        item: impl Into<String>,
        detail: impl std::fmt::Display,
    ) -> Self {
        Self {
            name,
            file: file.to_owned(),
            item: item.into(),
            detail: detail.to_string(),
        }
    }

    pub fn json(&self) -> Value {
        json!({
            "error": self.name,
            "file": self.file,
            "item": self.item,
            "detail": self.detail,
        })
    }
}
