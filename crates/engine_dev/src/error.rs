use engine_settings::ErrorCode;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DevError {
    pub code: ErrorCode,
    pub path: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub allowed: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nearest: Option<String>,
}

impl DevError {
    pub fn new(code: ErrorCode, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
            expected: None,
            allowed: Vec::new(),
            nearest: None,
        }
    }

    pub fn unknown<S: Into<String>>(
        path: impl Into<String>,
        got: &str,
        allowed: impl IntoIterator<Item = S>,
    ) -> Self {
        let allowed: Vec<String> = allowed.into_iter().map(Into::into).collect();
        let nearest = nearest(got, &allowed);
        Self {
            nearest,
            allowed,
            ..Self::new(ErrorCode::UnknownVariant, path, got)
        }
    }

    #[must_use]
    pub fn expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    #[must_use]
    pub fn allowed<S: Into<String>>(mut self, allowed: impl IntoIterator<Item = S>) -> Self {
        self.allowed = allowed.into_iter().map(Into::into).collect();
        self
    }

    #[must_use]
    pub fn in_args(mut self) -> Self {
        if let Some(rest) = self.path.strip_prefix("$.") {
            self.path = format!("$.args.{rest}");
        }
        self
    }
}

pub fn nearest(got: &str, allowed: &[String]) -> Option<String> {
    allowed
        .iter()
        .map(|candidate| (distance(got, candidate), candidate))
        .filter(|&(distance, candidate)| distance <= candidate.len().max(got.len()) / 2)
        .min_by_key(|&(distance, _)| distance)
        .map(|(_, candidate)| candidate.clone())
}

fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, left) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &right) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(left != right));
            diagonal = above;
        }
    }
    row[b.len()]
}

impl From<engine_scene::SceneError> for DevError {
    fn from(error: engine_scene::SceneError) -> Self {
        Self::new(error.code, error.path, error.message)
    }
}

impl From<engine_core::Error> for DevError {
    fn from(error: engine_core::Error) -> Self {
        Self::new(error.code, error.path, error.message)
    }
}

impl std::fmt::Display for DevError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {}: {}", self.code, self.path, self.message)
    }
}

impl std::error::Error for DevError {}
