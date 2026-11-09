pub mod cook;
pub mod iteration_build;
pub mod look;
pub mod pipeline;
pub mod process;
pub mod serve;

pub struct Failure {
    pub message: String,
    pub code: u8,
}

impl Failure {
    pub fn found(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 3,
        }
    }
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self { message, code: 1 }
    }
}

impl From<&str> for Failure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}
