#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PanicReport {
    pub message: String,
    pub location: String,
}

pub const PANIC_PREFIX: &str = "engine-panic ";

pub const PANIC_GLOBAL: &str = "__enginePanic";

impl PanicReport {
    fn of(info: &std::panic::PanicHookInfo<'_>) -> Self {
        let location = info.location().map_or_else(
            || "an unknown location".to_owned(),
            |location| {
                format!(
                    "{}:{}:{}",
                    location.file(),
                    location.line(),
                    location.column()
                )
            },
        );
        let payload = info.payload_as_str().unwrap_or("Box<dyn Any>");
        Self {
            message: format!("panicked at {location}: {payload}"),
            location,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            r#"{{"message":{},"location":{}}}"#,
            json_string(&self.message),
            json_string(&self.location)
        )
    }
}

pub fn install() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    #[expect(
        clippy::disallowed_methods,
        reason = "this is the engine's one panic hook"
    )]
    INSTALLED.call_once(|| std::panic::set_hook(Box::new(hook)));
}

fn hook(info: &std::panic::PanicHookInfo<'_>) {
    let report = PanicReport::of(info).to_json();
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsValue;
        let _ = js_sys::Reflect::set(
            &js_sys::global(),
            &JsValue::from_str(PANIC_GLOBAL),
            &JsValue::from_str(&report),
        );
        wasm_bindgen::throw_str(&format!("{PANIC_PREFIX}{report}"));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::io::Write;
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(stderr, "{PANIC_PREFIX}{report}");
        let backtrace = std::backtrace::Backtrace::capture();
        if backtrace.status() == std::backtrace::BacktraceStatus::Captured {
            let _ = writeln!(stderr, "{backtrace}");
        }
        let _ = stderr.flush();
        std::process::abort();
    }
}

fn json_string(text: &str) -> String {
    use std::fmt::Write;
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('"');
    for character in text.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            control if u32::from(control) < 0x20 => {
                let _ = write!(quoted, "\\u{:04x}", u32::from(control));
            }
            other => quoted.push(other),
        }
    }
    quoted.push('"');
    quoted
}
