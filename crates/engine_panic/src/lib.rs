//! The engine's one panic hook, for every host. A panic is fatal to the
//! engine instance it happens in, a job worker's included: there is no
//! unwinding and no recovery. The hook records what panicked and where as a
//! [`PanicReport`] before the instance stops. In a page or a Web Worker it
//! keeps the report on the global object ([`PANIC_GLOBAL`]) and throws it
//! out of the module as an error whose message is [`PANIC_PREFIX`] and the
//! report's JSON; the module is unusable after, and hosts call into it no
//! more. Natively it writes that line on standard error and aborts the
//! process. Hosts then answer every development request with
//! `runtime_panicked` and the report's message.

/// What a panic records: its message, which says where it happened, and
/// that place.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PanicReport {
    /// What panicked and where: `panicked at FILE:LINE:COLUMN: MESSAGE`.
    pub message: String,
    /// Where it panicked: `FILE:LINE:COLUMN`.
    pub location: String,
}

/// What precedes a panic report's JSON: on the native host's standard error
/// line, and in the message of the error a page's or worker's module
/// throws.
pub const PANIC_PREFIX: &str = "engine-panic ";

/// The global object's property where a page's or worker's module keeps
/// its panic report's JSON once it panicked.
pub const PANIC_GLOBAL: &str = "__enginePanic";

impl PanicReport {
    /// The report of the panic `info` describes.
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

    /// The report as a JSON object.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"message":{},"location":{}}}"#,
            json_string(&self.message),
            json_string(&self.location)
        )
    }
}

/// Makes the engine's hook the process's or module's panic hook; installing
/// it again changes nothing.
pub fn install() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    #[expect(
        clippy::disallowed_methods,
        reason = "this is the engine's one panic hook"
    )]
    INSTALLED.call_once(|| std::panic::set_hook(Box::new(hook)));
}

/// Records the panic `info` describes, then stops the engine instance.
fn hook(info: &std::panic::PanicHookInfo<'_>) {
    let report = PanicReport::of(info).to_json();
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsValue;
        // Setting a property of the global object cannot fail.
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
        // The process aborts next: where standard error is gone, nothing
        // else can be told.
        let _ = writeln!(stderr, "{PANIC_PREFIX}{report}");
        let backtrace = std::backtrace::Backtrace::capture();
        if backtrace.status() == std::backtrace::BacktraceStatus::Captured {
            let _ = writeln!(stderr, "{backtrace}");
        }
        let _ = stderr.flush();
        std::process::abort();
    }
}

/// `text` as a JSON string literal.
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
