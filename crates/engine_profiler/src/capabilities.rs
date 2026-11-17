use serde_json::{Value, json};

pub(crate) fn machine(adapter: &engine::AdapterInfo) -> Value {
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": cpu_name(),
        "logical_processors": std::thread::available_parallelism().map_or(0, usize::from),
        "gpu": {
            "name": adapter.name,
            "vendor": vendor(adapter),
            "type": adapter.device_type,
            "backend": adapter.backend,
            "driver": adapter.driver,
        },
    })
}

fn vendor(adapter: &engine::AdapterInfo) -> &'static str {
    match engine::host::vendor(adapter) {
        "" => "other",
        vendor => vendor,
    }
}

fn cpu_name() -> Option<String> {
    if cfg!(windows) {
        return std::env::var("PROCESSOR_IDENTIFIER").ok();
    }
    None
}
