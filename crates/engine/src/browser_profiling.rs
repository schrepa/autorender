//! Optional capture composition. No recorder logic belongs in the frame loop.
use super::BrowserApp;
use serde_json::{Value, json};

fn now_ms() -> f64 {
    thread_local! {
        static CLOCK: web_sys::Performance = web_sys::window()
            .and_then(|window| window.performance()).expect("browser clock");
    }
    CLOCK.with(web_sys::Performance::now)
}

impl BrowserApp {
    pub(super) fn begin_profile_frame(&mut self) {
        self.profile_frame = self.profile_frame.saturating_add(1);
        engine_profile::set_frame(self.profile_frame);
        self.renderer.profile_poll();
        if engine_profile::active() {
            let memory = engine_profile::allocation_snapshot();
            engine_profile::counter("memory.live_requested_bytes", memory.live_bytes as f64);
            engine_profile::counter(
                "memory.lifetime_peak_requested_bytes",
                memory.peak_live_bytes as f64,
            );
            engine_profile::counter("memory.allocations_total", memory.allocations as f64);
            engine_profile::counter("memory.reallocations_total", memory.reallocations as f64);
            engine_profile::counter(
                "memory.requested_bytes_total",
                memory.requested_bytes as f64,
            );
            engine_profile::counter(
                "memory.wasm_linear_bytes",
                (core::arch::wasm32::memory_size(0) * 65536) as f64,
            );
        }
    }

    /// Start one bounded capture for this Wasm instance. Calls while capturing fail.
    pub fn profiling_start(&mut self, max_events: usize) -> Result<(), &'static str> {
        engine_profile::set_clock(now_ms);
        let id = engine_profile::start(max_events)?;
        self.renderer.profile_begin(id);
        Ok(())
    }

    /// Stop CPU collection before allocating/serializing the exported capture.
    /// GPU readbacks may still be pending; poll and retrieve their final report.
    pub fn profiling_stop(&mut self) -> Result<Value, &'static str> {
        if !engine_profile::active() {
            return Err("no profiling capture is active");
        }
        self.renderer.profile_poll();
        let cpu = engine_profile::finish();
        Ok(json!({
            "cpu": cpu, "gpu": self.renderer.profile_report(),
            "memory": {"wasm_linear_bytes": core::arch::wasm32::memory_size(0) * 65536,
                "scope": "Wasm linear memory capacity, not live allocation bytes or process memory"}
        }))
    }

    pub fn profiling_poll(&mut self) {
        self.renderer.profile_poll();
    }

    pub fn profiling_status(&self) -> Value {
        json!({"enabled": true, "active": engine_profile::active(),
            "capture_id": engine_profile::capture_id(), "gpu": self.renderer.profile_status()})
    }

    pub fn profiling_report(&mut self) -> Value {
        self.renderer.profile_report()
    }
}
