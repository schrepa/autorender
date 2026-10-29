use super::{AppHost, now_ms};
use engine_core::{Error, ErrorCode, Expected};
use serde_json::{Value, json};

fn linear_memory_bytes() -> Option<usize> {
    #[cfg(target_arch = "wasm32")]
    {
        Some(core::arch::wasm32::memory_size(0) * 65536)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}

impl AppHost {
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
            if let Some(bytes) = linear_memory_bytes() {
                engine_profile::counter("memory.wasm_linear_bytes", bytes as f64);
            }
        }
    }

    pub fn profiling_start(&mut self, max_events: usize) -> Result<(), Error> {
        engine_profile::set_clock(now_ms);
        if !(1..=engine_profile::MAX_EVENTS).contains(&max_events) {
            return Err(Error::new(ErrorCode::InvalidArguments, "$.max_events")
                .range(1.0, engine_profile::MAX_EVENTS as f64)
                .got(max_events));
        }
        if engine_profile::active() {
            return Err(Error::new(ErrorCode::CaptureFailed, "$").expected(Expected::Absent));
        }
        let id = engine_profile::start(max_events)
            .map_err(|_| Error::new(ErrorCode::CaptureFailed, "$"))?;
        self.attribution.captured.clear();
        self.renderer.profile_begin(id);
        Ok(())
    }

    pub fn profiling_stop(&mut self) -> Result<Value, Error> {
        if !engine_profile::active() {
            return Err(Error::new(ErrorCode::CaptureFailed, "$").expected(Expected::Running));
        }
        self.renderer.profile_poll();
        let cpu = engine_profile::finish();
        let memory = linear_memory_bytes().map(|bytes| {
            json!({"wasm_linear_bytes": bytes,
                "scope": "Wasm linear memory capacity, not live allocation bytes or process memory"})
        });
        let captured = std::mem::take(&mut self.attribution.captured);
        let bricks = engine_core::BrickCosts::table(&captured);
        Ok(json!({
            "bricks": bricks,
            "cpu": cpu,
            "gpu": self.renderer.profile_report(),
            "memory": memory,
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
