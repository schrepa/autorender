#[cfg(any(feature = "dev", feature = "profiling"))]
use super::Renderer;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuPassTimes {
    pub frame_ms: f64,
    pub idle_ms: f64,
    pub passes: Vec<(&'static str, f64)>,
}

#[cfg(any(feature = "dev", feature = "profiling"))]
impl Renderer {
    pub fn take_gpu_memory(&mut self) -> Vec<crate::GpuMemory> {
        crate::memory::take()
    }

    pub fn end_gpu_warmup(&mut self) {
        crate::memory::end_warmup();
    }

    #[cfg(feature = "profiling")]
    pub fn gpu_pass_times(&self) -> Option<GpuPassTimes> {
        let latest = self.core.profiler.latest()?;
        let mut passes: Vec<(&'static str, f64)> = Vec::new();
        for &(span, ms) in &latest.spans {
            let pass = span.strip_prefix("stage_").or_else(|| {
                self.prelude_owners
                    .iter()
                    .find(|&&(prelude, _)| prelude == span)
                    .map(|&(_, pass)| pass)
            });
            let Some(pass) = pass else {
                continue;
            };
            match passes.iter_mut().find(|(known, _)| *known == pass) {
                Some((_, total)) => *total += ms,
                None => passes.push((pass, ms)),
            }
        }
        Some(GpuPassTimes {
            frame_ms: latest.frame_ms,
            idle_ms: latest.idle_ms,
            passes,
        })
    }

    #[cfg(not(feature = "profiling"))]
    pub fn gpu_pass_times(&self) -> Option<GpuPassTimes> {
        None
    }
}
