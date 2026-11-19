use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub(crate) fn floor() -> wgpu::Limits {
    wgpu::Limits {
        max_bind_groups: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_storage_buffers_per_shader_stage: 8,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_storage_textures_per_shader_stage: 4,
        max_dynamic_uniform_buffers_per_pipeline_layout: 8,
        max_dynamic_storage_buffers_per_pipeline_layout: 4,
        max_uniform_buffer_binding_size: engine_place::UNIFORM_BINDING_BYTES,
        ..wgpu::Limits::downlevel_defaults()
    }
}

thread_local! {
    static CREATED: RefCell<Vec<(&'static str, u32)>> = const { RefCell::new(Vec::new()) };
}

fn add(counts: &mut Vec<(&'static str, u32)>, pass: &'static str, created: u32) {
    match counts.iter_mut().find(|(known, _)| *known == pass) {
        Some((_, held)) => *held = held.saturating_add(created),
        None => counts.push((pass, created)),
    }
}

pub(crate) fn count(pass: &'static str, created: u32) {
    CREATED.with(|counted| add(&mut counted.borrow_mut(), pass, created));
}

fn take_created() -> Vec<(&'static str, u32)> {
    CREATED.with(|created| std::mem::take(&mut *created.borrow_mut()))
}

#[derive(Default)]
pub(crate) struct Ledger {
    created: u32,
    per_pass: Vec<(&'static str, u32)>,
    frames: u64,
    compiled: Arc<AtomicU64>,
}

impl Ledger {
    pub(crate) fn frame_submitted(&mut self, queue: &crate::gpu_calls::Queue) -> u32 {
        let mut created = 0_u32;
        for (pass, more) in take_created() {
            add(&mut self.per_pass, pass, more);
            created = created.saturating_add(more);
        }
        self.created = self.created.saturating_add(created);
        if created > 0 {
            self.frames += 1;
            let (compiled, frame) = (Arc::clone(&self.compiled), self.frames);
            queue.on_submitted_work_done(move || {
                compiled.fetch_max(frame, Ordering::Relaxed);
            });
        }
        created
    }

    pub(crate) fn created(&self) -> u32 {
        self.created
    }

    pub(crate) fn per_pass(&self) -> &[(&'static str, u32)] {
        &self.per_pass
    }

    pub(crate) fn compiling(&self) -> u32 {
        let left = self
            .frames
            .saturating_sub(self.compiled.load(Ordering::Relaxed));
        u32::try_from(left).unwrap_or(u32::MAX)
    }
}

#[expect(clippy::disallowed_methods, reason = "the one way to make bind groups")]
pub(crate) fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries,
    })
}

#[cfg_attr(
    not(feature = "dev"),
    expect(dead_code, reason = "only dev builds read it")
)]
#[derive(Clone, Copy)]
pub(crate) struct Asker {
    pub(crate) entry: &'static str,
    pub(crate) part: &'static str,
    pub(crate) brick: &'static str,
    pub(crate) pass: &'static str,
    pub(crate) by: &'static str,
}

#[cfg(not(feature = "dev"))]
#[derive(Default)]
pub(crate) struct Book;

#[cfg(not(feature = "dev"))]
impl Book {
    #[inline(always)]
    pub(crate) fn asked(&mut self, _made: (u64, u64), _frame: u64, _asker: impl FnOnce() -> Asker) {}

    #[inline(always)]
    pub(crate) fn ready(&mut self, _made: (u64, u64), _frame: u64) {}

    #[inline(always)]
    pub(crate) fn refused(&mut self, _made: (u64, u64), _frame: u64, _error: &str) {}

    #[inline(always)]
    pub(crate) fn used(&mut self, _made: (u64, u64), _frame: u64) {}
}

#[cfg(feature = "dev")]
pub(crate) use book::Book;

#[cfg(feature = "dev")]
mod book {
    use serde_json::{Map, Value, json};

    use super::Asker;

    struct Booked {
        made: (u64, u64),
        asker: Asker,
        asked: u64,
        ready: Option<u64>,
        used: Option<u64>,
        refused: Option<(u64, String)>,
    }

    #[derive(Default)]
    pub(crate) struct Book {
        booked: Vec<Booked>,
    }

    impl Book {
        fn find(&mut self, made: (u64, u64)) -> Option<&mut Booked> {
            self.booked.iter_mut().find(|booked| booked.made == made)
        }

        pub(crate) fn asked(&mut self, made: (u64, u64), frame: u64, asker: impl FnOnce() -> Asker) {
            let booked = Booked {
                made,
                asker: asker(),
                asked: frame,
                ready: None,
                used: None,
                refused: None,
            };
            match self.find(made) {
                Some(known) => *known = booked,
                None => self.booked.push(booked),
            }
        }

        pub(crate) fn ready(&mut self, made: (u64, u64), frame: u64) {
            if let Some(booked) = self.find(made) {
                booked.ready = Some(frame);
            }
        }

        pub(crate) fn refused(&mut self, made: (u64, u64), frame: u64, error: &str) {
            if let Some(booked) = self.find(made) {
                booked.refused = Some((frame, error.to_owned()));
            }
        }

        pub(crate) fn used(&mut self, made: (u64, u64), frame: u64) {
            if let Some(booked) = self.find(made).filter(|booked| booked.used.is_none()) {
                booked.used = Some(frame);
            }
        }

        pub(crate) fn state(&self, declared: impl Fn(u64) -> Value) -> Value {
            Value::Object(
                self.booked
                    .iter()
                    .map(|booked| {
                        let Asker {
                            entry,
                            part,
                            brick,
                            pass,
                            by,
                        } = booked.asker;
                        let mut facts = Map::new();
                        facts.insert("part".to_owned(), json!(part));
                        facts.insert("brick".to_owned(), json!(brick));
                        facts.insert("pass".to_owned(), json!(pass));
                        facts.insert("asked_by".to_owned(), json!(by));
                        facts.insert("declared".to_owned(), declared(booked.made.0));
                        facts.insert("asked".to_owned(), json!(booked.asked));
                        if let Some(ready) = booked.ready {
                            facts.insert("ready".to_owned(), json!(ready));
                        }
                        if let Some(used) = booked.used {
                            facts.insert("used".to_owned(), json!(used));
                        }
                        if let Some((frame, error)) = &booked.refused {
                            facts.insert(
                                "refused".to_owned(),
                                json!({"frame": frame, "error": error}),
                            );
                        }
                        let (program, key) = booked.made;
                        (format!("{entry}/{program:x}/{key:x}"), Value::Object(facts))
                    })
                    .collect(),
            )
        }

        pub(crate) fn counts(&self) -> Value {
            let count = |had: fn(&Booked) -> bool| self.booked.iter().filter(|booked| had(booked)).count();
            json!({
                "asked": self.booked.len(),
                "ready": count(|booked| booked.ready.is_some()),
                "used": count(|booked| booked.used.is_some()),
                "refused": count(|booked| booked.refused.is_some()),
            })
        }
    }
}
