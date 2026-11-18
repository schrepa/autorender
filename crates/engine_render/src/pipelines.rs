use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Copy, Debug)]
pub struct StageLimits {
    pub bind_groups: u32,
    pub uniform_buffers: u32,
    pub storage_buffers: u32,
    pub sampled_textures: u32,
    pub samplers: u32,
    pub storage_textures: u32,
    pub dynamic_uniform_buffers: u32,
    pub dynamic_storage_buffers: u32,
}

pub const REQUIRED_LIMITS: StageLimits = StageLimits {
    bind_groups: 4,
    uniform_buffers: 12,
    storage_buffers: 8,
    sampled_textures: 16,
    samplers: 16,
    storage_textures: 4,
    dynamic_uniform_buffers: 8,
    dynamic_storage_buffers: 4,
};

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

pub fn view(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

pub fn sampler(sampler: &wgpu::Sampler) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Sampler(sampler)
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

#[derive(Clone, Copy)]
pub(crate) enum Listing {
    FromList,
    Listed,
    Unlisted,
    NoList,
    MakesRecords,
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(dead_code, reason = "only dev and profiling builds read it")
)]
#[derive(Clone, Copy)]
pub(crate) struct Asker {
    pub(crate) entry: &'static str,
    pub(crate) part: &'static str,
    pub(crate) brick: &'static str,
    pub(crate) pass: &'static str,
    pub(crate) by: &'static str,
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
#[derive(Default)]
pub(crate) struct Book;

#[cfg(not(any(feature = "dev", feature = "profiling")))]
impl Book {
    #[inline(always)]
    pub(crate) fn asked(&mut self, _made: (u64, u64), _frame: u64, _asker: impl FnOnce() -> Asker) {}

    #[inline(always)]
    pub(crate) fn ready(&mut self, _made: (u64, u64), _frame: u64, _list: Listing) {}

    #[inline(always)]
    pub(crate) fn refused(&mut self, _made: (u64, u64), _frame: u64, _error: &str) {}

    #[inline(always)]
    pub(crate) fn used(&mut self, _made: (u64, u64), _frame: u64) {}
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) use book::Book;

#[cfg(any(feature = "dev", feature = "profiling"))]
mod book {
    use engine_profile::Fact;
    use serde_json::{Map, Value, json};

    use super::{Asker, Listing};

    struct Booked {
        made: (u64, u64),
        asker: Asker,
        cause: engine_profile::Cause,
        asked: u64,
        ready: Option<u64>,
        used: Option<u64>,
        refused: Option<(u64, String)>,
        list: Option<Listing>,
    }

    #[derive(Default)]
    pub(crate) struct Book {
        booked: Vec<Booked>,
    }

    fn emit(state: &'static str, (asker, cause): (&Asker, engine_profile::Cause), error: Option<&str>) {
        if !engine_profile::recording() {
            return;
        }
        let mut facts = vec![
            ("state", Fact::Text(state)),
            ("part", Fact::Text(asker.part)),
            ("brick", Fact::Text(asker.brick)),
            ("pass", Fact::Text(asker.pass)),
            ("asked_by", Fact::Text(asker.by)),
        ];
        if let Some(error) = error {
            facts.push(("error", Fact::Text(error)));
        }
        engine_profile::caused(cause, || engine_profile::event("pipeline", asker.entry, &facts));
    }

    impl Book {
        fn find(&mut self, made: (u64, u64)) -> Option<&mut Booked> {
            self.booked.iter_mut().find(|booked| booked.made == made)
        }

        pub(crate) fn asked(&mut self, made: (u64, u64), frame: u64, asker: impl FnOnce() -> Asker) {
            let (asker, cause) = (asker(), engine_profile::cause());
            emit("asked", (&asker, cause), None);
            let booked = Booked {
                made,
                asker,
                cause,
                asked: frame,
                ready: None,
                used: None,
                refused: None,
                list: None,
            };
            match self.find(made) {
                Some(known) => *known = booked,
                None => self.booked.push(booked),
            }
        }

        pub(crate) fn ready(&mut self, made: (u64, u64), frame: u64, list: Listing) {
            if let Some(booked) = self.find(made) {
                booked.ready = Some(frame);
                booked.list = Some(list);
                emit("ready", (&booked.asker, booked.cause), None);
            }
        }

        pub(crate) fn refused(&mut self, made: (u64, u64), frame: u64, error: &str) {
            if let Some(booked) = self.find(made) {
                emit("refused", (&booked.asker, booked.cause), Some(error));
                booked.refused = Some((frame, error.to_owned()));
            }
        }

        pub(crate) fn used(&mut self, made: (u64, u64), frame: u64) {
            if let Some(booked) = self.find(made).filter(|booked| booked.used.is_none()) {
                booked.used = Some(frame);
                emit("used", (&booked.asker, booked.cause), None);
            }
        }

        pub(crate) fn state(&self, declared: impl Fn(u64) -> Value) -> Value {
            let listed = |list: Listing| match list {
                Listing::FromList => json!("from_list"),
                Listing::Listed => json!("listed"),
                Listing::Unlisted => json!("unlisted"),
                Listing::NoList => json!({"unavailable": "no_list_loaded"}),
                Listing::MakesRecords => json!({"unavailable": "makes_records"}),
            };
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
                        if booked.cause.0 != 0 {
                            facts.insert("cause".to_owned(), json!(booked.cause.0));
                        }
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
                        if let Some(list) = booked.list {
                            facts.insert("list".to_owned(), listed(list));
                        }
                        let (program, key) = booked.made;
                        (format!("{entry}/{program:x}/{key:x}"), Value::Object(facts))
                    })
                    .collect(),
            )
        }

        pub(crate) fn counts(&self, (from_list, unlisted): (u32, u32)) -> Value {
            let count = |had: fn(&Booked) -> bool| self.booked.iter().filter(|booked| had(booked)).count();
            json!({
                "asked": self.booked.len(),
                "ready": count(|booked| booked.ready.is_some()),
                "used": count(|booked| booked.used.is_some()),
                "refused": count(|booked| booked.refused.is_some()),
                "from_list": from_list,
                "unlisted": unlisted,
            })
        }
    }
}
