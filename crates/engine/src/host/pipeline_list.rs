use engine_core::{
    Engine, Interest, JobExchange, RecordKey, Records, Request, URGENT_JOBS, job_priority,
    look_record,
};
use engine_registry::Registry;
use engine_render::Renderer;
use sha2::{Digest, Sha256};

const KIND: &str = "pipeline_list";

#[derive(Debug, Default)]
pub(crate) struct PipelineList {
    state: State,
    cooks: bool,
}

#[derive(Debug, Default)]
enum State {
    #[default]
    Unasked,
    Asked {
        key: RecordKey,
        fetch: Interest<RecordKey>,
    },
    Making {
        key: RecordKey,
        kept: u32,
    },
    Done,
}

fn compiled(renderer: &Renderer) -> u32 {
    renderer
        .compiled_per_pass()
        .iter()
        .map(|(_, count)| *count)
        .sum()
}

fn key_of(identity: &[u8]) -> RecordKey {
    let mut hash = Sha256::new();
    hash.update((KIND.len() as u64).to_le_bytes());
    hash.update(KIND.as_bytes());
    hash.update(identity);
    RecordKey::from_hash(hash.finalize().into())
}

impl PipelineList {
    pub(crate) fn cook(&mut self) {
        self.cooks = true;
    }

    pub(crate) fn restart(&mut self) {
        self.state = State::Unasked;
    }

    pub(crate) fn unasked(&self) -> bool {
        matches!(self.state, State::Unasked)
    }

    pub(crate) fn waiting(&self) -> bool {
        matches!(self.state, State::Asked { .. })
    }

    pub(crate) fn ask(&mut self, engine: &mut Engine, content: Option<Vec<u8>>) {
        let Some(mut identity) = content else {
            self.state = State::Done;
            return;
        };
        let world = engine.app.world();
        if let Some(registry) = world.get_resource::<Registry>() {
            for brick in registry.bricks() {
                identity.extend(brick.package.bytes());
                identity.push(0);
                identity.extend(brick.source());
            }
            let context = world.get_resource::<engine_core::SceneContext>();
            for section in registry.sections() {
                identity.extend(section.key.bytes());
                let current = match context {
                    Some(context) => context.scope(|| section.current(world)),
                    None => section.current(world),
                };
                identity.extend(current.to_string().bytes());
            }
        }
        let key = key_of(&identity);
        if self.cooks {
            self.state = State::Making { key, kept: 0 };
            return;
        }
        let world = engine.app.world_mut();
        self.state = if world.contains_resource::<JobExchange>() {
            State::Asked {
                key,
                fetch: look_record(world, None, key, job_priority(URGENT_JOBS, 0)),
            }
        } else {
            State::Done
        };
    }

    pub(crate) fn take(&mut self, engine: &mut Engine, renderer: &mut Renderer) {
        let State::Asked { key, fetch } = &self.state else {
            return;
        };
        let key = *key;
        let Some(records) = engine.app.world().get_resource::<Records>() else {
            return;
        };
        self.state = match records.request(fetch) {
            Some(Request::Ready(record)) => match renderer.create_listed_pipelines(record) {
                Ok(_) => State::Making {
                    key,
                    kept: compiled(renderer),
                },
                Err(_) => State::Making { key, kept: 0 },
            },
            Some(Request::Missing) => State::Making { key, kept: 0 },
            Some(Request::Failed(_)) => State::Done,
            Some(Request::Asked(_)) | None => return,
        };
    }

    pub(crate) fn after_frame(&mut self, engine: &mut Engine, renderer: &Renderer, created: u32) {
        let State::Making { key, kept } = self.state else {
            return;
        };
        let total = compiled(renderer);
        if created > 0 || total <= kept || !engine.readiness().settled() {
            return;
        }
        if let Some(mut jobs) = engine.app.world_mut().get_resource_mut::<JobExchange>() {
            jobs.keep(key, None, renderer.pipeline_list());
        }
        self.state = State::Making { key, kept: total };
    }
}
