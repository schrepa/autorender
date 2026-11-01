use engine_core::{
    CookMode, Engine, JobExchange, JobId, MISSING_RECORD, RecordKey, URGENT_JOBS, job_priority,
};
use engine_registry::Registry;
use engine_render::Renderer;
use sha2::{Digest, Sha256};

const KIND: &str = "pipeline_list";

const QUIET_FRAMES: u32 = 30;

#[derive(Debug, Default)]
pub(crate) struct PipelineList {
    state: State,
}

#[derive(Debug, Default)]
enum State {
    #[default]
    Unasked,
    Asked {
        key: RecordKey,
        fetch: JobId,
    },
    Making {
        key: RecordKey,
        quiet: u32,
    },
    Done,
}

fn cooking(engine: &Engine) -> bool {
    engine
        .app
        .world()
        .get_resource::<CookMode>()
        .is_some_and(|mode| mode.cooking)
}

fn key_of(identity: &[u8]) -> RecordKey {
    let mut hash = Sha256::new();
    hash.update((KIND.len() as u64).to_le_bytes());
    hash.update(KIND.as_bytes());
    hash.update(identity);
    RecordKey::from_hash(hash.finalize().into())
}

impl PipelineList {
    pub(crate) fn unasked(&self) -> bool {
        matches!(self.state, State::Unasked)
    }

    pub(crate) fn waiting(&self) -> bool {
        matches!(self.state, State::Asked { .. })
    }

    pub(crate) fn making(&self) -> bool {
        matches!(self.state, State::Making { .. })
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
            for section in registry.sections() {
                identity.extend(section.key.bytes());
                identity.extend(section.current(world).to_string().bytes());
            }
        }
        let key = key_of(&identity);
        if cooking(engine) {
            self.state = State::Making { key, quiet: 0 };
            return;
        }
        self.state = match engine.app.world_mut().get_resource_mut::<JobExchange>() {
            Some(mut jobs) => State::Asked {
                key,
                fetch: jobs.fetch(None, key, job_priority(URGENT_JOBS, 0)),
            },
            None => State::Done,
        };
    }

    pub(crate) fn take(&mut self, engine: &mut Engine, renderer: &mut Renderer) {
        let State::Asked { key, fetch } = self.state else {
            return;
        };
        let world = engine.app.world_mut();
        let Some(mut jobs) = world.get_resource_mut::<JobExchange>() else {
            return;
        };
        let Some(output) = jobs.take(fetch) else {
            return;
        };
        self.state = match output.map(|record| renderer.create_listed_pipelines(&record)) {
            Ok(Ok(_)) => State::Done,
            Ok(Err(_)) => State::Making { key, quiet: 0 },
            Err(error) if error == MISSING_RECORD => State::Making { key, quiet: 0 },
            Err(_) => State::Done,
        };
    }

    pub(crate) fn after_frame(&mut self, engine: &mut Engine, renderer: &Renderer, created: u32) {
        let State::Making { key, quiet } = &mut self.state else {
            return;
        };
        let key = *key;
        let world = engine.app.world();
        let others = cooking(engine).then(|| {
            world
                .get_resource::<Registry>()
                .map_or(0, |registry| registry.pending(world))
        });
        let still = created == 0 && others.is_none_or(|others| others == 0);
        *quiet = if still { *quiet + 1 } else { 0 };
        if *quiet < QUIET_FRAMES {
            return;
        }
        if let Some(mut jobs) = engine.app.world_mut().get_resource_mut::<JobExchange>() {
            jobs.keep(key, None, renderer.pipeline_list());
        }
        self.state = State::Done;
    }
}
