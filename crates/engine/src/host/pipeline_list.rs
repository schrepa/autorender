use std::sync::Arc;

use engine_core::{
    Background, Engine, JobExchange, Landed, RecordKey, Records, URGENT_JOBS, job_priority,
    look_record, record_missing,
};
use engine_registry::Registry;
use engine_render::Renderer;
use sha2::{Digest, Sha256};

const KIND: &str = "pipeline_list";

#[derive(Debug, Default)]
pub(crate) struct PipelineList {
    state: State,
    cached: u32,
}

#[derive(Debug, Default)]
enum State {
    #[default]
    Unasked,
    Asked {
        key: RecordKey,
        fetch: Background<Arc<[u8]>>,
        cause: engine_profile::Cause,
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
        .sum::<u32>()
        .saturating_add(renderer.pipelines_warmed())
}

fn key_of(identity: &[u8]) -> RecordKey {
    let mut hash = Sha256::new();
    hash.update((KIND.len() as u64).to_le_bytes());
    hash.update(KIND.as_bytes());
    hash.update(identity);
    RecordKey::from_hash(hash.finalize().into())
}

impl PipelineList {
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
        if let Some(registry) = engine.app.world().get_resource::<Registry>() {
            for brick in registry
                .bricks()
                .iter()
                .filter(|brick| !brick.render.is_empty())
            {
                identity.extend(brick.package.bytes());
                identity.push(0);
                identity.extend(brick.source());
            }
        }
        let key = key_of(&identity);
        let world = engine.app.world_mut();
        self.state = if world.contains_resource::<JobExchange>() {
            let cause = engine_profile::open("launch", "pipeline_list");
            State::Asked {
                key,
                fetch: engine_profile::caused(cause, || {
                    look_record(world, None, (key, job_priority(URGENT_JOBS, 0)), Ok)
                }),
                cause,
            }
        } else {
            State::Done
        };
    }

    pub(crate) fn take(&mut self, engine: &mut Engine, renderer: &mut Renderer) {
        let State::Asked { key, fetch, cause } = &mut self.state else {
            return;
        };
        let (key, cause) = (*key, *cause);
        let Some(records) = engine.app.world().get_resource::<Records>() else {
            return;
        };
        self.state = match fetch.poll(records) {
            Landed::Ready(record) => match engine_profile::caused(cause, || {
                renderer.create_listed_pipelines(record)
            }) {
                Ok(()) => State::Making {
                    key,
                    kept: compiled(renderer),
                },
                Err(_) => State::Making { key, kept: 0 },
            },
            Landed::Failed(error) if record_missing(error) => State::Making { key, kept: 0 },
            Landed::Failed(_) => State::Done,
            Landed::Waiting => return,
        };
    }

    pub(crate) fn after_frame(&mut self, engine: &mut Engine, renderer: &Renderer) {
        if engine.readiness().compiling > 0 || renderer.pipelines_warming() > 0 {
            return;
        }
        let total = compiled(renderer);
        if total > self.cached {
            renderer.keep_pipeline_cache();
            self.cached = total;
        }
        let State::Making { key, kept } = self.state else {
            return;
        };
        if total <= kept || !engine.readiness().settled() {
            return;
        }
        if let Some(mut jobs) = engine.app.world_mut().get_resource_mut::<JobExchange>()
            && !jobs.verifying()
        {
            jobs.keep(key, None, renderer.pipeline_list());
        }
        self.state = State::Making { key, kept: total };
    }
}
