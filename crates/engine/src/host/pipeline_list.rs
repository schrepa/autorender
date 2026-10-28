use engine_core::{
    Cooking, Engine, Environment, JobId, Jobs, MISSING_RECORD, RecordKey, URGENT_JOBS, job_priority,
};
use engine_render::Renderer;

const KIND: &str = "pipeline_list";

const FEATURE: &str = "pipelines";

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

impl PipelineList {
    pub(crate) fn unasked(&self) -> bool {
        matches!(self.state, State::Unasked)
    }

    pub(crate) fn waiting(&self) -> bool {
        matches!(self.state, State::Asked { .. })
    }

    pub(crate) fn ask(&mut self, engine: &mut Engine, content: Option<Vec<u8>>, gpu: bool) {
        let Some(mut identity) = content else {
            self.state = State::Done;
            return;
        };
        let world = engine.app.world_mut();
        identity.extend(format!("{:?}", Environment::current(world).sanitized()).bytes());
        identity.push(u8::from(gpu));
        let key = RecordKey::new(&engine_render::SOURCE, KIND, &identity);
        if world.contains_resource::<Cooking>() {
            self.state = State::Making { key, quiet: 0 };
            return;
        }
        self.state = match world.get_resource_mut::<Jobs>() {
            Some(mut jobs) => State::Asked {
                key,
                fetch: jobs.fetch(key, job_priority(URGENT_JOBS, 0)),
            },
            None => State::Done,
        };
    }

    pub(crate) fn take(&mut self, engine: &mut Engine, renderer: &mut Renderer) {
        let State::Asked { key, fetch } = self.state else {
            return;
        };
        let Some(mut jobs) = engine.app.world_mut().get_resource_mut::<Jobs>() else {
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
        let world = engine.app.world_mut();
        let others = world.get_resource::<Cooking>().map(|cooking| {
            cooking
                .features()
                .iter()
                .filter(|(feature, _)| *feature != FEATURE)
                .map(|(_, left)| left)
                .sum::<u64>()
        });
        let still = created == 0 && others.is_none_or(|others| others == 0);
        *quiet = if still { *quiet + 1 } else { 0 };
        let done = *quiet >= QUIET_FRAMES;
        if done && let Some(mut jobs) = world.get_resource_mut::<Jobs>() {
            jobs.keep(key, renderer.pipeline_list());
        }
        if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
            cooking.left(FEATURE, u64::from(!done));
        }
        if done {
            self.state = State::Done;
        }
    }
}
