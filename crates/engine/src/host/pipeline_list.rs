//! The scene's pipeline list ([`Renderer::pipeline_list`]) as hosts ask
//! for it, create it up front and keep it: a record of every pipeline a
//! session of the scene created. It is kept by the renderer's code and the
//! scene's content as its first frame starts: what the game says of it
//! ([`crate::Game::content`]), its environment and whether frames present
//! on the GPU. Before that frame the host asks for it, and once it arrives
//! creates every pipeline on it up front
//! ([`Renderer::create_listed_pipelines`]): preparation waits for it.
//! Where no pack or store holds it, the session's own pipelines become it
//! once frames stopped creating them a while; while the session cooks,
//! once every other record is made too, so the cook ships it.

use engine_core::{
    Cooking, Engine, Environment, JobId, Jobs, MISSING_RECORD, RecordKey, URGENT_JOBS, job_priority,
};
use engine_render::Renderer;

/// The kind of the pipeline list's record.
const KIND: &str = "pipeline_list";

/// The feature cooking sessions count the list under.
const FEATURE: &str = "pipelines";

/// Frames in a row creating no pipeline after which a session's pipelines
/// become the list.
const QUIET_FRAMES: u32 = 30;

/// The list's state for the running scene.
#[derive(Debug, Default)]
pub(crate) struct PipelineList {
    state: State,
}

#[derive(Debug, Default)]
enum State {
    /// To ask for before the next frame.
    #[default]
    Unasked,
    /// Asked for: its key, and the fetch of its record.
    Asked { key: RecordKey, fetch: JobId },
    /// No pack or store held it, or the session cooks: the session's
    /// pipelines become it once frames stop creating them, after `quiet`
    /// such frames so far.
    Making { key: RecordKey, quiet: u32 },
    /// Created up front or kept; or none, for a game naming no content.
    Done,
}

impl PipelineList {
    /// Whether the list has yet to be asked for.
    pub(crate) fn unasked(&self) -> bool {
        matches!(self.state, State::Unasked)
    }

    /// Whether the list was asked for and has yet to arrive.
    pub(crate) fn waiting(&self) -> bool {
        matches!(self.state, State::Asked { .. })
    }

    /// Asks for the list of `engine`'s scene, whose game names its content
    /// `content` (`None`: no list), its frames presenting on the GPU when
    /// `gpu`: from the cook's packs or the host's store or, while the
    /// session cooks, to make.
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

    /// Takes the list once it arrived and creates its pipelines; where none
    /// held it, or it does not read, the session's pipelines become it.
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
            // A list that does not read is made again.
            Ok(Err(_)) => State::Making { key, quiet: 0 },
            Err(error) if error == MISSING_RECORD => State::Making { key, quiet: 0 },
            Err(_) => State::Done,
        };
    }

    /// After a frame that created `created` pipelines: once frames stopped
    /// creating them (and, cooking, every other record is made), keeps the
    /// session's pipelines as the list; while cooking, says whether the
    /// list is left.
    pub(crate) fn after_frame(&mut self, engine: &mut Engine, renderer: &Renderer, created: u32) {
        let State::Making { key, quiet } = &mut self.state else {
            return;
        };
        let key = *key;
        *quiet = if created == 0 { *quiet + 1 } else { 0 };
        let world = engine.app.world_mut();
        let others = world.get_resource::<Cooking>().map(|cooking| {
            cooking
                .features()
                .iter()
                .filter(|(feature, _)| *feature != FEATURE)
                .map(|(_, left)| left)
                .sum::<u64>()
        });
        let done = *quiet >= QUIET_FRAMES && others.is_none_or(|others| others == 0);
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
