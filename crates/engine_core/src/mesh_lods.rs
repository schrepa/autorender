use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetLibrary, MeshHandle, MeshLods, lod_input};

use crate::jobs::{MESH_LODS_NAME, record_key};
use crate::{
    Census, CookMode, Error, ErrorCode, Interest, JobExchange, JobId, JobRef, MESH_LODS,
    MISSING_RECORD, RecordKey, Request, Requests, Ticket, VIEW_JOBS, content_hash, job_priority,
};

#[derive(Resource, Default, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MeshLevels {
    requests: Requests<RecordKey, Arc<MeshLods>>,
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    jobs: Vec<(RecordKey, Ticket, JobId)>,
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    waiting: Vec<(MeshHandle, Interest<RecordKey>)>,
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    ended: Vec<(MeshHandle, Interest<RecordKey>)>,
}

impl MeshLevels {
    pub fn census(&self) -> Census {
        self.requests.census()
    }

    pub fn unsettled(&self) -> impl Iterator<Item = (&RecordKey, &Request<Arc<MeshLods>>)> + '_ {
        self.requests
            .iter()
            .filter(|(_, request)| request.ready().is_none())
    }
}

fn lod_record(input: &[u8]) -> RecordKey {
    record_key(
        &crate::sources::LODBUILD,
        MESH_LODS_NAME,
        &content_hash(input).to_le_bytes(),
    )
}

fn landed(output: Result<Arc<[u8]>, String>) -> Result<Option<MeshLods>, Error> {
    match output {
        Ok(bytes) => MeshLods::from_bytes(&bytes).map(Some).map_err(|error| {
            Error::new(ErrorCode::InvalidRecord, "$.mesh_lods").got(error.to_string())
        }),
        Err(message) if message == MISSING_RECORD => Ok(None),
        Err(message) => Err(Error::new(ErrorCode::InvalidRecord, "$.mesh_lods").got(message)),
    }
}

impl MeshLevels {
    fn ask(&mut self, jobs: &mut JobExchange, handle: MeshHandle, input: Vec<u8>, cooking: bool) {
        let key = lod_record(&input);
        let interest = self.requests.ask(key);
        if let Some(ticket) = self.requests.ticket(&key)
            && !self.jobs.iter().any(|&(asked, ..)| asked == key)
        {
            let priority = job_priority(VIEW_JOBS, self.jobs.len() as u32);
            let job = if cooking {
                jobs.submit_record(JobRef::from(&MESH_LODS), input, priority, key)
            } else {
                jobs.fetch(Some(JobRef::from(&MESH_LODS)), key, priority)
            };
            self.jobs.push((key, ticket, job));
        }
        self.waiting.push((handle, interest));
    }

    fn land(&mut self, jobs: &mut JobExchange) {
        let requests = &mut self.requests;
        self.jobs.retain(|&(key, ticket, job)| {
            let Some(output) = jobs.take(job) else {
                return true;
            };
            match landed(output) {
                Ok(Some(lods)) => requests.ready(&key, ticket, Arc::new(lods)),
                Ok(None) => requests.miss(&key, ticket),
                Err(error) => requests.fail(&key, ticket, error),
            };
            false
        });
    }

    fn settle_meshes(&mut self, assets: &mut AssetLibrary) {
        for (handle, interest) in std::mem::take(&mut self.waiting) {
            if assets.mesh(handle).is_none() {
                continue;
            }
            match self.requests.of(&interest) {
                Some(Request::Ready(lods)) => {
                    if assets.settle_lods(handle, Some(Arc::clone(lods))).is_err() {
                        let _ = assets.settle_lods(handle, None);
                    }
                }
                Some(Request::Failed(_) | Request::Missing) => {
                    let _ = assets.settle_lods(handle, None);
                    self.ended.push((handle, interest));
                }
                Some(Request::Asked(_)) | None => self.waiting.push((handle, interest)),
            }
        }
        self.ended.retain(|&(handle, _)| assets.mesh(handle).is_some());
    }

    fn release(&mut self, jobs: &mut JobExchange, now: f64) {
        for (key, released) in self.requests.settle(now) {
            if released.asked().is_none() {
                continue;
            }
            if let Some(at) = self.jobs.iter().position(|&(asked, ..)| asked == key) {
                let (_, _, job) = self.jobs.remove(at);
                jobs.cancel(job);
            }
        }
    }
}

pub(crate) fn settle(world: &mut World) {
    let cooking = world
        .get_resource::<CookMode>()
        .is_some_and(|mode| mode.cooking);
    let now = world
        .get_resource::<crate::FrameTime>()
        .map_or(0.0, |time| time.elapsed_seconds);
    let awaiting = world.resource_mut::<AssetLibrary>().take_awaiting_lods();
    world.resource_scope(|world, mut levels: Mut<MeshLevels>| {
        world.resource_scope(|world, mut jobs: Mut<JobExchange>| {
            let mut assets = world.resource_mut::<AssetLibrary>();
            for handle in awaiting {
                match assets.mesh(handle).and_then(|mesh| lod_input(mesh)) {
                    Some(input) => levels.ask(&mut jobs, handle, input, cooking),
                    None => {
                        let _ = assets.settle_lods(handle, None);
                    }
                }
            }
            levels.land(&mut jobs);
            levels.settle_meshes(&mut assets);
            levels.release(&mut jobs, now);
        });
    });
}
