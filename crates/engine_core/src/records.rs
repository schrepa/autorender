//! Records game code makes itself: content a world builds on the CPU with
//! code the job workers lack, such as the meshes and textures a scene's
//! generators make. Each request names its record's key and the code making
//! it. Hosts hand the record back from the cook's packs or what they kept;
//! where none holds it, the code makes it here, in place, and the host keeps
//! it, so the bytes are the record wherever they came from and a cooking
//! session's pack takes them. Each request's delivery then puts the bytes
//! where they go, such as a mesh or texture reserved for them
//! ([`crate::AssetLibrary::fill_mesh`]). While a session cooks, the records still
//! to come count as left.
//!
//! Requests made before a host serves the engine wait for the next frame:
//! the host answers them from then on, and an engine no host serves makes
//! every one in place at once.

use bevy_ecs::prelude::*;

use crate::{Cooking, JobFailures, JobId, Jobs, RecordKey, URGENT_JOBS, job_priority};

/// The code making a record's bytes, or why it could not.
type MakeRecord = Box<dyn FnOnce() -> Result<Vec<u8>, String> + Send + Sync>;
/// What a record's bytes become in the world, or why they could not.
type DeliverRecord = Box<dyn FnOnce(&mut World, &[u8]) -> Result<(), String> + Send + Sync>;

/// A record asked for and not yet delivered.
struct Request {
    key: RecordKey,
    make: MakeRecord,
    deliver: DeliverRecord,
    // The host's lookup under way.
    job: Option<JobId>,
}

/// Records game code makes itself, asked for and not yet delivered, and the
/// ones that failed.
#[derive(Resource, Default)]
pub struct Records {
    requests: Vec<Request>,
    failures: JobFailures,
    frame: u64,
}

impl Records {
    /// Asks for the record `key`, which `make` makes where no host holds
    /// it; `deliver` takes its bytes once they are here, whichever way they
    /// came.
    pub fn request(
        &mut self,
        key: RecordKey,
        make: impl FnOnce() -> Result<Vec<u8>, String> + Send + Sync + 'static,
        deliver: impl FnOnce(&mut World, &[u8]) -> Result<(), String> + Send + Sync + 'static,
    ) {
        self.requests.push(Request {
            key,
            make: Box::new(make),
            deliver: Box::new(deliver),
            job: None,
        });
    }

    /// Records asked for and not yet delivered.
    pub fn pending(&self) -> usize {
        self.requests.len()
    }

    /// Records that failed to make or deliver, and the last one's error.
    pub fn failures(&self) -> (u64, Option<&str>) {
        (
            self.failures.count,
            self.failures.last.as_ref().map(|(_, error)| error.as_str()),
        )
    }
}

/// Twice a frame, before its presentation and before the world follows the
/// camera ([`crate::WorldSync`]), so frames stepped without drawing and
/// paused frames drawing alone both take them: hands the host the records
/// asked
/// for since, and delivers those it answered: its bytes, or where it held
/// none (or failed) the bytes their code makes here, which it keeps.
pub(crate) fn exchange(world: &mut World) {
    let requests = {
        let mut records = world.resource_mut::<Records>();
        records.frame += 1;
        std::mem::take(&mut records.requests)
    };
    if requests.is_empty() {
        if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
            cooking.left("records", 0);
        }
        return;
    }
    let mut waiting = Vec::with_capacity(requests.len());
    let mut arrived = Vec::new();
    {
        let mut jobs = world.resource_mut::<Jobs>();
        for (place, mut request) in requests.into_iter().enumerate() {
            let id = *request.job.get_or_insert_with(|| {
                jobs.fetch(request.key, job_priority(URGENT_JOBS, place as u32))
            });
            match jobs.take(id) {
                Some(output) => arrived.push((request, output)),
                None => waiting.push(request),
            }
        }
    }
    let mut failed = Vec::new();
    for (request, output) in arrived {
        // Where no host holds it, it is made here and kept.
        let bytes = match output {
            Ok(bytes) => Ok(bytes),
            Err(_) => (request.make)().inspect(|bytes| {
                world
                    .resource_mut::<Jobs>()
                    .keep(request.key, bytes.clone());
            }),
        };
        if let Err(error) = bytes.and_then(|bytes| (request.deliver)(world, &bytes)) {
            failed.push(format!("record {}: {error}", request.key.hex()));
        }
    }
    let mut records = world.resource_mut::<Records>();
    let frame = records.frame;
    for error in failed {
        records.failures.record(frame, error);
    }
    // Requests the deliveries made wait behind those still waiting.
    waiting.append(&mut records.requests);
    records.requests = waiting;
    let left = records.requests.len() as u64;
    if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
        cooking.left("records", left);
    }
}
