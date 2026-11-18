#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod attributed;
mod entries;
mod jobs;
mod members;
mod record;
mod registry;
mod scene;
mod services;
mod state;

pub use engine_place::CORE_ROW;
pub use entries::{JoinEntry, Owned, SettingsEntry, StatsEntry, parse_section};
#[cfg(feature = "profiling")]
pub use entries::names::{
    Answer, Kind, Lands, NameEntry, Reach, ReachMut, unavailable,
};
pub use jobs::{
    JobBatch, JobKindRef, JobList, JobListed, JobMember, JobOrder, JobOrders, JobQueue,
    RecordParity, Shares,
};
pub use members::MembersOf;
pub use record::{BrickRecord, RecordExt};
pub use registry::{BrickId, Registry, RegistryBuilder, settings};
#[cfg(feature = "profiling")]
pub use registry::register;
pub use scene::{SceneRead, SpawnedBy, authored_entity, authored_id};
pub use services::FrameFacts;
pub use state::{BrickStates, Despawn, SwitchFn, Switches};
