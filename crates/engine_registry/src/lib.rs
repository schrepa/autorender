#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod access;
mod attribution;
mod entries;
mod hooks;
mod jobs;
#[cfg(feature = "profiling")]
mod measured;
mod members;
mod order;
mod owners;
mod record;
mod registry;
mod scene;
mod services;
mod state;

pub use access::{AccessRule, CORE_CRATES, CORE_INTERNAL_CRATES, MoverRule};
pub use attribution::{
    BrickCost, BrickCosts, BrickRow, BrickTable, CORE_ROW, GPU_IDLE_ROW, UNATTRIBUTED_ROW,
};
pub use entries::{
    Filled, JoinEntry, OperationEntry, Owned, PieceEntry, SettingsEntry, StatsEntry, access_of,
    parse_section, set_section,
};
pub use hooks::{BrickHooks, NamedField, Removal};
pub use jobs::{
    JobBatch, JobKindRef, JobList, JobListed, JobMember, JobOrder, JobOrders, JobQueue,
    RecordParity, Shares,
};
pub use members::MembersOf;
pub use owners::{Shared, ValueKind};
pub use record::{BrickRecord, PlainSystems, RecordExt, SectionRule};
pub use registry::{
    BrickId, BrickSet, Registry, RegistryBuilder, check_sections, run_operation, settings,
};
pub use scene::{
    AuthoredId, OperationScene, SceneEntity, SceneRead, SpawnedBy, authored_entity, authored_id,
};
pub use services::FrameFacts;
pub use state::{
    BrickState, BrickStates, Despawn, GateDriver, GateFn, GateId, Scope, ask_on, fault, fault_of,
};
