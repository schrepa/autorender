#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod attributed;
mod entries;
mod members;
mod record;
mod registry;
mod scene;
mod services;
mod state;

pub use engine_place::CORE_ROW;
#[cfg(feature = "profiling")]
pub use entries::names::{Answer, Kind, Lands, NameEntry, Reach, ReachMut, unavailable};
pub use entries::{Owned, SettingsEntry, StatsEntry, parse_section};
pub use members::MembersOf;
pub use record::{BrickRecord, RecordExt};
#[cfg(feature = "profiling")]
pub use registry::register;
pub use registry::{BrickId, Registry, RegistryBuilder, settings};
pub use scene::{SceneRead, SpawnedBy};
pub use services::FrameFacts;
pub use state::{BrickStates, Despawn, SwitchFn, Switches};
