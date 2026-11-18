use super::numbers::Level;
use super::places::ZoneId;
use super::team::{Approach, Duty, Step};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Activity {
    Dead,
    Wait,
    Move,
    Hold,
    Fight,
    Hunt,
    Retreat,
    Reload,
    Plant,
    Defuse,
    Utility,
    Save,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Why {
    RoundStart,
    Order,
    Spotted,
    Heard,
    Called,
    LostTarget,
    Reload,
    LowHealth,
    Blinded,
    Stuck,
    Arrived,
    MateInLine,
    Planting,
    Defusing,
    Saving,
    Utility,
    Hesitating,
    TimeShort,
    PlanDied,
    HumanLead,
    InLine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BotDebug {
    pub level: Level,
    pub duty: Duty,
    pub step: Step,
    pub activity: Activity,
    pub goal_zone: Option<ZoneId>,
    pub target: Option<u32>,
    pub why: Why,
    pub since_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SquadDebug {
    pub approach: Approach,
    pub site: Option<usize>,
    pub step: Step,
    pub why: Why,
    pub since_tick: u64,
}
