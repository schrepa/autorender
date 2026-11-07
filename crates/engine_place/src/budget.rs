use std::time::Duration;

use bevy_ecs::resource::Resource;
use serde::Serialize;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BudgetUse {
    pub by: String,
    pub asked: u64,
    pub spent: u64,
    pub cut: u64,
}

#[derive(Resource, Clone, Debug, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FrameBudget {
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    clock: Option<fn() -> f64>,
    deadline: f64,
    upload_left: u64,
    uses: Vec<BudgetUse>,
}

impl FrameBudget {
    #[doc(hidden)]
    pub fn start(clock: Option<fn() -> f64>, millis: f64, upload_bytes: u64) -> Self {
        Self {
            clock,
            deadline: clock.map_or(0.0, |clock| clock() + millis),
            upload_left: upload_bytes,
            uses: Vec::new(),
        }
    }

    pub fn spend(&mut self, by: &str, bytes: u64) -> bool {
        let fits = bytes <= self.upload_left;
        if fits {
            self.upload_left -= bytes;
        }
        let used = match self.uses.iter().position(|used| used.by == by) {
            Some(at) => &mut self.uses[at],
            None => {
                self.uses.push(BudgetUse {
                    by: by.to_owned(),
                    ..BudgetUse::default()
                });
                let last = self.uses.len() - 1;
                &mut self.uses[last]
            }
        };
        used.asked = used.asked.saturating_add(bytes);
        if fits {
            used.spent = used.spent.saturating_add(bytes);
        } else {
            used.cut = used.cut.saturating_add(bytes);
        }
        fits
    }

    pub fn time_left(&self) -> Duration {
        match self.clock {
            Some(clock) => Duration::from_secs_f64(((self.deadline - clock()) / 1000.0).max(0.0)),
            None => Duration::MAX,
        }
    }

    pub fn expired(&self) -> bool {
        self.time_left().is_zero()
    }

    pub fn upload_left(&self) -> u64 {
        self.upload_left
    }

    pub fn uses(&self) -> &[BudgetUse] {
        &self.uses
    }
}
