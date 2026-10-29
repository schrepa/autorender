use serde_json::Value;

use crate::protocol::{WatchRequest, WatchedComponent};
use crate::results::WatchRecord;

#[derive(Debug)]
pub struct Watcher {
    request: WatchRequest,
    stats: Vec<Option<Value>>,
    components: Vec<Option<Value>>,
    sent: u32,
}

impl Watcher {
    pub fn new(request: WatchRequest) -> Self {
        let stats = vec![None; request.stats.len()];
        let components = vec![None; request.entities.len()];
        Self {
            request,
            stats,
            components,
            sent: 0,
        }
    }

    pub fn frames(&self) -> u32 {
        self.request.frames
    }

    pub fn interval_ms(&self) -> f64 {
        self.request.interval_ms
    }

    pub fn sent(&self) -> u32 {
        self.sent
    }

    pub fn watched(&self) -> &[WatchedComponent] {
        &self.request.entities
    }

    pub fn changes(
        &mut self,
        (frame, tick): (Option<String>, Option<String>),
        stats: &Value,
        components: Vec<Value>,
    ) -> Vec<Value> {
        let mut records = Vec::new();
        for (index, path) in self.request.stats.iter().enumerate() {
            let value = stat(stats, path);
            if self.stats[index].as_ref() != Some(&value) {
                records.push(WatchRecord {
                    frame: frame.clone(),
                    tick: tick.clone(),
                    stat: Some(path.clone()),
                    entity: None,
                    component: None,
                    value: value.clone(),
                });
                self.stats[index] = Some(value);
            }
        }
        for (index, value) in components.into_iter().enumerate() {
            let Some(watched) = self.request.entities.get(index) else {
                break;
            };
            if self.components[index].as_ref() != Some(&value) {
                records.push(WatchRecord {
                    frame: frame.clone(),
                    tick: tick.clone(),
                    stat: None,
                    entity: Some(watched.entity.clone()),
                    component: Some(watched.component.clone()),
                    value: value.clone(),
                });
                self.components[index] = Some(value);
            }
        }
        self.sent += records.len() as u32;
        records
            .into_iter()
            .map(|record| serde_json::to_value(record).expect("record_serializes"))
            .collect()
    }
}

pub fn stat(stats: &Value, path: &str) -> Value {
    path.split('.')
        .try_fold(stats, |value, field| value.get(field))
        .cloned()
        .unwrap_or(Value::Null)
}
