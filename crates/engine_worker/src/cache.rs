use engine_math::collections::HashMap;

#[derive(Clone, Copy, Debug)]
struct Entry {
    bytes: u64,
    used: f64,
}

#[derive(Debug)]
pub struct JobCache {
    budget: u64,
    verify: bool,
    entries: HashMap<String, Entry>,
    total: u64,
    hits: u64,
    misses: u64,
    mismatches: u64,
    changed: bool,
    broken: bool,
    removed: Vec<String>,
}

impl JobCache {
    pub fn open(budget: u64, available: Option<u64>, verify: bool, index: Option<&str>) -> Self {
        let entries = index.map(read_index).unwrap_or_default();
        let total: u64 = entries.values().map(|entry| entry.bytes).sum();
        let budget = available.map_or(budget, |free| budget.min(free.saturating_add(total) / 2));
        let mut cache = Self {
            budget,
            verify,
            entries,
            total,
            hits: 0,
            misses: 0,
            mismatches: 0,
            changed: false,
            broken: false,
            removed: Vec::new(),
        };
        cache.evict();
        cache
    }

    pub fn verify(&self) -> bool {
        self.verify
    }

    pub fn counters(&self) -> (u64, u64, u64, u64) {
        (self.hits, self.misses, self.total, self.mismatches)
    }

    pub fn lists(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn holds(&self, key: &str) -> bool {
        !self.broken && self.entries.contains_key(key)
    }

    pub fn hit(&mut self, key: &str, bytes: Option<u64>) -> bool {
        match self.entries.get_mut(key) {
            Some(entry) if bytes == Some(entry.bytes) && !self.broken => {
                entry.used = wall_ms();
                self.changed = true;
                self.hits += 1;
                true
            }
            _ => {
                self.remove(key);
                self.misses += 1;
                false
            }
        }
    }

    pub fn miss(&mut self) {
        self.misses += 1;
    }

    pub fn store(&mut self, key: &str, bytes: u64) -> bool {
        if self.broken || bytes > self.budget {
            return false;
        }
        let replaced = self.entries.insert(
            key.to_owned(),
            Entry {
                bytes,
                used: wall_ms(),
            },
        );
        self.total = self.total + bytes - replaced.map_or(0, |entry| entry.bytes);
        self.changed = true;
        self.evict();
        true
    }

    pub fn mismatch(&mut self, key: &str) {
        self.mismatches += 1;
        self.remove(key);
    }

    pub fn failed(&mut self) {
        self.broken = true;
    }

    pub fn take_removed(&mut self) -> Vec<String> {
        std::mem::take(&mut self.removed)
    }

    pub fn index(&mut self) -> Option<String> {
        if !self.changed {
            return None;
        }
        self.changed = false;
        let index: serde_json::Map<String, serde_json::Value> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), serde_json::json!([entry.bytes, entry.used])))
            .collect();
        Some(serde_json::Value::Object(index).to_string())
    }

    fn evict(&mut self) {
        if self.total <= self.budget {
            return;
        }
        let mut oldest: Vec<(String, f64)> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), entry.used))
            .collect();
        oldest.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (key, _) in oldest {
            if self.total <= self.budget {
                break;
            }
            self.remove(&key);
        }
    }

    fn remove(&mut self, key: &str) {
        if let Some(entry) = self.entries.remove(key) {
            self.total -= entry.bytes;
            self.changed = true;
            self.removed.push(key.to_owned());
        }
    }
}

fn read_index(index: &str) -> HashMap<String, Entry> {
    let Ok(index) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(index)
    else {
        return HashMap::default();
    };
    index
        .into_iter()
        .filter_map(|(key, value)| {
            let valid = key.len() == 64
                && key
                    .bytes()
                    .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
            let bytes = value.get(0)?.as_u64()?;
            let used = value.get(1)?.as_f64()?;
            valid.then_some((key, Entry { bytes, used }))
        })
        .collect()
}

fn wall_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |elapsed| elapsed.as_secs_f64() * 1000.0)
    }
}
