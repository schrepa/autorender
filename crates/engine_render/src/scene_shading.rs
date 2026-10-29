use std::cell::RefCell;
use std::hash::{Hash, Hasher};

use crate::parts::Shading;

type Values = &'static [(&'static str, f64)];

#[derive(Clone, Copy, Debug)]
pub struct SceneShading(Values);

thread_local! {
    static KNOWN: RefCell<Vec<Values>> = const { RefCell::new(Vec::new()) };
}

fn same(first: &[(&'static str, f64)], second: &[(&'static str, f64)]) -> bool {
    first.len() == second.len()
        && first
            .iter()
            .zip(second)
            .all(|(&(name, value), &(other, known))| {
                name == other && value.to_bits() == known.to_bits()
            })
}

impl PartialEq for SceneShading {
    fn eq(&self, other: &Self) -> bool {
        same(self.0, other.0)
    }
}

impl Eq for SceneShading {}

impl Hash for SceneShading {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &(name, value) in self.0 {
            name.hash(state);
            value.to_bits().hash(state);
        }
    }
}

impl Default for SceneShading {
    fn default() -> Self {
        Self::NONE
    }
}

impl SceneShading {
    pub const NONE: Self = Self(&[]);

    pub fn new(values: &[(&'static str, f64)]) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(|first, second| first.0.cmp(second.0));
        sorted.dedup_by(|later, earlier| later.0 == earlier.0);
        KNOWN.with_borrow_mut(|known| {
            if let Some(&found) = known.iter().find(|found| same(found, &sorted)) {
                return Self(found);
            }
            let leaked: Values = Box::leak(sorted.into_boxed_slice());
            known.push(leaked);
            Self(leaked)
        })
    }

    pub fn of_parts(shading: &Shading) -> (Self, Vec<Self>) {
        let set = Self::new(&shading.values);
        let warm = shading
            .warm
            .iter()
            .map(|&entry| set.warmed(entry))
            .filter(|warmed| *warmed != set)
            .collect();
        (set, warm)
    }

    #[must_use]
    pub fn warmed(self, (name, value): (&'static str, f64)) -> Self {
        let values: Vec<(&'static str, f64)> = std::iter::once((name, value))
            .chain(self.0.iter().copied().filter(|&(known, _)| known != name))
            .collect();
        Self::new(&values)
    }

    pub fn constants(self) -> &'static [(&'static str, f64)] {
        self.0
    }

    pub fn value(self, name: &str) -> Option<f64> {
        self.0
            .iter()
            .find(|&&(known, _)| known == name)
            .map(|&(_, value)| value)
    }

    pub fn with(self, constants: &[(&'static str, f64)]) -> Vec<(&'static str, f64)> {
        constants
            .iter()
            .copied()
            .chain(self.0.iter().copied())
            .collect()
    }
}
