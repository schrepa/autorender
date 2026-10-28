use std::any::{Any, TypeId, type_name};
use std::fmt;

trait ExtractedValue: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn cloned(&self) -> Box<dyn ExtractedValue>;
    fn copy_into(&self, target: &mut dyn Any) -> bool;
}

impl<T: Clone + Send + Sync + 'static> ExtractedValue for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtractedValue> {
        Box::new(self.clone())
    }

    fn copy_into(&self, target: &mut dyn Any) -> bool {
        match target.downcast_mut::<T>() {
            Some(target) => {
                target.clone_from(self);
                true
            }
            None => false,
        }
    }
}

struct Entry {
    type_id: TypeId,
    name: &'static str,
    value: Box<dyn ExtractedValue>,
}

#[derive(Default)]
pub struct Extracted {
    entries: Vec<Entry>,
}

impl Extracted {
    pub fn insert<T: Clone + Send + Sync + 'static>(&mut self, value: T) {
        let type_id = TypeId::of::<T>();
        match self.entries.iter_mut().find(|entry| entry.type_id == type_id) {
            Some(entry) => entry.value = Box::new(value),
            None => self.entries.push(Entry {
                type_id,
                name: type_name::<T>(),
                value: Box::new(value),
            }),
        }
    }

    pub fn get<T: 'static>(&self) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.entries
            .iter()
            .find(|entry| entry.type_id == type_id)
            .and_then(|entry| entry.value.as_ref().as_any().downcast_ref::<T>())
    }

    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.entries
            .iter_mut()
            .find(|entry| entry.type_id == type_id)
            .and_then(|entry| entry.value.as_mut().as_any_mut().downcast_mut::<T>())
    }

    pub fn remove<T: 'static>(&mut self) {
        let type_id = TypeId::of::<T>();
        self.entries.retain(|entry| entry.type_id != type_id);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Clone for Extracted {
    fn clone(&self) -> Self {
        let mut copy = Self::default();
        copy.clone_from(self);
        copy
    }

    fn clone_from(&mut self, source: &Self) {
        self.entries
            .retain(|entry| source.entries.iter().any(|kept| kept.type_id == entry.type_id));
        for kept in &source.entries {
            match self
                .entries
                .iter_mut()
                .find(|entry| entry.type_id == kept.type_id)
            {
                Some(entry) => {
                    if !kept
                        .value
                        .as_ref()
                        .copy_into(entry.value.as_mut().as_any_mut())
                    {
                        entry.value = kept.value.as_ref().cloned();
                    }
                }
                None => self.entries.push(Entry {
                    type_id: kept.type_id,
                    name: kept.name,
                    value: kept.value.as_ref().cloned(),
                }),
            }
        }
    }
}

impl fmt::Debug for Extracted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_list()
            .entries(self.entries.iter().map(|entry| entry.name))
            .finish()
    }
}
