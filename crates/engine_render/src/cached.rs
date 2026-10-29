use std::sync::atomic::{AtomicU64, Ordering};

fn next_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
pub(crate) struct Versioned<T> {
    value: T,
    generation: u64,
}

impl<T> Versioned<T> {
    pub(crate) fn new(value: T) -> Self {
        Self {
            value,
            generation: next_generation(),
        }
    }

    pub(crate) fn replace(&mut self, value: T)
    where
        T: 'static,
    {
        crate::memory::release_any(&self.value);
        self.value = value;
        self.generation = next_generation();
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
}

impl<T> std::ops::Deref for Versioned<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl Versioned<wgpu::Buffer> {
    pub(crate) fn bound(&self) -> Bound<'_> {
        Bound::of(self, self.value.as_entire_binding())
    }
}

impl Versioned<wgpu::TextureView> {
    pub(crate) fn bound(&self) -> Bound<'_> {
        Bound::of(self, wgpu::BindingResource::TextureView(&self.value))
    }
}

#[derive(Clone)]
pub(crate) struct Bound<'a> {
    generation: u64,
    resource: wgpu::BindingResource<'a>,
}

impl<'a> Bound<'a> {
    pub(crate) fn of<T>(versioned: &Versioned<T>, resource: wgpu::BindingResource<'a>) -> Self {
        Self {
            generation: versioned.generation,
            resource,
        }
    }

    pub(crate) fn fixed(resource: wgpu::BindingResource<'a>) -> Self {
        Self {
            generation: 0,
            resource,
        }
    }

    pub(crate) fn counted(generation: u64, resource: wgpu::BindingResource<'a>) -> Self {
        Self {
            generation,
            resource,
        }
    }
}

pub(crate) struct GroupCache<const N: usize> {
    cached: Cached<[u64; N], wgpu::BindGroup>,
}

impl<const N: usize> Default for GroupCache<N> {
    fn default() -> Self {
        Self {
            cached: Cached::default(),
        }
    }
}

impl<const N: usize> GroupCache<N> {
    pub(crate) fn bind(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        (table, layout): (&crate::bindings::Table, &crate::pipelines::GroupLayout),
        entries: [(&str, Bound<'_>); N],
    ) -> &wgpu::BindGroup {
        let key = entries.each_ref().map(|(_, bound)| bound.generation);
        self.cached.get_or_insert_with(key, || {
            let resources = entries.map(|(name, bound)| (name, bound.resource));
            crate::pipelines::group(device, label, table, layout, &resources)
        })
    }

    pub(crate) fn bind_filled(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        (table, layout): (&crate::bindings::Table, &crate::pipelines::GroupLayout),
        entries: [(&str, Bound<'_>); N],
        fillers: &crate::pipelines::Fillers,
    ) -> &wgpu::BindGroup {
        let key = entries.each_ref().map(|(_, bound)| bound.generation);
        self.cached.get_or_insert_with(key, || {
            let resources = entries.map(|(name, bound)| (name, bound.resource));
            crate::pipelines::filled_group(device, label, table, layout, &resources, fillers)
        })
    }

    pub(crate) fn get(&self) -> Option<&wgpu::BindGroup> {
        self.cached.get()
    }

    pub(crate) fn clear(&mut self) {
        self.cached.clear();
    }
}

#[derive(Debug)]
pub(crate) struct Cached<K, V> {
    entry: Option<(K, V)>,
}

impl<K, V> Default for Cached<K, V> {
    fn default() -> Self {
        Self { entry: None }
    }
}

impl<K: PartialEq, V> Cached<K, V> {
    pub(crate) fn stale(&self, key: &K) -> bool {
        self.entry.as_ref().is_none_or(|(built, _)| built != key)
    }

    pub(crate) fn set(&mut self, key: K, value: V) {
        self.entry = Some((key, value));
    }

    pub(crate) fn get_or_insert_with(&mut self, key: K, build: impl FnOnce() -> V) -> &V {
        let entry = match self.entry.take() {
            Some(entry) if entry.0 == key => entry,
            stale => {
                let built = (key, build());
                drop(stale);
                built
            }
        };
        &self.entry.insert(entry).1
    }
}

impl<K, V> Cached<K, V> {
    pub(crate) fn get(&self) -> Option<&V> {
        self.entry.as_ref().map(|(_, value)| value)
    }

    pub(crate) fn key(&self) -> Option<&K> {
        self.entry.as_ref().map(|(key, _)| key)
    }

    pub(crate) fn clear(&mut self) {
        self.entry = None;
    }
}
