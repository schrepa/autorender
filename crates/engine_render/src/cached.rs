//! Values derived from inputs a key identifies, rebuilt when the inputs
//! change: mostly bind groups over the buffers and textures of given
//! generations, which every pass keeps between frames. A [`Versioned`]
//! resource advances its generation whenever it is replaced, and a
//! [`GroupCache`] keys its group by the generations of exactly what it
//! binds, so no key leaves a bound resource out.

use std::sync::atomic::{AtomicU64, Ordering};

/// The next generation of any resource: unique among every resource the
/// renderer makes, so one replaced, or dropped and made again, never shows
/// a generation it had before.
fn next_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A resource and its generation, which replacing it advances: groups
/// binding it rebuild when it changes.
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

    /// Replaces the resource, advancing the generation.
    pub(crate) fn replace(&mut self, value: T) {
        self.value = value;
        self.generation = next_generation();
    }

    /// Advances when the resource is replaced.
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
    /// The whole buffer as a group binds it.
    pub(crate) fn bound(&self) -> Bound<'_> {
        Bound::of(self, self.value.as_entire_binding())
    }
}

impl Versioned<wgpu::TextureView> {
    /// The view as a group binds it.
    pub(crate) fn bound(&self) -> Bound<'_> {
        Bound::of(self, wgpu::BindingResource::TextureView(&self.value))
    }
}

/// A resource a group binds, with the generation of what it came from.
pub(crate) struct Bound<'a> {
    generation: u64,
    resource: wgpu::BindingResource<'a>,
}

impl<'a> Bound<'a> {
    /// `resource`, part of `versioned` (a range of a buffer, a view of a
    /// texture), as of its generation.
    pub(crate) fn of<T>(versioned: &Versioned<T>, resource: wgpu::BindingResource<'a>) -> Self {
        Self {
            generation: versioned.generation,
            resource,
        }
    }

    /// `resource`, made once and never replaced while a group binds it
    /// (a sampler, a fixed-size uniform).
    pub(crate) fn fixed(resource: wgpu::BindingResource<'a>) -> Self {
        Self {
            generation: 0,
            resource,
        }
    }

    /// `resource`, whose owner counts its replacements itself as
    /// `generation`.
    pub(crate) fn counted(generation: u64, resource: wgpu::BindingResource<'a>) -> Self {
        Self {
            generation,
            resource,
        }
    }
}

/// A bind group binding `N` resources by name, keyed by their generations:
/// it rebuilds whenever one of them was replaced, and as its key comes from
/// the entries it binds, the key cannot leave one out.
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
    /// The group binding `entries` through `layout`, the layout of
    /// `table`, built anew unless it binds them as they stand.
    pub(crate) fn bind(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        (table, layout): (&crate::bindings::Table, &wgpu::BindGroupLayout),
        entries: [(&str, Bound<'_>); N],
    ) -> &wgpu::BindGroup {
        let key = entries.each_ref().map(|(_, bound)| bound.generation);
        if self.cached.stale(&key) {
            let resources = entries.map(|(name, bound)| (name, bound.resource));
            let group = crate::pipelines::group(device, label, table, layout, &resources);
            self.cached.set(key, group);
        }
        self.cached.get().expect("built above")
    }

    /// [`Self::bind`] of a shared layout, `fillers` standing in for the
    /// bindings `entries` leave out.
    pub(crate) fn bind_filled(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        (table, layout): (&crate::bindings::Table, &wgpu::BindGroupLayout),
        entries: [(&str, Bound<'_>); N],
        fillers: &crate::pipelines::Fillers,
    ) -> &wgpu::BindGroup {
        let key = entries.each_ref().map(|(_, bound)| bound.generation);
        if self.cached.stale(&key) {
            let resources = entries.map(|(name, bound)| (name, bound.resource));
            let group =
                crate::pipelines::filled_group(device, label, table, layout, &resources, fillers);
            self.cached.set(key, group);
        }
        self.cached.get().expect("built above")
    }

    /// The group last built.
    pub(crate) fn get(&self) -> Option<&wgpu::BindGroup> {
        self.cached.get()
    }

    /// Forgets the group.
    pub(crate) fn clear(&mut self) {
        self.cached.clear();
    }
}

/// A value built for a key, rebuilt when asked for another key.
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
    /// Whether the value is missing or was built for another key.
    pub(crate) fn stale(&self, key: &K) -> bool {
        self.entry.as_ref().is_none_or(|(built, _)| built != key)
    }

    /// Keeps `value`, built for `key`.
    pub(crate) fn set(&mut self, key: K, value: V) {
        self.entry = Some((key, value));
    }

    /// The value for `key`, built by `build` unless it already was.
    pub(crate) fn get_or_insert_with(&mut self, key: K, build: impl FnOnce() -> V) -> &V {
        if self.stale(&key) {
            self.entry = Some((key, build()));
        }
        &self.entry.as_ref().expect("built above").1
    }
}

impl<K, V> Cached<K, V> {
    /// The value last built, whatever its key.
    pub(crate) fn get(&self) -> Option<&V> {
        self.entry.as_ref().map(|(_, value)| value)
    }

    /// The key the value was last built for.
    pub(crate) fn key(&self) -> Option<&K> {
        self.entry.as_ref().map(|(key, _)| key)
    }

    /// Forgets the value.
    pub(crate) fn clear(&mut self) {
        self.entry = None;
    }
}
