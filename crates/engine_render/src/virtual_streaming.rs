//! Streaming of virtual geometry: which pages of the resident
//! virtual meshes sit in the GPU's page pool, which groups can draw from
//! them, and which pages to load, evict and place. Bookkeeping only:
//! `virtual_geometry` owns the pool and fetches pages.
//!
//! A group can draw when every page holding its clusters is resident and so
//! is every coarser group holding clusters made from it. Selection draws a
//! coarser cluster wherever a finer group cannot, so each surface is drawn
//! once at whatever detail is resident; the pinned pages, always resident,
//! cover every surface. Views request the groups they would refine into;
//! loads go coarser groups first, and pages unused for a while make room
//! when a budget or the device's limits leave none.

use engine_core::VirtualMesh;

/// No page in a slot, or no slot for a page.
pub(crate) const NO_PAGE: u32 = u32::MAX;
/// Frames a page must go unused before it makes room for another: longer
/// than the read-backs reporting use take.
const EVICT_AFTER: u64 = 8;
/// Frames a page whose load failed waits before it is requested again.
const RETRY_AFTER: u64 = 120;

/// What streaming needs of one mesh: its pages (the pinned ones first, and
/// how many of the first its words hold), each group's error, and each
/// cluster's group, source group and page.
#[derive(Clone, Debug, Default)]
pub(crate) struct MeshLinks {
    pub pages: u32,
    pub pinned: u32,
    pub held: u32,
    pub errors: Vec<f32>,
    pub clusters: Vec<(u32, Option<u32>, u32)>,
}

impl MeshLinks {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn of(mesh: &VirtualMesh) -> Self {
        Self {
            pages: mesh.page_count() as u32,
            pinned: mesh.pinned_pages() as u32,
            held: mesh.available_pages() as u32,
            errors: (0..mesh.group_count())
                .map(|group| mesh.group_sphere(group).error)
                .collect(),
            clusters: (0..mesh.cluster_count())
                .map(|cluster| {
                    let info = mesh.cluster(cluster);
                    (info.group, info.refined, info.page)
                })
                .collect(),
        }
    }
}

/// What a read-back frame's requests began: the pages to load, how many
/// pages it wanted and lacked, and how many of those found no room to load.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Requests {
    pub begun: Vec<u32>,
    pub missing: usize,
    pub dropped: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PageState {
    Absent,
    /// Waiting for its words.
    Loading,
    Resident(u32),
}

struct Page {
    mesh: u32,
    local: u32,
    pinned: bool,
    held: bool,
    state: PageState,
    last_used: u64,
    // Frame before which a failed load is not retried.
    retry_at: u64,
    // Groups with clusters in it.
    groups: Vec<u32>,
}

struct Group {
    // Pages holding its clusters.
    pages: Vec<u32>,
    // Coarser groups holding clusters made from it, and the finer groups
    // its clusters were made from.
    parents: Vec<u32>,
    children: Vec<u32>,
    resident: bool,
    error: f32,
}

/// Where a page to place goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    /// Into this slot, whose previous page (if any) was evicted.
    Slot(u32),
    /// Every slot is taken and more may be added ([`Residency::add_slots`]).
    Grow,
    /// Every slot holds a page used lately and no more may be added: the
    /// load is dropped.
    Full,
}

fn push_unique(list: &mut Vec<u32>, value: u32) {
    if list.last() != Some(&value) && !list.contains(&value) {
        list.push(value);
    }
}

fn bits(words: &[u32]) -> impl Iterator<Item = u32> + '_ {
    words.iter().enumerate().flat_map(|(index, &word)| {
        (0..32)
            .filter(move |bit| word & (1 << bit) != 0)
            .map(move |bit| index as u32 * 32 + bit)
    })
}

#[derive(Default)]
pub(crate) struct Residency {
    pages: Vec<Page>,
    groups: Vec<Group>,
    // Page in each slot, or NO_PAGE.
    slots: Vec<u32>,
    free: Vec<u32>,
    resident: usize,
    loading: usize,
    bits_dirty: bool,
    table_dirty: bool,
}

impl Residency {
    /// Every page of `meshes` absent and every group linked; each mesh's
    /// global groups and pages start at its `bases` entry. Slots stay as
    /// they are, all free.
    pub(crate) fn rebuild(&mut self, meshes: &[MeshLinks], bases: &[(u32, u32)]) {
        self.pages.clear();
        self.groups.clear();
        for (mesh_index, (mesh, &(group_base, page_base))) in meshes.iter().zip(bases).enumerate() {
            debug_assert_eq!(self.pages.len(), page_base as usize);
            debug_assert_eq!(self.groups.len(), group_base as usize);
            self.pages.extend((0..mesh.pages).map(|local| Page {
                mesh: mesh_index as u32,
                local,
                pinned: local < mesh.pinned,
                held: local < mesh.held,
                state: PageState::Absent,
                last_used: 0,
                retry_at: 0,
                groups: Vec::new(),
            }));
            self.groups.extend(mesh.errors.iter().map(|&error| Group {
                pages: Vec::new(),
                parents: Vec::new(),
                children: Vec::new(),
                resident: false,
                error,
            }));
            for &(group, refined, page) in &mesh.clusters {
                let (group, page) = (group_base + group, page_base + page);
                push_unique(&mut self.groups[group as usize].pages, page);
                push_unique(&mut self.pages[page as usize].groups, group);
                if let Some(refined) = refined {
                    let refined = group_base + refined;
                    push_unique(&mut self.groups[refined as usize].parents, group);
                    push_unique(&mut self.groups[group as usize].children, refined);
                }
            }
        }
        // The rebuilt pool's slots follow (`add_slots`).
        self.slots.clear();
        self.free.clear();
        self.resident = 0;
        self.loading = 0;
        self.bits_dirty = true;
        self.table_dirty = true;
    }

    pub(crate) fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Page `page`'s mesh, index in its mesh, and whether the mesh's words
    /// hold it.
    pub(crate) fn page(&self, page: u32) -> (u32, u32, bool) {
        let page = &self.pages[page as usize];
        (page.mesh, page.local, page.held)
    }

    pub(crate) fn state(&self, page: u32) -> PageState {
        self.pages[page as usize].state
    }

    /// Pages that must always be resident.
    pub(crate) fn pinned(&self) -> Vec<u32> {
        (0..self.pages.len() as u32)
            .filter(|&page| self.pages[page as usize].pinned)
            .collect()
    }

    /// Pages the meshes' words hold.
    pub(crate) fn held(&self) -> Vec<u32> {
        (0..self.pages.len() as u32)
            .filter(|&page| self.pages[page as usize].held)
            .collect()
    }

    pub(crate) fn resident_pages(&self) -> usize {
        self.resident
    }

    pub(crate) fn loading_pages(&self) -> usize {
        self.loading
    }

    /// Each resident page and its slot.
    pub(crate) fn resident_slots(&self) -> Vec<(u32, u32)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|&(_, &page)| page != NO_PAGE)
            .map(|(slot, &page)| (page, slot as u32))
            .collect()
    }

    /// Marks each absent page of `pages` resident in its free slot, whose
    /// words a rebuilt pool kept, as used at `frame`.
    pub(crate) fn restore(&mut self, pages: &[(u32, u32)], frame: u64) {
        let mut groups = Vec::new();
        for &(page, slot) in pages {
            let (Some(entry), Some(held)) = (
                self.pages.get_mut(page as usize),
                self.slots.get_mut(slot as usize),
            ) else {
                continue;
            };
            if entry.state != PageState::Absent || *held != NO_PAGE {
                continue;
            }
            entry.state = PageState::Resident(slot);
            entry.last_used = frame;
            *held = page;
            self.resident += 1;
            groups.extend_from_slice(&entry.groups);
        }
        if groups.is_empty() {
            return;
        }
        self.free = (0..self.slots.len() as u32)
            .rev()
            .filter(|&slot| self.slots[slot as usize] == NO_PAGE)
            .collect();
        self.table_dirty = true;
        self.refresh(groups);
    }

    /// Adds `count` free slots (the pool grew).
    pub(crate) fn add_slots(&mut self, count: usize) {
        let start = self.slots.len() as u32;
        self.slots.extend(std::iter::repeat_n(NO_PAGE, count));
        self.free.extend((start..start + count as u32).rev());
    }

    /// Marks page `page` loading, unless it is not absent or waits to retry.
    pub(crate) fn begin(&mut self, page: u32, frame: u64) -> bool {
        let entry = &mut self.pages[page as usize];
        if entry.state != PageState::Absent || frame < entry.retry_at {
            return false;
        }
        entry.state = PageState::Loading;
        self.loading += 1;
        true
    }

    /// Marks the groups set in `used` (a bitset of global groups) used at
    /// `frame`, and begins loading, coarser groups first, the absent pages
    /// of the groups set in `requested` and of every coarser group they
    /// need, as many as have room while at most `limit` pages are resident.
    pub(crate) fn feedback(
        &mut self,
        used: &[u32],
        requested: &[u32],
        frame: u64,
        limit: usize,
    ) -> Requests {
        for group in bits(used) {
            if let Some(group) = self.groups.get(group as usize) {
                for &page in &group.pages {
                    let page = &mut self.pages[page as usize];
                    page.last_used = page.last_used.max(frame);
                }
            }
        }
        let mut seen = vec![false; self.groups.len()];
        let mut needed = Vec::new();
        let mut stack: Vec<u32> = bits(requested)
            .filter(|&group| (group as usize) < self.groups.len())
            .collect();
        while let Some(group) = stack.pop() {
            if std::mem::replace(&mut seen[group as usize], true)
                || self.groups[group as usize].resident
            {
                continue;
            }
            needed.push(group);
            stack.extend(&self.groups[group as usize].parents);
        }
        needed.sort_by(|a, b| {
            self.groups[*b as usize]
                .error
                .total_cmp(&self.groups[*a as usize].error)
        });
        let mut room = self.room(frame, limit);
        let mut counted = vec![false; self.pages.len()];
        let mut requests = Requests::default();
        for group in needed {
            for index in 0..self.groups[group as usize].pages.len() {
                let page = self.groups[group as usize].pages[index];
                if std::mem::replace(&mut counted[page as usize], true)
                    || matches!(self.pages[page as usize].state, PageState::Resident(_))
                {
                    continue;
                }
                requests.missing += 1;
                let entry = &self.pages[page as usize];
                if entry.state != PageState::Absent || frame < entry.retry_at {
                    continue;
                }
                if room == 0 {
                    requests.dropped += 1;
                    continue;
                }
                room -= 1;
                self.begin(page, frame);
                requests.begun.push(page);
            }
        }
        requests
    }

    /// How many more pages can load at `frame` while at most `limit` are
    /// resident: the room under it and the pages unused long enough to
    /// make room, less the pages already loading.
    fn room(&self, frame: u64, limit: usize) -> usize {
        let evictable = self
            .slots
            .iter()
            .filter(|&&held| held != NO_PAGE && self.evictable(held, frame))
            .count();
        limit
            .saturating_sub(self.resident)
            .saturating_add(evictable)
            .saturating_sub(self.loading)
    }

    /// Whether resident page `page` may make room at `frame`.
    fn evictable(&self, page: u32, frame: u64) -> bool {
        let page = &self.pages[page as usize];
        !page.pinned && page.last_used + EVICT_AFTER < frame
    }

    /// A slot for loading page `page` at `frame` while at most `limit`
    /// pages may be resident: a free one, a new one ([`Placement::Grow`]),
    /// or the least recently used unpinned page's, evicted. The page is
    /// resident in it; [`Placement::Full`] drops the load.
    pub(crate) fn place(&mut self, page: u32, frame: u64, limit: usize) -> Placement {
        let slot = if self.resident < limit {
            match self.free.pop() {
                Some(slot) => slot,
                None => return Placement::Grow,
            }
        } else {
            let victim = self
                .slots
                .iter()
                .enumerate()
                .filter(|&(_, &held)| held != NO_PAGE && self.evictable(held, frame))
                .min_by_key(|&(_, &held)| self.pages[held as usize].last_used)
                .map(|(slot, &held)| (slot as u32, held));
            match victim {
                Some((slot, held)) => {
                    self.evict(held);
                    self.free.retain(|&free| free != slot);
                    slot
                }
                None => {
                    self.fail(page, frame, 0);
                    return Placement::Full;
                }
            }
        };
        let entry = &mut self.pages[page as usize];
        if entry.state == PageState::Loading {
            self.loading -= 1;
        }
        entry.state = PageState::Resident(slot);
        entry.last_used = entry.last_used.max(frame);
        self.slots[slot as usize] = page;
        self.resident += 1;
        self.table_dirty = true;
        let groups = self.pages[page as usize].groups.clone();
        self.refresh(groups);
        Placement::Slot(slot)
    }

    /// Page `page`'s load failed (or was dropped) at `frame`: it waits
    /// `wait` frames (or the retry delay, when 0) before loading again.
    pub(crate) fn fail(&mut self, page: u32, frame: u64, wait: u64) {
        let entry = &mut self.pages[page as usize];
        if entry.state == PageState::Loading {
            self.loading -= 1;
        }
        if !matches!(entry.state, PageState::Resident(_)) {
            entry.state = PageState::Absent;
            entry.retry_at = frame + if wait == 0 { RETRY_AFTER } else { wait };
        }
    }

    /// Evicts the least recently used unpinned pages, used lately or not,
    /// until at most `limit` are resident (pinned ones stay).
    pub(crate) fn shrink(&mut self, limit: usize) {
        while self.resident > limit {
            let victim = self
                .slots
                .iter()
                .filter(|&&held| held != NO_PAGE && !self.pages[held as usize].pinned)
                .min_by_key(|&&held| self.pages[held as usize].last_used)
                .copied();
            let Some(victim) = victim else {
                break;
            };
            let PageState::Resident(slot) = self.pages[victim as usize].state else {
                unreachable!("slots hold resident pages");
            };
            self.evict(victim);
            self.free.push(slot);
        }
    }

    fn evict(&mut self, page: u32) {
        let PageState::Resident(slot) = self.pages[page as usize].state else {
            return;
        };
        self.pages[page as usize].state = PageState::Absent;
        self.slots[slot as usize] = NO_PAGE;
        self.resident -= 1;
        self.table_dirty = true;
        let groups = self.pages[page as usize].groups.clone();
        self.refresh(groups);
    }

    /// Recomputes the residency of `work`'s groups and, wherever it
    /// changes, of the finer groups depending on them.
    fn refresh(&mut self, mut work: Vec<u32>) {
        while let Some(group) = work.pop() {
            let entry = &self.groups[group as usize];
            let resident = entry
                .pages
                .iter()
                .all(|&page| matches!(self.pages[page as usize].state, PageState::Resident(_)))
                && entry
                    .parents
                    .iter()
                    .all(|&parent| self.groups[parent as usize].resident);
            if resident != entry.resident {
                self.groups[group as usize].resident = resident;
                self.bits_dirty = true;
                work.extend(&self.groups[group as usize].children);
            }
        }
    }

    /// The groups' residency as bits, and whether it changed since the last
    /// call.
    pub(crate) fn take_bits(&mut self) -> Option<Vec<u32>> {
        std::mem::take(&mut self.bits_dirty).then(|| {
            let mut words = vec![0u32; self.groups.len().div_ceil(32)];
            for (index, group) in self.groups.iter().enumerate() {
                if group.resident {
                    words[index / 32] |= 1 << (index % 32);
                }
            }
            words
        })
    }

    /// Each page's slot (NO_PAGE when absent), and whether it changed since the
    /// last call.
    pub(crate) fn take_table(&mut self) -> Option<Vec<u32>> {
        std::mem::take(&mut self.table_dirty).then(|| {
            self.pages
                .iter()
                .map(|page| match page.state {
                    PageState::Resident(slot) => slot,
                    _ => NO_PAGE,
                })
                .collect()
        })
    }
}
