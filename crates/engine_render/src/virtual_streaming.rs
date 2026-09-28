//! Streaming of virtual geometry (decision 044): which pages of the resident
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
pub(crate) const NONE: u32 = u32::MAX;
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
    // Page in each slot, or NONE.
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
            .filter(|&(_, &page)| page != NONE)
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
            if entry.state != PageState::Absent || *held != NONE {
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
            .filter(|&slot| self.slots[slot as usize] == NONE)
            .collect();
        self.table_dirty = true;
        self.refresh(groups);
    }

    /// Adds `count` free slots (the pool grew).
    pub(crate) fn add_slots(&mut self, count: usize) {
        let start = self.slots.len() as u32;
        self.slots.extend(std::iter::repeat_n(NONE, count));
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
            .filter(|&&held| held != NONE && self.evictable(held, frame))
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
                .filter(|&(_, &held)| held != NONE && self.evictable(held, frame))
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
                .filter(|&&held| held != NONE && !self.pages[held as usize].pinned)
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
        self.slots[slot as usize] = NONE;
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

    /// Whether group `group` can draw.
    #[cfg(test)]
    fn group_resident(&self, group: u32) -> bool {
        self.groups[group as usize].resident
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

    /// Each page's slot (NONE when absent), and whether it changed since the
    /// last call.
    pub(crate) fn take_table(&mut self) -> Option<Vec<u32>> {
        std::mem::take(&mut self.table_dirty).then(|| {
            self.pages
                .iter()
                .map(|page| match page.state {
                    PageState::Resident(slot) => slot,
                    _ => NONE,
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three levels over one surface: fine groups 0 and 1 (errors 1 and 2,
    /// pages 2 and 3), made into group 2 (error 5, page 1), made into group
    /// 3 (error MAX, pinned page 0).
    fn chain_links() -> MeshLinks {
        MeshLinks {
            pages: 4,
            pinned: 1,
            held: 1,
            errors: vec![1.0, 2.0, 5.0, f32::MAX],
            clusters: vec![
                (0, None, 2),
                (1, None, 3),
                (2, Some(0), 1),
                (2, Some(1), 1),
                (3, Some(2), 0),
            ],
        }
    }

    fn chain() -> Residency {
        let mut residency = Residency::default();
        residency.rebuild(&[chain_links()], &[(0, 0)]);
        residency
    }

    fn set(groups: &[u32]) -> Vec<u32> {
        let mut words = vec![0u32; 1];
        for &group in groups {
            words[0] |= 1 << group;
        }
        words
    }

    #[test]
    fn groups_draw_once_their_pages_and_coarser_groups_are_resident() {
        let mut residency = chain();
        residency.add_slots(2);
        assert_eq!(residency.group_count(), 4);
        assert_eq!((residency.pinned(), residency.held()), (vec![0], vec![0]));
        assert_eq!(residency.page(2), (0, 2, false));
        assert!(residency.begin(0, 0));
        assert_eq!(residency.place(0, 0, usize::MAX), Placement::Slot(0));
        assert!(residency.group_resident(3) && !residency.group_resident(2));
        assert_eq!(residency.take_bits(), Some(vec![0b1000]));
        // Requesting a fine group loads its page and its coarser group's,
        // coarser first.
        let requests = residency.feedback(&set(&[]), &set(&[0]), 1, usize::MAX);
        assert_eq!(
            requests,
            Requests {
                begun: vec![1, 2],
                missing: 2,
                dropped: 0
            }
        );
        assert_eq!(residency.loading_pages(), 2);
        // Asked again while loading, they are still missing.
        let requests = residency.feedback(&set(&[]), &set(&[0]), 2, usize::MAX);
        assert_eq!((requests.begun.len(), requests.missing), (0, 2));
        // The fine page alone draws nothing: its coarser group is absent.
        assert_eq!(residency.place(2, 1, usize::MAX), Placement::Slot(1));
        assert!(!residency.group_resident(0));
        assert_eq!(residency.place(1, 1, usize::MAX), Placement::Grow);
        residency.add_slots(2);
        assert_eq!(residency.place(1, 1, usize::MAX), Placement::Slot(2));
        assert!(residency.group_resident(2) && residency.group_resident(0));
        assert!(!residency.group_resident(1));
        assert_eq!(residency.take_bits(), Some(vec![0b1101]));
        assert_eq!(residency.take_table().unwrap(), [0, 2, 1, NONE]);
        assert_eq!(residency.take_table(), None);
    }

    #[test]
    fn rebuilds_start_a_new_pool_and_restore_kept_pages() {
        let mut residency = chain();
        residency.add_slots(4);
        for page in [0, 1, 2] {
            assert!(residency.begin(page, 0));
            assert!(matches!(residency.place(page, 0, 4), Placement::Slot(_)));
        }
        let kept = residency.resident_slots();
        assert_eq!(kept.len(), 3);
        // A rebuilt layout has only the slots its pool adds: no slot of the
        // old pool is free for placing pages.
        residency.rebuild(&[chain_links()], &[(0, 0)]);
        assert_eq!(
            (residency.resident_pages(), residency.resident_slots()),
            (0, vec![])
        );
        residency.add_slots(1);
        assert!(residency.begin(0, 1));
        assert_eq!(residency.place(0, 1, usize::MAX), Placement::Slot(0));
        assert!(residency.begin(3, 1));
        assert_eq!(residency.place(3, 1, usize::MAX), Placement::Grow);
        // Pages whose words the new pool kept return to their slots, and
        // their groups draw again.
        residency.rebuild(&[chain_links()], &[(0, 0)]);
        residency.add_slots(4);
        residency.restore(&kept, 2);
        assert_eq!(residency.resident_pages(), 3);
        for (page, slot) in kept {
            assert_eq!(residency.state(page), PageState::Resident(slot));
        }
        assert!(residency.group_resident(3) && residency.group_resident(2));
        assert!(residency.group_resident(0) && !residency.group_resident(1));
        assert!(residency.begin(3, 2));
        assert!(matches!(
            residency.place(3, 2, usize::MAX),
            Placement::Slot(3)
        ));
    }

    #[test]
    fn full_pools_evict_pages_unused_lately_and_never_pinned_ones() {
        let mut residency = chain();
        residency.add_slots(3);
        for page in [0, 1, 2] {
            assert!(residency.begin(page, 0));
            assert!(matches!(residency.place(page, 0, 3), Placement::Slot(_)));
        }
        // Groups 0 and 2 (pages 2 and 1) were used at frame 10: nothing
        // makes room for group 1's page yet, so it does not load.
        let requests = residency.feedback(&set(&[0, 2]), &set(&[1]), 10, 3);
        assert_eq!((requests.begun.len(), requests.missing), (0, 1));
        assert_eq!(requests.dropped, 1);
        assert!(residency.begin(3, 12));
        assert_eq!(residency.place(3, 12, 3), Placement::Full);
        assert_eq!(residency.state(3), PageState::Absent);
        // Failed loads wait before retrying.
        assert!(!residency.begin(3, 13));
        // Once unused long enough, the least recently used makes room.
        assert!(residency.begin(3, 200));
        assert_eq!(residency.place(3, 200, 3), Placement::Slot(1));
        assert_eq!(residency.resident_pages(), 3);
        assert_eq!(residency.state(1), PageState::Absent);
        assert_eq!(residency.state(0), PageState::Resident(0));
        // A budget of one keeps only the pinned page.
        residency.shrink(1);
        assert_eq!(residency.resident_pages(), 1);
        assert!(residency.group_resident(3));
        assert!(!residency.group_resident(2) && !residency.group_resident(0));
        residency.shrink(0);
        assert_eq!(residency.resident_pages(), 1);
    }
}
