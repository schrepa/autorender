//! Terrains read from files: the byte ranges hosts fetch, the tiles they
//! deliver, and decoding them, or tiles of a whole file at hand.

use super::*;

/// A byte range of a terrain file a host should fetch and hand back with
/// [`Terrains::deliver`].
#[derive(Clone, Debug, PartialEq)]
pub struct TileRequest {
    pub terrain: Entity,
    pub key: TileKey,
    pub url: Arc<str>,
    pub offset: u64,
    pub length: u32,
}

impl Terrains {
    /// Why the latest failed tile read did, if any failed.
    pub fn latest_read_failure(&self) -> Option<&str> {
        self.degraded.latest_read.as_deref()
    }

    /// Why the asset library refused the latest falls' white water, if it
    /// refused any.
    pub fn latest_fall_refusal(&self) -> Option<&str> {
        self.degraded.latest_fall.as_deref()
    }

    /// Counts falls drawn without their white water, which the asset
    /// library refused for `reason`.
    pub(crate) fn refuse_falls(&mut self, reason: String) {
        self.degraded.refused_falls += 1;
        self.degraded.latest_fall = Some(reason);
    }

    /// Moves the byte ranges hosts should fetch into `out`.
    pub fn take_requests(&mut self, out: &mut Vec<TileRequest>) {
        out.append(&mut self.requests);
    }

    /// Hands back a fetched tile's bytes, or why the fetch failed: the
    /// tile is requested again later, the failure counted
    /// ([`TerrainStats::failed_reads`]) and its reason kept
    /// ([`Self::latest_read_failure`]). Tiles no longer wanted are ignored.
    pub fn deliver(&mut self, terrain: Entity, key: TileKey, bytes: Result<Vec<u8>, String>) {
        let bytes = match bytes {
            Ok(bytes) => Some(bytes),
            Err(reason) => {
                self.degraded.failed_reads += 1;
                self.degraded.latest_read = Some(reason);
                None
            }
        };
        if let Some(state) = self
            .entries
            .iter_mut()
            .find(|state| state.entity == terrain && state.around.is_none())
            && state.in_flight.remove(&key)
        {
            state.arrivals.push((key, bytes));
        }
    }
}

/// Decodes `missing` tiles of `state`, a terrain whose whole file
/// (`header`, `data`) is here, within `work`'s time from slices of
/// `budget`, then gives each tile its surface.
pub(super) fn decode_file(
    state: &mut TerrainState,
    (header, data): (&FileHeader, &[u8]),
    missing: &[(TileKey, u32)],
    (work, budget): (&Work, &mut u64),
    frame: u64,
) {
    for &(key, _) in missing {
        if *budget == 0 && !work.refill(budget) {
            break;
        }
        if state.surface_job(key, budget) {
            continue;
        }
        let Some(entry) = header.tile(key) else {
            continue;
        };
        let range = usize::try_from(entry.offset)
            .ok()
            .and_then(|start| data.get(start..start.checked_add(entry.length as usize)?));
        *budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
        match range.map(|bytes| decode_tile(header, key, bytes)) {
            Some(Ok(tile)) => state.resurface(tile, budget),
            _ => {
                state.failed.insert(key, frame);
            }
        }
    }
}

/// Requests the byte ranges of `missing` tiles of `state`, a terrain
/// whose file (`header`) hosts fetch from `url`, unless under way or failed
/// a moment ago; decodes those delivered that `target` still holds within
/// `work`'s time from slices of `budget`, keeping the rest for later, and
/// gives each tile its surface.
pub(super) fn fetch_file(
    state: &mut TerrainState,
    (header, url): (&FileHeader, &Arc<str>),
    (missing, target): (&[(TileKey, u32)], &HashSet<TileKey>),
    (work, budget): (&Work, &mut u64),
    frame: u64,
    requests: &mut Vec<TileRequest>,
) {
    for &(key, _) in missing {
        if *budget > 0 || work.refill(budget) {
            state.surface_job(key, budget);
        }
        if state.in_flight.contains(&key)
            || state.jobs.contains_key(&key)
            || (state.store.contains(&key)
                && !matches!(state.store.phase(&key), Some(Phase::Shape | Phase::Placing)))
            || state
                .failed
                .get(&key)
                .is_some_and(|&at| frame < at + RETRY_AFTER)
        {
            continue;
        }
        let Some(entry) = header.tile(key) else {
            continue;
        };
        state.in_flight.insert(key);
        requests.push(TileRequest {
            terrain: state.entity,
            key,
            url: url.clone(),
            offset: entry.offset,
            length: entry.length,
        });
    }
    let arrivals = std::mem::take(&mut state.arrivals);
    let mut later = Vec::new();
    for (key, bytes) in arrivals {
        if *budget == 0 {
            work.refill(budget);
        }
        if *budget == 0 || !target.contains(&key) {
            if target.contains(&key) {
                later.push((key, bytes));
            }
            continue;
        }
        *budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
        match bytes.map(|bytes| decode_tile(header, key, &bytes)) {
            Some(Ok(tile)) => {
                state.failed.remove(&key);
                state.resurface(tile, budget);
            }
            _ => {
                state.failed.insert(key, frame);
            }
        }
    }
    state.arrivals = later;
}
