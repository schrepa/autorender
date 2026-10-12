//! Terrains read from files: the byte ranges hosts fetch, and the tiles
//! they deliver.

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
