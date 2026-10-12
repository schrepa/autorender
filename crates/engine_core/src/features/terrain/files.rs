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
    /// Moves the byte ranges hosts should fetch into `out`.
    pub fn take_requests(&mut self, out: &mut Vec<TileRequest>) {
        out.append(&mut self.requests);
    }

    /// Hands back a fetched tile's bytes, or `None` when the fetch failed:
    /// it is requested again later. Tiles no longer wanted are ignored.
    pub fn deliver(&mut self, terrain: Entity, key: TileKey, bytes: Option<Vec<u8>>) {
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
