//! What the terrain pass holds true after every preparation, checked in
//! builds with debug assertions:
//!
//! - T1 slots: every resident tile's slot holds that tile, and only
//!   resident tiles have arrival or surface times;
//! - T2 layers: each layer holds at most one tile, every layer handed out
//!   is resident or free, never both, and none lies past the arrays;
//! - T3 holes: each tile with hole bits has its own run of them past the
//!   layers' fades and within the buffer, and only such tiles do;
//! - T4 ranges: each frame terrain's ranges lie within the records laid
//!   out, without overlapping, with no more patches with holes than
//!   patches, and every record names resident layers and a hole run or
//!   none;
//! - T5 directory: every terrain's directory names the layers its
//!   resident tiles hold and nothing else.

use super::*;

impl TerrainPass {
    /// Asserts T1 to T5 over `terrains`, just prepared.
    #[expect(clippy::too_many_lines, reason = "one assertion pass over every table")]
    #[expect(
        clippy::iter_over_hash_type,
        reason = "assertions over every entry hold in any order"
    )]
    pub(super) fn check(&self, terrains: &RenderTerrains) {
        let atlas = &self.atlas;
        let tables = &self.tables;
        // T1 slots.
        for key in atlas.slots.keys() {
            debug_assert!(
                atlas.held.contains_key(key),
                "T1: slot {key:?} holds no tile"
            );
        }
        debug_assert!(
            atlas.held.len() == atlas.slots.len(),
            "T1: {} tiles held for {} slots",
            atlas.held.len(),
            atlas.slots.len()
        );
        for key in atlas.arrivals.keys().chain(atlas.surfaced.keys()) {
            debug_assert!(
                atlas.slots.contains_key(key),
                "T1: {key:?} has a time but no slot"
            );
        }
        // T2 layers.
        let mut owners = vec![0_u8; atlas.next_layer as usize];
        for (key, &layer) in &atlas.slots {
            debug_assert!(
                layer < atlas.next_layer,
                "T2: {key:?} holds layer {layer}, never handed out ({})",
                atlas.next_layer
            );
            if let Some(owner) = owners.get_mut(layer as usize) {
                *owner += 1;
            }
        }
        for &layer in &atlas.free {
            debug_assert!(
                layer < atlas.next_layer,
                "T2: free layer {layer} never handed out ({})",
                atlas.next_layer
            );
            if let Some(owner) = owners.get_mut(layer as usize) {
                *owner += 1;
            }
        }
        for (layer, &owner) in owners.iter().enumerate() {
            debug_assert!(
                owner == 1,
                "T2: layer {layer} is resident or free {owner} times"
            );
        }
        debug_assert!(
            atlas.next_layer <= atlas.layers(),
            "T2: {} layers handed out of the arrays' {}",
            atlas.next_layer,
            atlas.layers()
        );
        // T3 holes.
        let words = (tables.holes.size() / 4) as u32;
        let mut runs: Vec<u32> = tables.hole_offsets.values().copied().collect();
        runs.sort_unstable();
        for pair in runs.windows(2) {
            debug_assert!(
                pair[1] >= pair[0] + HOLE_WORDS as u32,
                "T3: hole runs at {} and {} overlap",
                pair[0],
                pair[1]
            );
        }
        for (key, &offset) in &tables.hole_offsets {
            debug_assert!(
                atlas
                    .held
                    .get(key)
                    .is_some_and(|tile| tile.hole_bits().is_some()),
                "T3: {key:?} has a hole run but no resident tile with holes"
            );
            debug_assert!(
                offset >= tables.fade_words && offset + HOLE_WORDS as u32 <= words,
                "T3: {key:?}'s hole run at {offset} lies outside {}..{words}",
                tables.fade_words
            );
        }
        for (key, tile) in &atlas.held {
            debug_assert!(
                tile.hole_bits().is_none() || tables.hole_offsets.contains_key(key),
                "T3: resident {key:?} has holes but no run"
            );
        }
        // T4 ranges.
        let records = self.patches.records();
        let mut ranges: Vec<Range<u32>> = Vec::new();
        for terrain in &self.patches.frame.terrains {
            debug_assert!(
                terrain.camera_holes as usize <= terrain.camera.len()
                    && terrain.all_holes as usize <= terrain.all.len(),
                "T4: more patches with holes than patches"
            );
            ranges.push(terrain.camera.clone());
            ranges.push(terrain.all.clone());
            ranges.extend(terrain.maps.iter().map(|(_, range)| range.clone()));
            debug_assert!(
                u64::from(terrain.uniform) + SLOT <= tables.uniforms.size(),
                "T4: a uniform at {} past the buffer",
                terrain.uniform
            );
        }
        ranges.retain(|range| !range.is_empty());
        ranges.sort_unstable_by_key(|range| range.start);
        for range in &ranges {
            debug_assert!(
                range.end as usize <= records.len(),
                "T4: range {range:?} past the {} records",
                records.len()
            );
        }
        for pair in ranges.windows(2) {
            debug_assert!(
                pair[0].end <= pair[1].start,
                "T4: ranges {:?} and {:?} overlap",
                pair[0],
                pair[1]
            );
        }
        let resident: engine_core::collections::HashSet<u32> =
            atlas.slots.values().copied().collect();
        for record in records {
            debug_assert!(
                resident.contains(&record.atlas[0]) && resident.contains(&record.atlas[1]),
                "T4: a record names layers {:?} no tile holds",
                &record.atlas[..2]
            );
            debug_assert!(
                record.holes == NO_HOLES || runs.binary_search(&record.holes).is_ok(),
                "T4: a record names hole run {} that is not one",
                record.holes
            );
        }
        // T5 directory.
        debug_assert!(
            tables.directory_bases.len() == terrains.terrains.len()
                && tables.palette_bases.len() == terrains.terrains.len(),
            "T5: {} directory and {} palette bases for {} terrains",
            tables.directory_bases.len(),
            tables.palette_bases.len(),
            terrains.terrains.len()
        );
        if tables.directory_written.is_empty() {
            return;
        }
        for (terrain, &base) in terrains.terrains.iter().zip(&tables.directory_bases) {
            for level in 0..terrain.pyramid.levels() {
                let header = base as usize + 4 * level as usize;
                let (first, [tiles_x, tiles_z]) = (
                    tables.directory_written[header] as usize,
                    terrain.pyramid.tiles(level),
                );
                for z in 0..tiles_z {
                    for x in 0..tiles_x {
                        let key = TileKey::new(level as u8, x as u16, z as u16);
                        let word = tables.directory_written[first + (z * tiles_x + x) as usize];
                        let layer = atlas
                            .slots
                            .get(&(terrain.id, key))
                            .copied()
                            .unwrap_or(NOT_RESIDENT);
                        debug_assert!(
                            word == layer,
                            "T5: the directory names layer {word} for {key:?} of terrain {}, which holds {layer}",
                            terrain.id
                        );
                    }
                }
            }
        }
    }
}
