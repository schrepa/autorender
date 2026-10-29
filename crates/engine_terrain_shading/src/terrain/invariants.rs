use super::*;

impl TerrainPass {
    #[expect(clippy::too_many_lines, reason = "one assertion pass over every table")]
    #[expect(
        clippy::iter_over_hash_type,
        reason = "assertions over every entry hold in any order"
    )]
    pub(super) fn check(&self, terrains: &RenderTerrains) {
        let atlas = &self.atlas;
        let tables = &self.tables;
        for key in atlas.slots.keys() {
            debug_assert!(atlas.held.contains_key(key), "t1_slot_holds_tile {key:?}");
        }
        debug_assert!(
            atlas.held.len() == atlas.slots.len(),
            "t1_held_matches_slots {} {}",
            atlas.held.len(),
            atlas.slots.len()
        );
        let timed = atlas.arrivals.keys().chain(atlas.surfaced.keys());
        for key in timed.chain(atlas.shaded.iter()) {
            debug_assert!(atlas.slots.contains_key(key), "t1_timed_has_slot {key:?}");
        }
        let mut owners = vec![0_u8; atlas.next_layer as usize];
        for (key, &layer) in &atlas.slots {
            debug_assert!(
                layer < atlas.next_layer,
                "t2_layer_handed_out {key:?} {layer} {}",
                atlas.next_layer
            );
            if let Some(owner) = owners.get_mut(layer as usize) {
                *owner += 1;
            }
        }
        for &layer in &atlas.free {
            debug_assert!(
                layer < atlas.next_layer,
                "t2_free_layer_handed_out {layer} {}",
                atlas.next_layer
            );
            if let Some(owner) = owners.get_mut(layer as usize) {
                *owner += 1;
            }
        }
        for (layer, &owner) in owners.iter().enumerate() {
            debug_assert!(owner == 1, "t2_layer_owned_once {layer} {owner}");
        }
        debug_assert!(
            atlas.next_layer <= atlas.layers(),
            "t2_layers_within_arrays {} {}",
            atlas.next_layer,
            atlas.layers()
        );
        let words = (tables.holes.size() / 4) as u32;
        let mut runs: Vec<u32> = tables.hole_offsets.values().copied().collect();
        runs.sort_unstable();
        for pair in runs.windows(2) {
            debug_assert!(
                pair[1] >= pair[0] + HOLE_WORDS as u32,
                "t3_hole_runs_apart {} {}",
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
                "t3_hole_run_has_tile {key:?}"
            );
            debug_assert!(
                offset >= tables.fade_words && offset + HOLE_WORDS as u32 <= words,
                "t3_hole_run_inside {key:?} {offset} {} {words}",
                tables.fade_words
            );
        }
        for (key, tile) in &atlas.held {
            debug_assert!(
                tile.hole_bits().is_none() || tables.hole_offsets.contains_key(key),
                "t3_holed_tile_has_run {key:?}"
            );
        }
        let records = self.patches.records();
        let mut ranges: Vec<Range<u32>> = Vec::new();
        for terrain in &self.patches.frame.terrains {
            ranges.extend(terrain.maps.iter().map(|(_, range)| range.clone()));
            debug_assert!(
                u64::from(terrain.uniform) + SLOT <= tables.uniforms.size(),
                "t4_uniform_inside {}",
                terrain.uniform
            );
        }
        ranges.retain(|range| !range.is_empty());
        ranges.sort_unstable_by_key(|range| range.start);
        for range in &ranges {
            debug_assert!(
                range.end as usize <= records.len(),
                "t4_range_inside {range:?} {}",
                records.len()
            );
        }
        for pair in ranges.windows(2) {
            debug_assert!(
                pair[0].end <= pair[1].start,
                "t4_ranges_apart {:?} {:?}",
                pair[0],
                pair[1]
            );
        }
        let resident: engine_core::collections::HashSet<u32> =
            atlas.slots.values().copied().collect();
        for record in records {
            debug_assert!(
                resident.contains(&record.atlas[0]),
                "t4_record_layer_held {}",
                record.atlas[0]
            );
            debug_assert!(
                record.holes == NO_HOLES || runs.binary_search(&record.holes).is_ok(),
                "t4_record_hole_run_known {}",
                record.holes
            );
        }
        let mut listed = 0;
        for terrain in &terrains.terrains {
            for tile in terrain.tiles.iter() {
                let key = (terrain.id, tile.key());
                listed += 1;
                let waits = atlas.unplaced.contains(&key);
                match atlas.held.get(&key) {
                    Some(held) => debug_assert!(
                        Arc::ptr_eq(held, tile) && !waits,
                        "t6_resident_listed_alone {key:?}"
                    ),
                    None => debug_assert!(waits, "t6_listed_held_or_waiting {key:?}"),
                }
            }
        }
        debug_assert!(
            atlas.held.len() + atlas.unplaced.len() == listed,
            "t6_tiles_match_listed {} {} {listed}",
            atlas.held.len(),
            atlas.unplaced.len()
        );
        let holed = atlas
            .held
            .values()
            .filter(|tile| tile.hole_bits().is_some())
            .count();
        debug_assert!(
            atlas.holed.len() == holed
                && atlas.holed.iter().all(|key| {
                    atlas
                        .held
                        .get(key)
                        .is_some_and(|tile| tile.hole_bits().is_some())
                }),
            "t6_holed_noted {} {holed}",
            atlas.holed.len()
        );
        if let Some((fades, _)) = atlas.faded_with {
            let mut words = vec![0_u32; atlas.fades.len()];
            for (key, &layer) in &atlas.slots {
                let (normals, materials) = atlas.fades_of(key, fades);
                if let Some(word) = words.get_mut(layer as usize / 2) {
                    *word |= atlas::fade_half(normals, materials) << ((layer % 2) * 16);
                }
            }
            debug_assert!(words == atlas.fades, "t7_fades_match_tiles");
        }
        debug_assert!(self.water.follows(&atlas.held), "t8_water_follows_resident");
        debug_assert!(
            tables.directory_bases.len() == terrains.terrains.len()
                && tables.palette_bases.len() == terrains.terrains.len(),
            "t5_bases_match_terrains {} {} {}",
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
                            "t5_directory_layer {word} {key:?} {} {layer}",
                            terrain.id
                        );
                    }
                }
            }
        }
    }
}
