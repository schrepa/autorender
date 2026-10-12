//! Surroundings: the far ground a terrain draws beyond its own tiles, and
//! how far it has been built.

use super::*;

settings! {
    unauthored;
    /// Ground generated past a terrain's edges to the horizon:
    /// a coarser terrain around it from the same layers, its samples four
    /// times as far apart as the terrain's finest, reaching about `reach`
    /// meters past every edge (the terrain's inspection reports how far: its
    /// tiles align with the terrain's sample grids). Its stamps, heightmaps,
    /// bakes and roads fade into its noise over `blend` meters past the edges,
    /// so the two meet where the terrain ends; it never collides, places
    /// objects, grows blades or holds water. Terrains read from files have
    /// none.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Surroundings {
        /// Meters the ground reaches past every edge.
        pub reach: f32 = 4096.0 => above_to(0.0, Surroundings::MAX_REACH),
        /// Meters past the edges over which the terrain's own features fade
        /// into its noise.
        pub blend: f32 = 1024.0 => range(0.0, Surroundings::MAX_REACH),
    }
}

impl Surroundings {
    /// The most meters surroundings reach past a terrain's edges or blend
    /// over.
    pub const MAX_REACH: f32 = 1.0e6;

    /// Where `terrain`'s surroundings lie: their origin, their pyramid and
    /// the rectangle of their ground (local) the terrain covers. Their
    /// tiles along a side are a power of two, as few as reach `reach` past
    /// the terrain's widest side while the terrain's edges stay on the
    /// edges of every level's patches, so the ground the terrain covers is
    /// left whole patches and both draw the same samples where they meet;
    /// `None` where no count fits.
    pub fn layout(&self, terrain: &Terrain) -> Option<(Vec3, Pyramid, (Vec2, Vec2))> {
        let base = terrain.pyramid;
        let extent = base.extent();
        let spacing = base.spacing(base.levels().saturating_sub(1).min(2));
        let tile = TILE_QUADS as f32 * spacing;
        let wanted = crate::math::max(extent.x, extent.y) + 2.0 * self.reach;
        let mut tiles = ((wanted / tile).ceil() as u32)
            .clamp(1, Pyramid::MAX_TILES)
            .next_power_of_two()
            .min(Pyramid::MAX_TILES);
        let whole = |length: f32, unit: f32| {
            let ratio = length / unit;
            ratio >= 1.0 && ratio == ratio.floor()
        };
        while tiles >= 1 {
            // The coarsest level's samples lie `tiles` times as far apart
            // as the finest's, its patches sixteen of them across.
            let patch = PATCH_QUADS as f32 * spacing * tiles as f32;
            let offset = (Vec2::splat(tiles as f32 * tile) - extent) * 0.5;
            if offset.x > 0.0
                && offset.y > 0.0
                && whole(offset.x, patch)
                && whole(offset.y, patch)
                && whole(extent.x, patch)
                && whole(extent.y, patch)
            {
                let pyramid = Pyramid::new([tiles, tiles], spacing).ok()?;
                let origin = terrain.origin - Vec3::new(offset.x, 0.0, offset.y);
                return Some((origin, pyramid, (offset, offset + extent)));
            }
            tiles /= 2;
        }
        None
    }
}

/// A terrain's surroundings as they stream: how far they
/// reach past its edges along x and z, their level-0 tiles along a side and
/// sample spacing, and their resident tiles and the patches drawn.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SurroundingsStatus {
    pub reach: [f32; 2],
    pub tiles: u32,
    pub spacing: f32,
    pub resident_tiles: u32,
    pub patches: u32,
}

impl Terrains {
    /// `entity`'s surroundings while they stream: they start
    /// once its bakes are done.
    pub fn surroundings(&self, entity: Entity) -> Option<SurroundingsStatus> {
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_some())?;
        let around = state.around.as_ref()?;
        Some(SurroundingsStatus {
            reach: around.cutout.0.to_array(),
            tiles: state.terrain.pyramid.base_tiles()[0],
            spacing: state.terrain.pyramid.spacing(0),
            resident_tiles: state.tiles.len() as u32,
            patches: state.selection.patches.len() as u32,
        })
    }

    /// Follows each terrain's surroundings: drawn, once its
    /// bakes are done, from the layers its tiles generate from, and dropped
    /// with it or when those or its settings for them change. They stream
    /// after every terrain, which take the frame's work first.
    pub(super) fn sync_surroundings(&mut self) {
        let mut wanted = Vec::new();
        for state in &mut self.entries {
            if state.around.is_some() {
                continue;
            }
            let Some(surroundings) = state
                .terrain
                .surroundings
                .filter(|surroundings| surroundings.validate().is_ok())
            else {
                continue;
            };
            let TerrainSource::Generated(authored) = &state.terrain.source else {
                continue;
            };
            let layers = state.baked.clone().unwrap_or_else(|| authored.clone());
            if layers.unbaked().is_some() {
                continue;
            }
            let base = state.generation(layers);
            if let Some(layout) = surroundings.layout(&state.terrain) {
                let surface = state.terrain.surface.clone();
                wanted.push((state.entity, base, surroundings, layout, surface));
            }
        }
        let retired = &mut self.retired;
        self.entries.retain(|state| {
            let Some(around) = &state.around else {
                return true;
            };
            let kept = wanted
                .iter()
                .any(|(entity, base, surroundings, (_, _, cutout), _)| {
                    *entity == state.entity
                        && Arc::ptr_eq(base, &around.base)
                        && *surroundings == around.surroundings
                        && *cutout == around.cutout
                });
            if !kept {
                state.retire(retired);
            }
            kept
        });
        for (entity, base, surroundings, (origin, pyramid, cutout), surface) in wanted {
            if self
                .entries
                .iter()
                .any(|state| state.entity == entity && state.around.is_some())
            {
                continue;
            }
            // Their ground reads the terrain's layers where it lies in the
            // terrain's, whose own ground spans the cutout.
            let layers = base.surroundings(
                -cutout.0,
                (Vec2::ZERO, cutout.1 - cutout.0),
                surroundings.blend,
            );
            let terrain = Terrain {
                origin,
                pyramid,
                source: TerrainSource::Generated(Arc::new(layers)),
                surface: TerrainSurface {
                    rules: Arc::new(surface.rules.without_placements()),
                    materials: surface.materials,
                    assets: Vec::new(),
                    bridge: None,
                },
                collision: false,
                edits: Arc::from(Vec::new()),
                surroundings: None,
            };
            let mut state = TerrainState::new(entity, terrain);
            state.around = Some(Around {
                base,
                surroundings,
                cutout,
            });
            self.entries.push(state);
        }
    }
}
