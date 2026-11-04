use super::*;

fields! {
    unauthored;
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Surroundings {
        pub reach: f32 = 4096.0 => above_to(0.0, Surroundings::MAX_REACH),
        pub blend: f32 = 1024.0 => range(0.0, Surroundings::MAX_REACH),
    }
}

impl Surroundings {
    pub const MAX_REACH: f32 = 1.0e6;

    pub fn layout(&self, terrain: &Terrain) -> Option<(Vec3, Pyramid, (Vec2, Vec2))> {
        let base = terrain.pyramid;
        let extent = base.extent();
        let spacing = base.spacing(base.levels().saturating_sub(1).min(2));
        let tile = TILE_QUADS as f32 * spacing;
        let wanted = engine_math::max(extent.x, extent.y) + 2.0 * self.reach;
        let mut tiles = ((wanted / tile).ceil() as u32)
            .clamp(1, Pyramid::MAX_TILES)
            .next_power_of_two()
            .min(Pyramid::MAX_TILES);
        let whole = |length: f32, unit: f32| {
            let ratio = length / unit;
            ratio >= 1.0 && ratio == ratio.floor()
        };
        while tiles >= 1 {
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SurroundingsStatus {
    pub reach: [f32; 2],
    pub tiles: u32,
    pub spacing: f32,
    pub resident_tiles: u32,
    pub drawn: bool,
}

impl Terrains {
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
            resident_tiles: state.store.len() as u32,
            drawn: state.drawn,
        })
    }

    pub(super) fn sync_surroundings(&mut self, kinds: &[&'static LayerKind]) {
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
            let base = state
                .baked
                .clone()
                .unwrap_or_else(|| state.terrain.layers.clone());
            if base.unbaked().is_some() {
                continue;
            }
            if let Some(layout) = surroundings.layout(&state.terrain) {
                let surface = state.terrain.surface.clone();
                wanted.push((state.entity, base, surroundings, layout, surface));
            }
        }
        let mut kept = Vec::with_capacity(self.entries.len());
        for state in std::mem::take(&mut self.entries) {
            let held = state.around.as_ref().is_none_or(|around| {
                wanted
                    .iter()
                    .any(|(entity, base, surroundings, (_, _, cutout), _)| {
                        *entity == state.entity
                            && Arc::ptr_eq(base, &around.base)
                            && *surroundings == around.surroundings
                            && *cutout == around.cutout
                    })
            });
            if held {
                kept.push(state);
            } else {
                state.retire(&mut self.retired);
                self.stale = Stale::ALL;
            }
        }
        self.entries = kept;
        for (entity, base, surroundings, (origin, pyramid, cutout), surface) in wanted {
            if self
                .entries
                .iter()
                .any(|state| state.entity == entity && state.around.is_some())
            {
                continue;
            }
            let layers = base.surroundings(
                -cutout.0,
                (Vec2::ZERO, cutout.1 - cutout.0),
                surroundings.blend,
            );
            let terrain = Terrain {
                origin,
                pyramid,
                layers: Arc::new(layers),
                surface: TerrainSurface {
                    rules: Arc::new(surface.rules.without_placements()),
                    materials: surface.materials,
                    assets: Vec::new(),
                    bridge: None,
                },
                collision: false,
                edits: Arc::from(Vec::new()),
                rivers: Arc::from(Vec::new()),
                surroundings: None,
            };
            let mut state = TerrainState::new(entity, terrain, kinds);
            state.around = Some(Around {
                base,
                surroundings,
                cutout,
            });
            self.entries.push(state);
        }
    }
}
