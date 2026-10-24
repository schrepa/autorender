//! What a frame of the clipmap draws and reads besides its tiles: each
//! region's view, which its casters are drawn and culled through, the views
//! of the atlas's pass, the moving tiles' routes onto their slots, the
//! tiles composed from the regions into their slots, the caster
//! selection's uniform and the lookups'.

use engine_core::glam::{Mat4, Vec3, Vec4};

use super::maps::AtlasLayout;
use super::{
    COMPOSE_VIEW, KeptRefresh, MAPS, MOVING_MAP, MOVING_VIEW, SKY_MAP, SKY_VIEW, VIEW_STRIDE,
    VIRTUAL_SHADOW_PLANES, VirtualShadowView,
};
use crate::gpu_data::{ComposeData, MAX_LEVELS, RouteData, ShadowCullUniform, ShadowUniform};
use crate::gpu_layouts::constants::{SHADOW_REGIONS, STILLNESS};
use crate::shadow_fit::{ClipPlan, Clipmap, LightBasis, MapRegion, Region, region_tiles};

/// Lookups move this many texels along the surface normal. Receiver
/// planes keep tilted surfaces off their own depths; the offset only covers
/// where a lookup's texels meet another surface.
const NORMAL_OFFSET_TEXELS: f32 = 0.5;
/// Receiver planes steeper than this, in texels of depth per texel (86
/// degrees from the sun), are clamped.
const SLOPE_LIMIT_TEXELS: f32 = 16.0;

/// A frame's clipmap as its views and uniforms read it: the light basis,
/// the clipmap placed and its plan for the frame, and the atlas's layout.
pub(super) struct FrameRegions<'a> {
    pub(super) basis: LightBasis,
    pub(super) clipmap: &'a Clipmap,
    pub(super) plan: &'a ClipPlan,
    pub(super) layout: AtlasLayout,
}

/// The casters a frame selects: candidates, skinned slots among them,
/// caster groups, the ID capacity of each region's map, and whether this
/// frame keeps still casters (`classify`).
pub(super) struct SelectionCounts {
    pub(super) candidates: u32,
    pub(super) skinned: u32,
    pub(super) groups: u32,
    pub(super) capacity: u32,
    pub(super) classify: bool,
}

impl FrameRegions<'_> {
    /// World to clip space of `region`'s layer: its rectangle over the
    /// layer, its band over depths 0 to 1.
    fn region_clip(&self, region: &Region) -> Mat4 {
        MapRegion {
            rect: region.rect,
            depth: region.depth,
        }
        .clip(&self.basis)
    }

    /// Inward world planes around the light-space rectangle `rect` and up
    /// to the band's farthest depth `depth`, beyond which casters darken
    /// none of its receivers, then one passing everything: what casts
    /// into it.
    fn rect_planes(&self, rect: [f32; 4], depth: f32) -> [[f32; 4]; 6] {
        let [x0, y0, x1, y1] = rect;
        let [right, up, away] = self.basis.rows;
        let plane = |normal: Vec3, distance: f32| normal.extend(distance).to_array();
        [
            plane(right, -x0),
            plane(-right, x1),
            plane(up, -y0),
            plane(-up, y1),
            plane(-away, depth),
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Each region this frame redraws as terrain and placed objects cull
    /// their casters for it: the map whose lists it reads (its index), the
    /// inward world planes around it and its texels' world size.
    pub(super) fn region_views(&self, out: &mut Vec<(usize, KeptRefresh)>) {
        out.clear();
        for (index, region) in self.plan.regions.iter().enumerate() {
            out.push((
                index,
                KeptRefresh {
                    planes: self.rect_planes(region.rect, region.depth[1]),
                    texel: region.texel,
                },
            ));
        }
    }

    /// Each region this frame redraws as virtual geometry selects its
    /// casters' clusters for it: within half its texels whatever the camera
    /// sees, as the regions draw every other caster.
    pub(super) fn virtual_views(&self, out: &mut Vec<VirtualShadowView>) {
        for (index, region) in self.plan.regions.iter().enumerate() {
            let mut planes = [[0.0, 0.0, 0.0, 1.0]; VIRTUAL_SHADOW_PLANES];
            for (slot, plane) in planes
                .iter_mut()
                .zip(self.rect_planes(region.rect, region.depth[1]))
            {
                *slot = plane;
            }
            out.push(VirtualShadowView {
                map: index,
                planes,
                direction: self.basis.rows[2],
                view_offset: (index as u64 * VIEW_STRIDE) as u32,
                error: 0.5 * region.texel,
            });
        }
    }

    /// The world error each map's casters may draw with: half a texel of
    /// a region's level, so rasterized depths differ from the mesh's by
    /// less than rounding; half a texel of the finest moving tile in the
    /// moving map; and the sky map draws meshes whole, its texels differing
    /// near and far from frame to frame.
    pub(super) fn errors(&self) -> [f32; MAPS] {
        let finest_moving = self
            .plan
            .moving
            .iter()
            .map(|tile| self.clipmap.texel(tile.level))
            .reduce(engine_core::math::min)
            .unwrap_or(0.0);
        std::array::from_fn(|map| match map {
            MOVING_MAP => 0.5 * finest_moving,
            SKY_MAP => 0.0,
            region => self
                .plan
                .regions
                .get(region)
                .map_or(0.0, |region| 0.5 * region.texel),
        })
    }

    /// The views' matrices and what follows each at its view offset, up to
    /// the sky map's, which a frame redrawing it writes: each region's,
    /// holding placed objects at the mean bend of `wind` (its velocity)
    /// without gusts at time zero; the moving tiles' view, whose matrix
    /// their routes stand in for, holding the wind `gusting` (its velocity,
    /// gust strength and time) and `viewer`, the detail eye whose distance
    /// sizes the wind's shaking on screen and (w) the pixels a radian spans
    /// there; and the compose view, whose first column holds two over the
    /// atlas's width and height.
    pub(super) fn view_matrices(
        &self,
        (wind, gusting): ([f32; 2], [f32; 4]),
        viewer: Vec4,
    ) -> Vec<f32> {
        let stride = VIEW_STRIDE as usize / 4;
        let mut views = vec![0.0_f32; (SKY_VIEW / 4) as usize];
        for (index, region) in self.plan.regions.iter().enumerate() {
            let start = index * stride;
            views[start..start + 16].copy_from_slice(&self.region_clip(region).to_cols_array());
            views[start + 16..start + 20].copy_from_slice(&[wind[0], wind[1], 0.0, 0.0]);
        }
        let moving = (MOVING_VIEW / 4) as usize;
        views[moving..moving + 16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
        views[moving + 16..moving + 20].copy_from_slice(&gusting);
        views[moving + 20..moving + 24].copy_from_slice(&viewer.to_array());
        let (width, height) = self.layout.size();
        let compose = (COMPOSE_VIEW / 4) as usize;
        views[compose] = 2.0 / width as f32;
        views[compose + 1] = 2.0 / height as f32;
        views
    }

    /// The moving tiles' routes: each tile's world to the atlas's clip
    /// space, its rectangle (with its apron) mapped onto its slot, and the
    /// slot's texels.
    pub(super) fn routes(&self) -> RouteData {
        let mut routes: RouteData = bytemuck::Zeroable::zeroed();
        let (width, height) = self.layout.size();
        let (width, height) = (width as f32, height as f32);
        let side = self.layout.slot as f32;
        for tile in &self.plan.moving {
            let index = usize::from(tile.index);
            let slot = self
                .layout
                .texels(self.layout.still + u32::from(tile.index));
            let clip = MapRegion {
                rect: self.clipmap.tile_rect(tile.level, tile.tile),
                depth: self.clipmap.band(tile.level),
            }
            .clip(&self.basis);
            // The tile's clip x and y, over [-1, 1], onto the slot's part of
            // the atlas's; texel rows run downward.
            let onto = Mat4::from_cols(
                Vec4::new(side / width, 0.0, 0.0, 0.0),
                Vec4::new(0.0, side / height, 0.0, 0.0),
                Vec4::Z,
                Vec4::new(
                    (2.0 * slot[0] + side) / width - 1.0,
                    1.0 - (2.0 * slot[1] + side) / height,
                    0.0,
                    1.0,
                ),
            );
            routes.clips[index] = (onto * clip).to_cols_array_2d();
            routes.slots[index] = slot;
        }
        routes.atlas = [2.0 / width, 2.0 / height, 0.0, 0.0];
        routes
    }

    /// The tiles composed from the regions into their slots this frame, an
    /// instance each, for tiles `tile_texels` a side.
    pub(super) fn composes(&self, tile_texels: u32, out: &mut Vec<ComposeData>) {
        out.clear();
        let square = region_tiles(tile_texels);
        for compose in &self.plan.composes {
            let place = self.layout.texels(u32::from(compose.slot));
            // The tile's first texel in its region's layer, rows running
            // down from the square's last row.
            let first = [
                compose.at[0] * tile_texels,
                (square - 1 - compose.at[1]) * tile_texels,
            ];
            out.push(ComposeData {
                place,
                source: [
                    i32::from(compose.region),
                    first[0] as i32 - place[0] as i32,
                    first[1] as i32 - place[1] as i32,
                    0,
                ],
            });
        }
    }

    /// The selection pass's uniform for `counts`' casters: the regions
    /// this frame redraws, every level's window, and the counters' layout
    /// for `mask_words` words of a mask of every level's tiles.
    pub(super) fn selection(&self, counts: &SelectionCounts, mask_words: u32) -> ShadowCullUniform {
        let row = |axis: Vec3| axis.extend(0.0).to_array();
        let mut regions = [[0.0; 4]; SHADOW_REGIONS];
        let mut region_depths = [[0.0; 4]; SHADOW_REGIONS];
        for (index, region) in self.plan.regions.iter().enumerate() {
            regions[index] = region.rect;
            region_depths[index] = [region.depth[1], 0.0, 0.0, 0.0];
        }
        let mut windows = [[0.0; 4]; MAX_LEVELS];
        for (level, window) in windows.iter_mut().enumerate().take(self.clipmap.len()) {
            let [x, y, tile, _] = self.clipmap.window(level);
            *window = [x, y, tile, 0.0];
        }
        ShadowCullUniform {
            light: self.basis.rows.map(row),
            regions,
            region_depths,
            windows,
            candidates: counts.candidates,
            groups: counts.groups,
            capacity: counts.capacity,
            mode: if counts.classify { STILLNESS } else { 0 },
            counts: [
                self.plan.regions.len() as u32,
                self.clipmap.len() as u32,
                counts.skinned,
                self.clipmap.side(),
            ],
            words: [mask_words, 0, mask_words, 2 * mask_words],
            moving: [self.layout.still, self.plan.moving.len() as u32, 0, 0],
            // The sky map's, set where a frame redraws one.
            sky: [[0.0; 4]; 3],
            sky_rect: [0.0; 4],
            sky_depths: [0.0; 4],
        }
    }

    /// The main passes' lookup uniform, the meters into crowns the sun
    /// reaches being `crown_depth`; the horizon records' count follows
    /// (`ShadowMaps::write_horizon_records`).
    pub(super) fn lookup_uniform(&self, tile_texels: u32, crown_depth: f32) -> ShadowUniform {
        let (width, height) = self.layout.size();
        let levels = self.clipmap.len();
        let mut windows = [[0.0; 4]; MAX_LEVELS];
        let mut bands = [[0.0; 4]; MAX_LEVELS];
        for level in 0..levels.min(MAX_LEVELS) {
            windows[level] = self.clipmap.window(level);
            let texel = self.clipmap.texel(level);
            let [near, far] = self.clipmap.band(level);
            let depth = engine_core::math::max(far - near, 1.0e-6);
            bands[level] = [
                near,
                1.0 / depth,
                texel * NORMAL_OFFSET_TEXELS,
                SLOPE_LIMIT_TEXELS * texel / depth,
            ];
        }
        ShadowUniform {
            light: self.basis.rows.map(|row| row.extend(0.0).to_array()),
            state: [1.0, crown_depth, 0.0, 0.0],
            level_state: [
                levels as f32,
                self.clipmap.texel(0),
                tile_texels as f32,
                // The horizon records, which the shadows set.
                0.0,
            ],
            atlas: [
                self.clipmap.side() as f32,
                self.layout.per_row as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            windows,
            bands,
        }
    }
}
