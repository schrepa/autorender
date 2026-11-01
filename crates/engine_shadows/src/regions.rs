use bytemuck::Zeroable;
use engine_core::glam::{Mat4, Vec3, Vec4};
use engine_places::shadows::{CasterMap, KeptRefresh, Refresh, Region as MapView, SHADOW_REGIONS};
use engine_render::CameraUniform;

use crate::atlas::AtlasLayout;
use crate::constants::{MAX_SHADOW_LEVELS, SKY_REFRESH};
use crate::gpu_layouts::{CasterCull, ComposeData, Routes, ShadowPiece};
use crate::shadow_fit::{ClipPlan, Clipmap, LightBasis, MapRegion, Region, region_tiles};
use crate::sky::SkyDraw;

const NORMAL_OFFSET_TEXELS: f32 = 0.5;
const SLOPE_LIMIT_TEXELS: f32 = 16.0;
const LEVELS: usize = MAX_SHADOW_LEVELS as usize;
const REGIONS: usize = SHADOW_REGIONS as usize;
const PASS_ALL: [[f32; 4]; 6] = [[0.0, 0.0, 0.0, 1.0]; 6];

pub(crate) const MOVING_VIEW: u32 = SHADOW_REGIONS;
pub(crate) const SKY_VIEW: u32 = SHADOW_REGIONS + 1;

pub(crate) struct FrameRegions<'a> {
    pub(crate) basis: LightBasis,
    pub(crate) clipmap: &'a Clipmap,
    pub(crate) plan: &'a ClipPlan,
    pub(crate) layout: AtlasLayout,
}

fn planes(rows: [Vec3; 3], rect: [f32; 4], depth: f32) -> [[f32; 4]; 6] {
    let [x0, y0, x1, y1] = rect;
    let [right, up, away] = rows;
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

fn view(clip: Mat4, wind: [f32; 4], eye: [f32; 4]) -> CameraUniform {
    CameraUniform {
        view_projection: clip.to_cols_array_2d(),
        position: eye,
        wind,
        ..CameraUniform::default()
    }
}

impl FrameRegions<'_> {
    fn region_clip(&self, region: &Region) -> Mat4 {
        MapRegion {
            rect: region.rect,
            depth: region.depth,
        }
        .clip(&self.basis)
    }

    fn finest_moving(&self) -> f32 {
        self.plan
            .moving
            .iter()
            .map(|tile| self.clipmap.texel(tile.level))
            .reduce(engine_core::math::min)
            .unwrap_or(0.0)
    }

    pub(crate) fn views(&self, sky: Option<&SkyDraw>, out: &mut Vec<MapView>) {
        out.clear();
        for region in &self.plan.regions {
            out.push(MapView {
                view: self.region_clip(region),
                map: CasterMap::Region,
                refresh: Refresh::Kept(KeptRefresh {
                    planes: planes(self.basis.rows, region.rect, region.depth[1]),
                    texel: region.texel,
                }),
            });
        }
        if !self.plan.moving.is_empty() {
            out.push(MapView {
                view: Mat4::IDENTITY,
                map: CasterMap::Moving,
                refresh: Refresh::Kept(KeptRefresh {
                    planes: PASS_ALL,
                    texel: self.finest_moving(),
                }),
            });
        }
        if let Some(draw) = sky {
            let rows = draw.rows.map(|row| Vec4::from_array(row).truncate());
            out.push(MapView {
                view: draw.clip,
                map: CasterMap::Sky,
                refresh: Refresh::Kept(KeptRefresh {
                    planes: planes(rows, draw.rect, draw.depth),
                    texel: 0.0,
                }),
            });
        }
    }

    pub(crate) fn cameras(
        &self,
        (wind, gusting): ([f32; 2], [f32; 4]),
        (viewer, sky): (Vec4, Option<(Mat4, [f32; 4])>),
        out: &mut Vec<CameraUniform>,
    ) {
        out.clear();
        out.resize(SKY_VIEW as usize + 1, CameraUniform::default());
        for (index, region) in self.plan.regions.iter().enumerate() {
            out[index] = view(self.region_clip(region), [wind[0], wind[1], 0.0, 0.0], [0.0; 4]);
        }
        out[MOVING_VIEW as usize] = view(Mat4::IDENTITY, gusting, viewer.to_array());
        if let Some((clip, written)) = sky {
            out[SKY_VIEW as usize] = view(clip, written, [0.0; 4]);
        }
    }

    pub(crate) fn routes(&self, routes: &mut Routes) {
        *routes = Routes::zeroed();
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
    }

    pub(crate) fn composes(&self, tile_texels: u32, out: &mut Vec<ComposeData>) {
        out.clear();
        let square = region_tiles(tile_texels);
        for compose in &self.plan.composes {
            let place = self.layout.texels(u32::from(compose.slot));
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

    pub(crate) fn cull(&self, sky: Option<&SkyDraw>) -> CasterCull {
        let row = |axis: Vec3| axis.extend(0.0).to_array();
        let mut regions = [[0.0; 4]; REGIONS];
        let mut region_depths = [[0.0; 4]; REGIONS];
        for (index, region) in self.plan.regions.iter().enumerate() {
            regions[index] = region.rect;
            region_depths[index] = [region.depth[1], 0.0, 0.0, 0.0];
        }
        let mut windows = [[0.0; 4]; LEVELS];
        for (level, window) in windows.iter_mut().enumerate().take(self.clipmap.len()) {
            let [x, y, tile, _] = self.clipmap.window(level);
            *window = [x, y, tile, 0.0];
        }
        let mut cull = CasterCull {
            light: self.basis.rows.map(row),
            regions,
            region_depths,
            windows,
            counts: [
                self.plan.regions.len() as u32,
                self.clipmap.len() as u32,
                0,
                self.clipmap.side(),
            ],
            moving: [self.layout.still, self.plan.moving.len() as u32, 0, 0],
            mode: [0; 4],
            sky: [[0.0; 4]; 3],
            sky_rect: [0.0; 4],
            sky_depths: [0.0; 4],
        };
        if let Some(draw) = sky {
            cull.sky = draw.rows;
            cull.sky_rect = draw.rect;
            cull.sky_depths = [draw.depth, 0.0, 0.0, 0.0];
            cull.mode[0] |= SKY_REFRESH;
        }
        cull
    }

    pub(crate) fn lookup_piece(
        &self,
        (tile_texels, crown_depth): (u32, f32),
        shown: u32,
    ) -> ShadowPiece {
        let (width, height) = self.layout.size();
        let levels = self.clipmap.len();
        let mut windows = [[0.0; 4]; LEVELS];
        let mut bands = [[0.0; 4]; LEVELS];
        for level in 0..levels.min(LEVELS) {
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
        ShadowPiece {
            light: self.basis.rows.map(|row| row.extend(0.0).to_array()),
            state: [1.0, crown_depth, shown as f32, 0.0],
            level_state: [
                levels as f32,
                self.clipmap.texel(0),
                tile_texels as f32,
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
