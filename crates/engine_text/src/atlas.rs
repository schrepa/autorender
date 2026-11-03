use cosmic_text::CacheKey;
use engine_core::collections::HashMap;

use crate::fonts::Fonts;

#[derive(Clone, Copy, Default)]
pub(crate) struct Slot {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) left: i32,
    pub(crate) top: i32,
}

pub(crate) struct Upload {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coverage: Vec<u8>,
}

#[derive(Default)]
pub(crate) struct Atlas {
    side: u32,
    slots: HashMap<CacheKey, Option<Slot>>,
    cursor: [u32; 2],
    shelf: u32,
    full: bool,
    uploads: Vec<Upload>,
}

impl Atlas {
    pub(crate) fn side(&self) -> u32 {
        self.side
    }

    pub(crate) fn resize(&mut self, side: u32) {
        if self.side != side {
            self.side = side;
            self.forget();
        }
    }

    pub(crate) fn forget(&mut self) {
        self.uploads.clear();
        self.empty();
        self.full = false;
    }

    pub(crate) fn take_uploads(&mut self) -> Vec<Upload> {
        std::mem::take(&mut self.uploads)
    }

    pub(crate) fn settle(&mut self) -> bool {
        let full = std::mem::take(&mut self.full);
        if full {
            self.empty();
        }
        full
    }

    pub(crate) fn slot(&mut self, key: CacheKey, fonts: &mut Fonts) -> Option<Slot> {
        if let Some(known) = self.slots.get(&key) {
            return *known;
        }
        if self.full {
            return None;
        }
        let slot = match fonts.raster(key) {
            Some(raster) if raster.width > 0 && raster.height > 0 => {
                if raster.width > self.side || raster.height > self.side {
                    None
                } else {
                    let Some([x, y]) = self.room(raster.width, raster.height) else {
                        self.full = true;
                        return None;
                    };
                    self.uploads.push(Upload {
                        x,
                        y,
                        width: raster.width,
                        height: raster.height,
                        coverage: raster.coverage,
                    });
                    Some(Slot {
                        x,
                        y,
                        width: raster.width,
                        height: raster.height,
                        left: raster.left,
                        top: raster.top,
                    })
                }
            }
            Some(_) | None => Some(Slot::default()),
        };
        self.slots.insert(key, slot);
        slot
    }

    fn empty(&mut self) {
        self.slots.clear();
        self.cursor = [0; 2];
        self.shelf = 0;
    }

    fn room(&mut self, width: u32, height: u32) -> Option<[u32; 2]> {
        if self.cursor[0] + width > self.side {
            self.cursor = [0, self.cursor[1] + self.shelf];
            self.shelf = 0;
        }
        if self.cursor[1] + height > self.side {
            return None;
        }
        let at = self.cursor;
        self.cursor[0] += width;
        self.shelf = self.shelf.max(height);
        Some(at)
    }
}
