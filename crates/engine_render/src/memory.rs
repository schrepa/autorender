use crate::render_graph::Pass;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMemory {
    pub pass: Option<&'static str>,
    pub bytes: u64,
    pub created_after_warmup: u64,
    pub duplicate_upload_bytes: u64,
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one buffer creation, which records its owner"
)]
pub(crate) fn buffer(
    device: &wgpu::Device,
    descriptor: &wgpu::BufferDescriptor<'_>,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(descriptor);
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::created(ledger::id_of(&buffer), descriptor.size);
    buffer
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one initialized buffer creation, which records its owner"
)]
pub(crate) fn buffer_init(
    device: &wgpu::Device,
    descriptor: &wgpu::util::BufferInitDescriptor<'_>,
) -> wgpu::Buffer {
    let buffer = wgpu::util::DeviceExt::create_buffer_init(device, descriptor);
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::created(ledger::id_of(&buffer), buffer.size());
    buffer
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one texture creation, which records its owner"
)]
pub(crate) fn texture(
    device: &wgpu::Device,
    descriptor: &wgpu::TextureDescriptor<'_>,
) -> wgpu::Texture {
    let texture = device.create_texture(descriptor);
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::created(ledger::id_of(&texture), texture_bytes(descriptor));
    texture
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub(crate) fn release_buffer(buffer: &wgpu::Buffer) {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::released(ledger::id_of(buffer));
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub(crate) fn release_ids(ids: &[u64]) {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ids.iter().copied().for_each(ledger::released);
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) fn created_count() -> u64 {
    ledger::count()
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
pub(crate) fn created_count() -> u64 {
    0
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) fn created_since(count: u64) -> Vec<u64> {
    ledger::since(count)
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
pub(crate) fn created_since(_count: u64) -> Vec<u64> {
    Vec::new()
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub(crate) fn release_any(value: &dyn std::any::Any) {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::released_any(value);
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub(crate) fn uploaded_buffer(buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::uploaded(ledger::id_of(&(ledger::id_of(buffer), offset)), data);
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub(crate) fn uploaded_texture(texture: &wgpu::TexelCopyTextureInfo<'_>, data: &[u8]) {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::uploaded(
        ledger::id_of(&(
            ledger::id_of(texture.texture),
            texture.mip_level,
            texture.origin.x,
            texture.origin.y,
            texture.origin.z,
        )),
        data,
    );
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(dead_code, reason = "only measured builds keep a ledger")
)]
fn texture_bytes(descriptor: &wgpu::TextureDescriptor<'_>) -> u64 {
    let format = descriptor.format;
    let (block_width, block_height) = format.block_dimensions();
    let block = u64::from(format.block_copy_size(None).unwrap_or(4));
    let levels: u64 = (0..descriptor.mip_level_count)
        .map(|level| {
            let size = descriptor.size.mip_level_size(level, descriptor.dimension);
            u64::from(size.width.div_ceil(block_width.max(1)))
                * u64::from(size.height.div_ceil(block_height.max(1)))
                * u64::from(size.depth_or_array_layers)
                * block
        })
        .sum();
    levels * u64::from(descriptor.sample_count.max(1))
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) use ledger::{end_warmup, set_owners, take};

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) struct Owned {
    previous: Option<Pass>,
    _brick: engine_profile::BrickScope,
    _span: engine_profile::Span,
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) fn owned(pass: Pass) -> Owned {
    let (previous, brick) = ledger::own(pass);
    Owned {
        previous,
        _span: engine_profile::span("brick", engine_profile::brick_name(brick)),
        _brick: engine_profile::enter(brick),
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
impl Drop for Owned {
    fn drop(&mut self) {
        ledger::restore(self.previous);
    }
}

pub(crate) fn feature(pass: crate::render_graph::FeaturePass) -> Owned {
    owned(Pass::Feature(pass))
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
pub(crate) struct Owned;

#[cfg(not(any(feature = "dev", feature = "profiling")))]
#[inline(always)]
pub(crate) fn owned(_pass: Pass) -> Owned {
    Owned
}

#[cfg(any(feature = "dev", feature = "profiling"))]
mod ledger {
    use std::cell::{Cell, RefCell};
    use std::collections::{BTreeMap, VecDeque};
    use std::hash::{BuildHasher, Hash};

    use engine_core::collections::{FixedState, HashMap};

    use super::GpuMemory;
    use crate::render_graph::Pass;

    type Owner = Option<&'static str>;

    const RECENT: usize = 256;
    const UPLOADS: usize = 1 << 16;

    #[derive(Default)]
    struct Ledger {
        warm: bool,
        count: u64,
        recent: VecDeque<(u64, u64)>,
        live: HashMap<u64, (Owner, u64)>,
        bytes: BTreeMap<Owner, u64>,
        created: BTreeMap<Owner, u64>,
        duplicates: BTreeMap<Owner, u64>,
        uploads: HashMap<u64, u64>,
    }

    impl Ledger {
        fn release(&mut self, id: u64) {
            if let Some((owner, size)) = self.live.remove(&id)
                && let Some(bytes) = self.bytes.get_mut(&owner)
            {
                *bytes = bytes.saturating_sub(size);
            }
        }
    }

    thread_local! {
        static LEDGER: RefCell<Ledger> = RefCell::new(Ledger::default());
        static OWNER: Cell<Option<Pass>> = const { Cell::new(None) };
        static BRICKS: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn id_of(value: &impl Hash) -> u64 {
        FixedState::default().hash_one(value)
    }

    fn owner() -> Owner {
        OWNER.with(Cell::get).map(Pass::name)
    }

    pub(super) fn own(pass: Pass) -> (Option<Pass>, u16) {
        let previous = OWNER.replace(Some(pass));
        let brick = BRICKS.with_borrow(|bricks| {
            bricks
                .get(pass.index())
                .copied()
                .unwrap_or(engine_profile::CORE)
        });
        (previous, brick)
    }

    pub(super) fn restore(previous: Option<Pass>) {
        OWNER.set(previous);
    }

    pub(crate) fn set_owners(bricks: Vec<u16>) {
        BRICKS.set(bricks);
    }

    pub(super) fn created(id: u64, bytes: u64) {
        let owner = owner();
        LEDGER.with_borrow_mut(|ledger| {
            ledger.release(id);
            ledger.count += 1;
            ledger.recent.push_back((ledger.count, id));
            if ledger.recent.len() > RECENT {
                ledger.recent.pop_front();
            }
            ledger.live.insert(id, (owner, bytes));
            *ledger.bytes.entry(owner).or_default() += bytes;
            if ledger.warm {
                *ledger.created.entry(owner).or_default() += 1;
            }
        });
    }

    pub(super) fn released(id: u64) {
        LEDGER.with_borrow_mut(|ledger| ledger.release(id));
    }

    pub(super) fn count() -> u64 {
        LEDGER.with_borrow(|ledger| ledger.count)
    }

    pub(super) fn since(count: u64) -> Vec<u64> {
        LEDGER.with_borrow(|ledger| {
            ledger
                .recent
                .iter()
                .filter(|&&(created, _)| created > count)
                .map(|&(_, id)| id)
                .collect()
        })
    }

    pub(super) fn released_any(value: &dyn std::any::Any) {
        let id = if let Some(buffer) = value.downcast_ref::<wgpu::Buffer>() {
            id_of(buffer)
        } else if let Some(Some(buffer)) = value.downcast_ref::<Option<wgpu::Buffer>>() {
            id_of(buffer)
        } else if let Some(texture) = value.downcast_ref::<wgpu::Texture>() {
            id_of(texture)
        } else {
            return;
        };
        released(id);
    }

    pub(super) fn uploaded(destination: u64, data: &[u8]) {
        let content = id_of(&data);
        let owner = owner();
        LEDGER.with_borrow_mut(|ledger| {
            if ledger.uploads.len() >= UPLOADS {
                ledger.uploads.clear();
            }
            if ledger.uploads.insert(destination, content) == Some(content) {
                *ledger.duplicates.entry(owner).or_default() += data.len() as u64;
            }
        });
    }

    pub(crate) fn end_warmup() {
        LEDGER.with_borrow_mut(|ledger| ledger.warm = true);
    }

    pub(crate) fn take() -> Vec<GpuMemory> {
        LEDGER.with_borrow_mut(|ledger| {
            let duplicates = std::mem::take(&mut ledger.duplicates);
            let mut owners: BTreeMap<Owner, GpuMemory> = BTreeMap::new();
            for (&owner, &bytes) in &ledger.bytes {
                row(&mut owners, owner).bytes += bytes;
            }
            for (&owner, &count) in &ledger.created {
                row(&mut owners, owner).created_after_warmup += count;
            }
            for (owner, bytes) in duplicates {
                row(&mut owners, owner).duplicate_upload_bytes += bytes;
            }
            owners.into_values().collect()
        })
    }

    fn row(owners: &mut BTreeMap<Owner, GpuMemory>, owner: Owner) -> &mut GpuMemory {
        owners.entry(owner).or_insert(GpuMemory {
            pass: owner,
            ..GpuMemory::default()
        })
    }
}
