#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMemory {
    pub brick: Option<&'static str>,
    pub bytes: u64,
    pub created_after_warmup: u64,
    pub duplicate_upload_bytes: u64,
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one buffer creation, which records its owner"
)]
pub fn buffer(device: &wgpu::Device, descriptor: &wgpu::BufferDescriptor<'_>) -> wgpu::Buffer {
    let buffer = device.create_buffer(descriptor);
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::created(ledger::id_of(&buffer), descriptor.size);
    buffer
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one initialized buffer creation, which records its owner"
)]
pub fn buffer_init(
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
pub fn texture(device: &wgpu::Device, descriptor: &wgpu::TextureDescriptor<'_>) -> wgpu::Texture {
    let texture = device.create_texture(descriptor);
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::created(ledger::id_of(&texture), texture_bytes(descriptor));
    texture
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub fn release_buffer(buffer: &wgpu::Buffer) {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::released(ledger::id_of(buffer));
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub fn release_any(value: &dyn std::any::Any) {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::released_any(value);
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub fn uploaded_buffer(buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    #[cfg(any(feature = "dev", feature = "profiling"))]
    ledger::uploaded(ledger::id_of(&(ledger::id_of(buffer), offset)), data);
}

#[cfg_attr(
    not(any(feature = "dev", feature = "profiling")),
    expect(unused_variables, reason = "only measured builds keep a ledger")
)]
pub fn uploaded_texture(texture: &wgpu::TexelCopyTextureInfo<'_>, data: &[u8]) {
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
pub use ledger::{end_warmup, take};

#[cfg(any(feature = "dev", feature = "profiling"))]
pub struct Owned {
    previous: Option<&'static str>,
    _brick: engine_profile::BrickScope,
    _span: engine_profile::Span,
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub fn owned(brick: &'static str, index: u16) -> Owned {
    let previous = ledger::own(brick);
    Owned {
        previous,
        _span: engine_profile::span("brick", brick),
        _brick: engine_profile::enter(index),
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
impl Drop for Owned {
    fn drop(&mut self) {
        ledger::restore(self.previous);
    }
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
pub struct Owned;

#[cfg(not(any(feature = "dev", feature = "profiling")))]
#[inline(always)]
pub fn owned(_brick: &'static str, _index: u16) -> Owned {
    Owned
}

#[cfg(any(feature = "dev", feature = "profiling"))]
mod ledger {
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;
    use std::hash::{BuildHasher, Hash};

    use engine_core::collections::{FixedState, HashMap};

    use super::GpuMemory;

    type Owner = Option<&'static str>;

    const UPLOADS: usize = 1 << 16;

    #[derive(Default)]
    struct Ledger {
        warm: bool,
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
        static OWNER: Cell<Owner> = const { Cell::new(None) };
    }

    pub(super) fn id_of(value: &impl Hash) -> u64 {
        FixedState::default().hash_one(value)
    }

    fn owner() -> Owner {
        OWNER.with(Cell::get)
    }

    pub(super) fn own(brick: &'static str) -> Owner {
        OWNER.replace(Some(brick))
    }

    pub(super) fn restore(previous: Owner) {
        OWNER.set(previous);
    }

    pub(super) fn created(id: u64, bytes: u64) {
        let owner = owner();
        LEDGER.with_borrow_mut(|ledger| {
            ledger.release(id);
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

    pub fn end_warmup() {
        LEDGER.with_borrow_mut(|ledger| ledger.warm = true);
    }

    pub fn take() -> Vec<GpuMemory> {
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
            brick: owner,
            ..GpuMemory::default()
        })
    }
}
