use std::cell::{Cell, RefCell};
use std::hash::BuildHasher;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use engine_core::collections::{FixedState, HashMap};

mod submission;

pub use submission::Handover;

static CREATED_BUFFERS: AtomicU64 = AtomicU64::new(0);
static CREATED_TEXTURES: AtomicU64 = AtomicU64::new(0);
static CREATED_BUFFER_BYTES: AtomicU64 = AtomicU64::new(0);
static PLACED_BYTES: AtomicU64 = AtomicU64::new(0);

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one buffer creation"
)]
pub(crate) fn create_buffer(
    device: &wgpu::Device,
    descriptor: &wgpu::BufferDescriptor<'_>,
) -> wgpu::Buffer {
    CREATED_BUFFERS.fetch_add(1, Ordering::Relaxed);
    CREATED_BUFFER_BYTES.fetch_add(descriptor.size, Ordering::Relaxed);
    device.create_buffer(descriptor)
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one buffer creation with data"
)]
pub(crate) fn create_buffer_init(
    device: &wgpu::Device,
    descriptor: &wgpu::util::BufferInitDescriptor<'_>,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let made = device.create_buffer_init(descriptor);
    CREATED_BUFFERS.fetch_add(1, Ordering::Relaxed);
    CREATED_BUFFER_BYTES.fetch_add(made.size(), Ordering::Relaxed);
    PLACED_BYTES.fetch_add(descriptor.contents.len() as u64, Ordering::Relaxed);
    made
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one texture creation"
)]
pub(crate) fn create_texture(
    device: &wgpu::Device,
    descriptor: &wgpu::TextureDescriptor<'_>,
) -> wgpu::Texture {
    CREATED_TEXTURES.fetch_add(1, Ordering::Relaxed);
    device.create_texture(descriptor)
}

#[cfg(feature = "dev")]
pub(crate) fn created() -> serde_json::Value {
    serde_json::json!({
        "buffers": CREATED_BUFFERS.load(Ordering::Relaxed),
        "textures": CREATED_TEXTURES.load(Ordering::Relaxed),
    })
}

#[cfg(feature = "dev")]
pub(crate) fn created_bytes() -> serde_json::Value {
    serde_json::json!({
        "buffers": CREATED_BUFFER_BYTES.load(Ordering::Relaxed),
        "placed": PLACED_BYTES.load(Ordering::Relaxed),
    })
}

#[derive(Clone)]
pub struct Queue(Rc<Shared>);

struct Shared {
    submission: submission::Submission,
    queue: wgpu::Queue,
    submitted_at_once: Cell<u64>,
    writes: RefCell<Writes>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Written {
    pub(crate) bytes: u64,
    pub(crate) unchanged: u64,
}

impl Written {
    fn add(&mut self, bytes: u64, unchanged: u64) {
        self.bytes += bytes;
        self.unchanged += unchanged;
    }
}

#[derive(Default)]
pub(crate) struct Writes {
    pub(crate) total: Written,
    pub(crate) engine: Written,
    pub(crate) buffers: Vec<Written>,
    pub(crate) textures: Vec<Written>,
    taken: u64,
    marks: Option<HashMap<u64, u64>>,
}

#[cfg(feature = "dev")]
impl Writes {
    pub(crate) fn marking(&self) -> bool {
        self.marks.is_some()
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Count {
    pub(crate) draws: u64,
    pub(crate) instances: u64,
}

impl Count {
    fn add(&mut self, instances: u32) {
        self.draws += 1;
        self.instances += u64::from(instances);
    }

    pub(crate) fn since(self, earlier: Self) -> Self {
        Self {
            draws: self.draws - earlier.draws,
            instances: self.instances - earlier.instances,
        }
    }
}

#[derive(Default)]
pub(crate) struct Drawn {
    pub(crate) total: Count,
    pub(crate) passes: Vec<Count>,
    pub(crate) parts: Vec<Count>,
}

impl Drawn {
    pub(crate) fn new(passes: usize, parts: usize) -> Self {
        Self {
            total: Count::default(),
            passes: vec![Count::default(); passes],
            parts: vec![Count::default(); parts],
        }
    }

    pub(crate) fn add(&mut self, (pass, part): (u32, u32), instances: u32) {
        self.total.add(instances);
        if let Some(count) = self.passes.get_mut(pass as usize) {
            count.add(instances);
        }
        if let Some(count) = self.parts.get_mut(part as usize) {
            count.add(instances);
        }
    }
}

#[cfg(feature = "dev")]
#[derive(bevy_ecs::resource::Resource)]
pub struct UnchangedWrites;

#[cfg(feature = "dev")]
pub fn unchanged_writes(
    world: &mut bevy_ecs::world::World,
    on: bool,
) -> Result<(), engine_core::Error> {
    if on {
        world.insert_resource(UnchangedWrites);
    } else {
        world.remove_resource::<UnchangedWrites>();
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Owner {
    Engine,
    Buffer(u32),
    Texture(u32),
}

fn slot(slots: &mut Vec<Written>, at: u32) -> &mut Written {
    let at = at as usize;
    if slots.len() <= at {
        slots.resize(at + 1, Written::default());
    }
    &mut slots[at]
}

impl Queue {
    pub fn new(queue: wgpu::Queue) -> Self {
        Self(Rc::new(Shared {
            submission: submission::Submission::new(&queue),
            queue,
            submitted_at_once: Cell::new(0),
            writes: RefCell::new(Writes::default()),
        }))
    }

    pub fn hand(&self, handover: Handover) {
        self.0.submission.hand(&self.0.queue, handover);
    }

    pub fn settle(&self) {
        self.0.submission.settle();
    }

    pub(crate) fn finishers(&self) -> usize {
        self.0.submission.finishers()
    }

    pub fn submit_now(&self, buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.settle();
        self.0
            .submitted_at_once
            .set(self.0.submitted_at_once.get() + 1);
        submit(&self.0.queue, buffers);
    }

    #[cfg(feature = "dev")]
    pub fn submitted_at_once(&self) -> u64 {
        self.0.submitted_at_once.get()
    }

    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.write_buffer_of(Owner::Engine, buffer, offset, data);
    }

    pub fn write_declared_buffer(
        &self,
        declared: u32,
        buffer: &wgpu::Buffer,
        offset: u64,
        data: &[u8],
    ) {
        self.write_buffer_of(Owner::Buffer(declared), buffer, offset, data);
    }

    pub fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.write_texture_of(Owner::Engine, texture, data, layout, size);
    }

    pub fn write_declared_texture(
        &self,
        declared: u32,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.write_texture_of(Owner::Texture(declared), texture, data, layout, size);
    }

    #[expect(clippy::disallowed_methods, reason = "the renderer's one buffer write")]
    fn write_buffer_of(&self, owner: Owner, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.count(owner, data, |hasher| {
            hasher.hash_one((buffer, offset, data.len()))
        });
        self.0.queue.write_buffer(buffer, offset, data);
    }

    #[expect(clippy::disallowed_methods, reason = "the renderer's one texture write")]
    fn write_texture_of(
        &self,
        owner: Owner,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.count(owner, data, |hasher| {
            let at = texture.origin;
            hasher.hash_one((
                texture.texture,
                texture.mip_level,
                [at.x, at.y, at.z],
                [size.width, size.height, size.depth_or_array_layers],
                data.len(),
            ))
        });
        self.0.queue.write_texture(texture, data, layout, size);
    }

    fn count(&self, owner: Owner, data: &[u8], range: impl FnOnce(&FixedState) -> u64) {
        let mut writes = self.0.writes.borrow_mut();
        let writes = &mut *writes;
        let bytes = data.len() as u64;
        let unchanged = match &mut writes.marks {
            Some(marks) => {
                let hasher = FixedState::default();
                let content = hasher.hash_one(data);
                if marks.insert(range(&hasher), content) == Some(content) {
                    bytes
                } else {
                    0
                }
            }
            None => 0,
        };
        writes.total.add(bytes, unchanged);
        match owner {
            Owner::Engine => writes.engine.add(bytes, unchanged),
            Owner::Buffer(at) => slot(&mut writes.buffers, at).add(bytes, unchanged),
            Owner::Texture(at) => slot(&mut writes.textures, at).add(bytes, unchanged),
        }
    }

    #[cfg(feature = "dev")]
    pub(crate) fn writes(&self) -> std::cell::Ref<'_, Writes> {
        self.0.writes.borrow()
    }

    pub fn take_written(&self) -> u64 {
        let mut writes = self.0.writes.borrow_mut();
        let fresh = writes.total.bytes - writes.taken;
        writes.taken = writes.total.bytes;
        fresh
    }

    #[cfg(feature = "dev")]
    pub fn mark_unchanged(&self, on: bool) {
        let mut writes = self.0.writes.borrow_mut();
        if on != writes.marks.is_some() {
            writes.marks = on.then(HashMap::default);
        }
    }

    pub fn on_submitted_work_done(&self, callback: impl FnOnce() + Send + 'static) {
        self.0.queue.on_submitted_work_done(callback);
    }

    pub fn get_timestamp_period(&self) -> f32 {
        self.0.queue.get_timestamp_period()
    }

    pub fn present(&self, texture: wgpu::SurfaceTexture) {
        self.0.queue.present(texture);
    }
}

#[expect(clippy::disallowed_methods, reason = "the renderer's one submission")]
fn submit(
    queue: &wgpu::Queue,
    buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
) -> wgpu::SubmissionIndex {
    queue.submit(buffers)
}

#[expect(clippy::disallowed_methods, reason = "the renderer's one error scope")]
pub fn validation_scope(device: &wgpu::Device) -> wgpu::ErrorScopeGuard {
    device.push_error_scope(wgpu::ErrorFilter::Validation)
}
