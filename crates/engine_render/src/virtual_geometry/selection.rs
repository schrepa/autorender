//! The storage of each frame's selection (decision 044): batches of both ID
//! sources, the bins of selected clusters and their regions, the node and
//! group queues, the retest list, counters, and dispatch and draw
//! arguments, grown to what frames read back needed, and the uniform of
//! views and the camera's tests.

use engine_core::glam::Vec3;

use crate::buffers::storage;

use super::{
    CAMERA, COUNTERS, CameraTests, Cone, MAX_VIEWS, NONE, TEST_PREVIOUS, TEST_SUBPIXEL,
    UNIFORM_BYTES, VIEW_FLOATS, VISIBLE_BYTES, VirtualBatch, VirtualView,
};

/// Starting capacities, grown from read-back counts.
const START_QUEUE: u64 = 1 << 16;
const START_BIN: u64 = 1 << 14;
const START_RETEST: u64 = 1 << 14;

pub(super) struct SelectionStorage {
    pub(super) uniform: wgpu::Buffer,
    // Batches of each ID source, their capacities and counts, and the most
    // instances any of them has.
    pub(super) batches: [wgpu::Buffer; 2],
    batch_capacity: [u64; 2],
    batch_counts: [u32; 2],
    max_instances: [u32; 2],
    pub(super) bins: wgpu::Buffer,
    // Each bin's base and capacity, as uploaded.
    bin_regions: Vec<(u64, u64)>,
    bin_capacity: u64,
    pub(super) queues: [wgpu::Buffer; 2],
    pub(super) work: wgpu::Buffer,
    queue_capacity: u64,
    pub(super) visible: wgpu::Buffer,
    visible_capacity: u64,
    pub(super) counters: wgpu::Buffer,
    // Counters and bins; the used and requested group bits follow, this
    // many words each.
    counter_capacity: u64,
    feedback_words: u64,
    pub(super) dispatch: wgpu::Buffer,
    pub(super) draw_args: wgpu::Buffer,
    // Clusters waiting for this frame's depth pyramid, and whether this
    // frame retests them.
    pub(super) retest: wgpu::Buffer,
    retest_capacity: u64,
    retesting: bool,
    // Advances whenever these buffers are replaced.
    generation: u64,
}

impl SelectionStorage {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let none = wgpu::BufferUsages::empty();
        Self {
            uniform: crate::buffers::uniform(device, "engine/virtual-frame", UNIFORM_BYTES),
            batches: [
                storage(device, "engine/virtual-camera-batches", 32, none),
                storage(device, "engine/virtual-shadow-batches", 32, none),
            ],
            batch_capacity: [1, 1],
            batch_counts: [0, 0],
            max_instances: [0, 0],
            bins: storage(device, "engine/virtual-bins", 16, none),
            bin_regions: Vec::new(),
            bin_capacity: START_BIN,
            queues: [
                storage(device, "engine/virtual-queue-a", START_QUEUE * 16, none),
                storage(device, "engine/virtual-queue-b", START_QUEUE * 16, none),
            ],
            work: storage(device, "engine/virtual-work", START_QUEUE * 16, none),
            queue_capacity: START_QUEUE,
            visible: storage(
                device,
                "engine/virtual-visible",
                VISIBLE_BYTES,
                wgpu::BufferUsages::VERTEX,
            ),
            visible_capacity: 1,
            counters: storage(
                device,
                "engine/virtual-counters",
                COUNTERS as u64 * 4,
                wgpu::BufferUsages::COPY_SRC,
            ),
            counter_capacity: COUNTERS as u64,
            feedback_words: 0,
            dispatch: storage(
                device,
                "engine/virtual-dispatch",
                24,
                wgpu::BufferUsages::INDIRECT,
            ),
            draw_args: storage(
                device,
                "engine/virtual-draws",
                16,
                wgpu::BufferUsages::INDIRECT,
            ),
            retest: storage(device, "engine/virtual-retest", START_RETEST * 16, none),
            retest_capacity: START_RETEST,
            retesting: false,
            generation: 0,
        }
    }

    /// Advances whenever these buffers are replaced.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        self.queues[0].size()
            + self.queues[1].size()
            + self.work.size()
            + self.visible.size()
            + self.counters.size()
            + self.retest.size()
    }

    /// Each source's batches this frame, and the most instances one has.
    pub(super) fn batches(&self) -> ([u32; 2], [u32; 2]) {
        (self.batch_counts, self.max_instances)
    }

    /// Bins this frame selects into.
    pub(super) fn bin_count(&self) -> usize {
        self.bin_regions.len()
    }

    /// Whether this frame retests the camera's waiting clusters.
    pub(super) fn retesting(&self) -> bool {
        self.retesting
    }

    /// The retest list's room, in clusters.
    pub(super) fn retest_capacity(&self) -> u32 {
        self.retest_capacity as u32
    }

    /// Where the used and requested group bits start in the counters, and
    /// their words each.
    pub(super) fn feedback(&self) -> (usize, usize) {
        (self.counter_capacity as usize, self.feedback_words as usize)
    }

    /// Grows what a frame read back lacked room for, `drops`: the queues
    /// (dropped node or group entries), bins' regions of this frame's
    /// `bins` bins (dropped clusters) and the retest list (overflowed),
    /// while a binding still holds them.
    pub(super) fn grow(&mut self, device: &wgpu::Device, drops: [u32; 4], bins: usize) {
        let [queue_drops, work_drops, cluster_drops, overflow] = drops;
        let binding = device.limits().max_storage_buffer_binding_size;
        if overflow > 0 && self.retest_capacity * 32 <= binding {
            self.retest_capacity *= 2;
            self.retest = storage(
                device,
                "engine/virtual-retest",
                self.retest_capacity * 16,
                wgpu::BufferUsages::empty(),
            );
            self.generation += 1;
        }
        if (queue_drops > 0 || work_drops > 0) && self.queue_capacity * 32 <= binding {
            self.queue_capacity *= 2;
            let none = wgpu::BufferUsages::empty();
            self.queues = [
                storage(
                    device,
                    "engine/virtual-queue-a",
                    self.queue_capacity * 16,
                    none,
                ),
                storage(
                    device,
                    "engine/virtual-queue-b",
                    self.queue_capacity * 16,
                    none,
                ),
            ];
            self.work = storage(
                device,
                "engine/virtual-work",
                self.queue_capacity * 16,
                none,
            );
            self.generation += 1;
        }
        if cluster_drops > 0 && self.bin_capacity * VISIBLE_BYTES * 2 * bins as u64 <= binding {
            self.bin_capacity *= 2;
        }
    }

    /// Gives `bins` bins equal shares of the visible storage and room in
    /// the counters, followed by `feedback_words` words of each feedback
    /// bitset; returns whether the counters were replaced.
    pub(super) fn size_bins(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bins: usize,
        feedback_words: u64,
    ) -> bool {
        let wanted = self.bin_capacity * bins.max(1) as u64;
        if wanted > self.visible_capacity {
            self.visible_capacity = wanted;
            self.visible = storage(
                device,
                "engine/virtual-visible",
                wanted * VISIBLE_BYTES,
                wgpu::BufferUsages::VERTEX,
            );
            self.generation += 1;
        }
        self.bin_regions = (0..bins)
            .map(|bin| (bin as u64 * self.bin_capacity, self.bin_capacity))
            .collect();
        let bin_words: Vec<u32> = self
            .bin_regions
            .iter()
            .flat_map(|&(base, capacity)| [base as u32, capacity as u32, 0, 0])
            .collect();
        let mut replaced = false;
        if bins as u64 > self.counter_capacity - COUNTERS as u64
            || bin_words.len() * 4 > self.bins.size() as usize
            || feedback_words != self.feedback_words
        {
            let capacity = (bins.max(1) as u64).next_power_of_two();
            self.counter_capacity = COUNTERS as u64 + capacity;
            self.feedback_words = feedback_words;
            self.counters = storage(
                device,
                "engine/virtual-counters",
                (self.counter_capacity + 2 * feedback_words) * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            self.bins = storage(
                device,
                "engine/virtual-bins",
                capacity * 16,
                wgpu::BufferUsages::empty(),
            );
            self.draw_args = storage(
                device,
                "engine/virtual-draws",
                capacity * 16,
                wgpu::BufferUsages::INDIRECT,
            );
            replaced = true;
            self.generation += 1;
        }
        if !bin_words.is_empty() {
            queue.write_buffer(&self.bins, 0, bytemuck::cast_slice(&bin_words));
        }
        replaced
    }

    /// Uploads both sources' `batches`, of which the largest of each has
    /// `max_instances` instances.
    pub(super) fn upload_batches(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batches: [&[VirtualBatch]; 2],
        max_instances: [u32; 2],
    ) {
        self.max_instances = max_instances;
        for (source, batches) in batches.iter().enumerate() {
            if batches.len() as u64 > self.batch_capacity[source] {
                self.batch_capacity[source] = (batches.len() as u64).next_power_of_two();
                self.batches[source] = storage(
                    device,
                    if source == CAMERA {
                        "engine/virtual-camera-batches"
                    } else {
                        "engine/virtual-shadow-batches"
                    },
                    self.batch_capacity[source] * 32,
                    wgpu::BufferUsages::empty(),
                );
                self.generation += 1;
            }
            let words: Vec<u32> = batches
                .iter()
                .flat_map(|batch| {
                    [
                        batch.start,
                        batch.count.unwrap_or(NONE),
                        batch.draw,
                        batch.mesh,
                        batch.bin,
                        u32::from(batch.double_sided),
                        batch.view,
                        0,
                    ]
                })
                .collect();
            if !words.is_empty() {
                queue.write_buffer(&self.batches[source], 0, bytemuck::cast_slice(&words));
            }
            self.batch_counts[source] = batches.len() as u32;
        }
    }

    /// Writes the uniform: counts of `bins` bins and `views` views, the
    /// camera's `tests`, where selection marks used and requested groups and
    /// where the groups' residency bits start (`bits_base`), then the views.
    pub(super) fn write_uniform(
        &mut self,
        queue: &wgpu::Queue,
        bins: usize,
        views: &[VirtualView],
        tests: &CameraTests,
        bits_base: u64,
    ) {
        let counts = [
            self.queue_capacity as u32,
            self.queue_capacity as u32,
            bins as u32,
            views.len() as u32,
        ];
        let mut bytes: Vec<u8> = bytemuck::cast_slice(&counts).to_vec();
        self.retesting = tests.phase_two.is_some();
        let (width, height, levels) = tests.viewport;
        let flags = match (tests.previous, tests.phase_two) {
            (Some(_), Some(_)) => TEST_PREVIOUS,
            _ => 0,
        } | if tests.subpixel { TEST_SUBPIXEL } else { 0 };
        bytes.extend_from_slice(bytemuck::cast_slice(&tests.view_projection.to_cols_array()));
        bytes.extend_from_slice(bytemuck::cast_slice(
            &tests
                .previous
                .unwrap_or(tests.view_projection)
                .to_cols_array(),
        ));
        bytes.extend_from_slice(bytemuck::cast_slice(&[
            width as f32,
            height as f32,
            levels as f32,
            0.0,
        ]));
        bytes.extend_from_slice(bytemuck::cast_slice(&[
            flags,
            tests.phase_two.unwrap_or(0),
            self.retest_capacity as u32,
            0,
        ]));
        // Where selection marks used and requested groups, and where the
        // groups' residency bits start.
        bytes.extend_from_slice(bytemuck::cast_slice(&[
            self.counter_capacity as u32,
            (self.counter_capacity + self.feedback_words) as u32,
            bits_base as u32,
            0,
        ]));
        for view in views.iter().take(MAX_VIEWS) {
            let mut floats = Vec::with_capacity(VIEW_FLOATS);
            for plane in view.planes {
                floats.extend(plane);
            }
            floats.extend([view.eye.x, view.eye.y, view.eye.z, view.min_distance]);
            let (mode, direction) = match view.cone {
                Cone::Eye => (0.0, Vec3::ZERO),
                Cone::Along(direction) => (1.0, direction),
            };
            floats.extend([view.projection, view.threshold, mode, view.unseen_error]);
            floats.extend([direction.x, direction.y, direction.z, 0.0]);
            bytes.extend_from_slice(bytemuck::cast_slice(&floats));
        }
        queue.write_buffer(&self.uniform, 0, &bytes);
    }

    /// Bin `bin`'s first entry of the visible storage.
    pub(super) fn bin_base(&self, bin: usize) -> u32 {
        self.bin_regions
            .get(bin)
            .map_or(0, |&(base, _)| base as u32)
    }

    /// The selected clusters of bin `bin` (an instance-stepped vertex stream
    /// of visible ID and cluster) and its indirect draw arguments.
    pub(super) fn bin_source(&self, bin: usize) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        let (base, capacity) = self.bin_regions[bin];
        (
            self.visible
                .slice(base * VISIBLE_BYTES..(base + capacity) * VISIBLE_BYTES),
            &self.draw_args,
            bin as u64 * 16,
        )
    }
}
