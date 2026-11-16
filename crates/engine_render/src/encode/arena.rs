use super::*;

#[derive(Default)]
pub(crate) struct Arena {
    held: Option<(wgpu::Buffer, u64)>,
    slices: Vec<Option<(u64, u64)>>,
    owed: Vec<u32>,
}

impl Arena {
    pub(crate) fn args(
        &self,
        buffer: u32,
        (offset, size): (u64, u64),
    ) -> Option<(&wgpu::Buffer, u64)> {
        if self.owed.contains(&buffer) {
            return None;
        }
        let (held, _) = self.held.as_ref()?;
        let (at, laid) = (*self.slices.get(buffer as usize)?)?;
        (offset + size <= laid).then_some((held, at + offset))
    }

    pub(crate) fn owed_after(&mut self, buffer: u32) {
        let laid = self
            .slices
            .get(buffer as usize)
            .is_some_and(Option::is_some);
        if laid && !self.owed.contains(&buffer) {
            self.owed.push(buffer);
        }
    }
}

impl Run {
    pub(crate) fn lay_arena(&mut self) {
        let mut slices = std::mem::take(&mut self.arena.slices);
        slices.clear();
        let mut end = 0;
        for (index, decl) in self.machine.declared.buffers.iter().enumerate() {
            let held = self.tables.buffers.get(index).and_then(Option::as_ref);
            let slice = held
                .filter(|_| decl.args && !decl.per_view)
                .map(|(_, bytes)| (end, *bytes));
            if let Some((_, bytes)) = slice {
                end += bytes.next_multiple_of(ARENA_ALIGN);
            }
            slices.push(slice);
        }
        self.arena.slices = slices;
        self.arena.owed.clear();
        if end == 0 {
            return;
        }
        if self.arena.held.as_ref().is_none_or(|(_, bytes)| *bytes < end) {
            if let Some((old, _)) = self.arena.held.take() {
                crate::memory::release_buffer(&old);
            }
            let bytes = end.next_power_of_two();
            let made = crate::memory::buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/indirect-args"),
                    size: bytes,
                    usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );
            self.arena.held = Some((made, bytes));
        }
        let owed = (0..self.arena.slices.len() as u32)
            .filter(|&buffer| self.arena.slices[buffer as usize].is_some())
            .collect();
        self.arena.owed = owed;
    }

    pub(crate) fn owe_written(&mut self, pass: usize) {
        let writes = self.machine.graph.writes(pass);
        for (buffer, id) in self.machine.ids.buffers.iter().enumerate() {
            if writes.contains(id.index()) {
                self.arena.owed_after(buffer as u32);
            }
        }
    }

    pub(crate) fn owe(&mut self, buffer: u32) {
        self.arena.owed_after(buffer);
    }

    pub(crate) fn pay_arena(&mut self) {
        if self.arena.owed.is_empty() || self.recording.collecting() {
            return;
        }
        self.end_render();
        let Some((arena, _)) = self.arena.held.clone() else {
            return;
        };
        for buffer in std::mem::take(&mut self.arena.owed) {
            let at = buffer as usize;
            let held = self.tables.buffers.get(at).and_then(Option::as_ref);
            let slice = self.arena.slices.get(at).copied().flatten();
            let copied = match (held, slice) {
                (Some((made, bytes)), Some((_, laid))) if *bytes >= laid => Some(made.clone()),
                (_, _) => None,
            };
            let (Some(made), Some((into, laid)), Some(encoder)) =
                (copied, slice, self.recording.encoder())
            else {
                if let Some(slice) = self.arena.slices.get_mut(at) {
                    *slice = None;
                }
                continue;
            };
            encoder.copy_buffer_to_buffer(&made, 0, &arena, into, laid);
        }
    }
}

const ARENA_ALIGN: u64 = 256;
