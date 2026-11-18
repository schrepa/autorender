    pub(crate) fn speedup_only(&mut self, pipeline: u32) {
        if let Some(declared) = self.declared.pipelines.get_mut(pipeline as usize) {
            declared.speedup_only = true;
        }
    }
