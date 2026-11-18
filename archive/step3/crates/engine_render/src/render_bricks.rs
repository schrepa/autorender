    fn slot_prepare(&mut self, prepare: SlotPrepare) -> &mut Self;
    fn open_slot(&mut self, slot: OpenSlot) -> &mut Self;
    fn point_set(&mut self, set: PointSet) -> &mut Self;
    fn surface_terms(&mut self, terms: SurfaceTerms) -> &mut Self;
    fn surface_kind(&mut self, kind: SurfaceKind) -> &mut Self;

        if module.slot == engine_core::SurfaceSlot::Position.name() {
            self.record()
                .render
                .push(Box::new(engine_core::PositionReach {
                    module: module.name,
                    reach: module.reach,
                }));
        }

    fn open_slot(&mut self, slot: OpenSlot) -> &mut Self {
        self.record().render.push(Box::new(slot));
        self
    }

    fn point_set(&mut self, set: PointSet) -> &mut Self {
        self.record().render.push(Box::new(set));
        self
    }

    fn surface_terms(&mut self, terms: SurfaceTerms) -> &mut Self {
        self.record().render.push(Box::new(terms));
        self
    }

    fn slot_prepare(&mut self, prepare: SlotPrepare) -> &mut Self {
        self.record().render.push(Box::new(prepare));
        self
    }

    fn surface_kind(&mut self, kind: SurfaceKind) -> &mut Self {
        self.record().render.push(Box::new(kind));
        self
    }
