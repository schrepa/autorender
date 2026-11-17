use bevy_ecs::resource::Resource;

#[derive(Resource)]
pub struct Asked<A: Send + Sync + 'static> {
    pending: Vec<A>,
}

impl<A: Send + Sync + 'static> Default for Asked<A> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
        }
    }
}

impl<A: Send + Sync + 'static> Asked<A> {
    pub fn ask(&mut self, ask: A) {
        self.pending.push(ask);
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn pending(&self) -> impl ExactSizeIterator<Item = &A> + '_ {
        self.pending.iter()
    }

    pub fn take(&mut self) -> Vec<A> {
        std::mem::take(&mut self.pending)
    }

    pub fn take_where(&mut self, taken: impl Fn(&A) -> bool) -> Vec<A> {
        let (chosen, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|ask| taken(ask));
        self.pending = kept;
        chosen
    }
}
