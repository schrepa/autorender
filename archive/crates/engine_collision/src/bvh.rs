use engine_core::profile::Aggregate;

struct Profiled<'a, I> {
    inner: I,
    timing: &'a mut Aggregate,
}

impl<I: Iterator> Iterator for Profiled<'_, I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.timing.measure(|| self.inner.next())
    }
}

pub(crate) fn profiled<'a, I: Iterator + 'a>(
    inner: I,
    timing: &'a mut Aggregate,
) -> impl Iterator<Item = I::Item> + 'a {
    Profiled { inner, timing }
}
