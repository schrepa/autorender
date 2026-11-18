impl PointInput {
    pub const fn set(self) -> PointSet {
        PointSet {
            name: self.name,
            ty: self.ty,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointSet {
    pub name: &'static str,
    pub ty: &'static str,
}
