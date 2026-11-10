#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointInput {
    pub name: &'static str,
    pub ty: &'static str,
    pub standard: &'static str,
}

impl PointInput {
    pub const fn new(name: &'static str, ty: &'static str, standard: &'static str) -> Self {
        Self { name, ty, standard }
    }

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
