use engine_render::entries::MeshEntries;
use engine_render::parts::{Create, GpuPart, PartDecl, PartEntry};

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "meshes",
        brick: "meshes",
        passes: &[],
        resources: &[],
        hooks: &[],
        contributes: &[],
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
        resolve: None,
    },
    create,
};

fn create(_cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(MeshesPart)
}

struct MeshesPart;

impl GpuPart for MeshesPart {
    fn mesh_entries(&self) -> Option<&'static MeshEntries> {
        Some(&crate::entries::ENTRIES)
    }
}
