//! The highlands' spruces: the engine's conifers ([`Conifer`]), each
//! level's crown and fringe a mesh record and their atlas a texture record,
//! built in place where no host holds them.

use engine::prelude::*;
use engine::terrain::{Conifer, ConiferPart};

use crate::kit::records;

/// Conifer `conifer`'s mesh of `part` at `level` as a record
/// ([`records::mesh`]): its handle, reserved until the record is here.
pub(crate) fn conifer_record(
    world: &mut World,
    conifer: Conifer,
    level: usize,
    part: ConiferPart,
) -> engine::Result<MeshHandle> {
    let mut identity = conifer.seed.to_le_bytes().to_vec();
    identity.extend_from_slice(&(level as u32).to_le_bytes());
    identity.push(match part {
        ConiferPart::Crown => 0,
        ConiferPart::Fringe => 1,
    });
    records::mesh(world, "highlands_conifer", &identity, move || {
        conifer.mesh(level, part).map_err(|error| {
            engine::Error::from(error).context(format!(
                "conifer {} at level {level}",
                conifer.seed
            ))
        })
    })
}

/// The conifers' atlas as a texture record ([`records::textures`]), cooked
/// as every generated texture is: its handle, which materials name at once.
pub(crate) fn atlas_record(world: &mut World) -> engine::Result<TextureHandle> {
    let handles = world.resource_scope(|world, mut stored: Mut<Records>| {
        let mut assets = world.resource_mut::<AssetLibrary>();
        records::textures(
            (&mut stored, &mut assets),
            "highlands/conifer_atlas".into(),
            &[("atlas", TextureUsage::Color)],
            || Ok(vec![Conifer::atlas()]),
        )
    })?;
    Ok(handles[0])
}
