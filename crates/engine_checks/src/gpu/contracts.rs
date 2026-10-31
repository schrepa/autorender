use std::fmt::Write as _;

use engine::bricks::{AddBrick, BrickRegistry};
use engine_core::{Place, PlaceKind};
use engine_render::faces::face_module;
use engine_render::linker::Linker;
use engine_render::parts::CORE_BRICK;
use engine_render::places::{
    Answered, Contract, PLACE_OWNER, contract, contract_problems, face_source, place_module,
};
use engine_render::shaders::MODULES;

const PROBE_ROOT: &str = "generated::probe";

pub fn catalog() -> Vec<Place> {
    let mut engine = engine::Engine::default();
    engine.app.add_bricks(engine::bricks::standard());
    let registered = engine
        .app
        .world()
        .get_resource::<BrickRegistry>()
        .map(BrickRegistry::places)
        .unwrap_or_default();
    let mut places: Vec<Place> = Vec::new();
    for place in [
        engine_render::composite::POST_PLACES,
        engine_faces::places::PLACES,
        engine_places::PLACES,
    ]
    .into_iter()
    .flatten()
    .chain(&registered)
    {
        if !places.iter().any(|known| known.name == place.name) {
            places.push(*place);
        }
    }
    places
}

fn probe_source(place: &str, contract: &Contract) -> String {
    let mut source = String::new();
    for import in &contract.imports {
        let _ = writeln!(source, "{import}");
    }
    let names = contract.names();
    if !names.is_empty() {
        let _ = writeln!(
            source,
            "import package::{}::{{{}}};",
            face_module(place),
            names.join(", ")
        );
    }
    let _ = writeln!(source, "\nfn probed() {{");
    for ty in &contract.types {
        let _ = writeln!(source, "    _ = {ty}();");
    }
    for answer in contract.shared.iter().chain(&contract.answers) {
        if answer.constant {
            let _ = writeln!(source, "    _ = {};", answer.name);
            continue;
        }
        let arguments: Vec<String> = answer
            .parameters
            .iter()
            .map(|(_, ty)| format!("{ty}()"))
            .collect();
        let call = format!("{}({})", answer.name, arguments.join(", "));
        let _ = match answer.returns {
            Some(_) => writeln!(source, "    _ = {call};"),
            None => writeln!(source, "    {call};"),
        };
    }
    let _ = writeln!(source, "}}");
    source
}

pub fn check() -> Result<usize, String> {
    let core = |module: &str| MODULES.iter().any(|known| known.path == module);
    let mut problems = Vec::new();
    let mut contracts: Vec<(Place, &'static str, Contract)> = Vec::new();
    for place in catalog() {
        let Some(source) = place.face.wgsl else {
            continue;
        };
        for problem in contract_problems(source, place.kind, &core) {
            problems.push(format!("place {}: {problem}", place.name));
        }
        match contract(source) {
            Ok(found) => contracts.push((place, source, found)),
            Err(error) => problems.push(format!("place {}: {error}", place.name)),
        }
    }
    let layout = super::placement(MODULES);
    for (place, _, found) in &contracts {
        let mut linker = Linker::new(MODULES);
        for (other, source, _) in &contracts {
            linker.add_generated(place_module(other.name), PLACE_OWNER, (*source).to_owned());
        }
        let answered = match place.kind {
            PlaceKind::Slot => Answered::Nobody,
            PlaceKind::List => Answered::Members(Vec::new()),
        };
        linker.add_generated(
            face_module(place.name),
            CORE_BRICK,
            face_source(place.name, found, &answered),
        );
        let probe = format!("{PROBE_ROOT}::{}", place.name);
        linker.add_generated(probe.clone(), CORE_BRICK, probe_source(place.name, found));
        let linked = linker
            .compose_module(&probe, &layout)
            .map_err(|error| error.to_string())
            .and_then(|composed| super::validated(&composed.text));
        if let Err(error) = linked {
            problems.push(format!(
                "place {} does not compose through its face: {error}",
                place.name
            ));
        }
    }
    if problems.is_empty() {
        Ok(contracts.len())
    } else {
        Err(problems.join("\n"))
    }
}
