    for (brick, open) in registry.render_entries::<OpenSlot>() {
        linker.add_source(open.standard.source());
        let (path, name) = crate::linker::path_of((open.standard, open.input.name));
        roots.push((path.to_owned(), name));
        declared.open.push((*open, path.to_owned(), package(brick)));
    }

    for (brick, set) in registry.render_entries::<PointSet>() {
        declared.sets.push((*set, package(brick)));
    }

    for (_, terms) in registry.render_entries::<SurfaceTerms>() {
        linker.add_source(terms.derive.source());
        let (path, name) = crate::linker::path_of((terms.derive, crate::surface::DERIVE_ENTRY));
        roots.push((path.to_owned(), name));
        declared.terms.push((*terms, path.to_owned()));
    }

    for (_, prepare) in registry.render_entries::<SlotPrepare>() {
        linker.add_source(prepare.module.source());
        let (path, name) = crate::linker::path_of((prepare.module, crate::surface::PREPARE_ENTRY));
        roots.push((path.to_owned(), name));
        declared.prepares.push(path.to_owned());
    }

    declared.kinds = registry
        .render_entries::<SurfaceKind>()
        .map(|(_, kind)| *kind)
        .collect();

    linker.add_generated(
        crate::shared::PASSES_MODULE.to_owned(),
        CORE_PACKAGE,
        scene_passes(world).wgsl(),
    );

    linker.add_generated(
        crate::shared::KINDS_MODULE.to_owned(),
        CORE_PACKAGE,
        world
            .get_resource::<engine_core::DrawnTable>()
            .map(engine_core::DrawnTable::wgsl)
            .unwrap_or_default(),
    );
