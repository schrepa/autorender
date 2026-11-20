use std::collections::BTreeMap;
use std::time::SystemTime;

use bevy_ecs::world::World;

use super::*;

#[derive(Default)]
pub(crate) struct Watched {
    files: Vec<(&'static str, &'static str, Option<SystemTime>)>,
    edited: BTreeMap<String, String>,
    errors: Vec<Error>,
}

impl Watched {
    pub(super) fn edited(&self) -> BTreeMap<String, String> {
        self.edited.clone()
    }

    pub(super) fn errors(&self) -> &[Error] {
        &self.errors
    }
}

fn modified(file: &str) -> Option<SystemTime> {
    std::fs::metadata(file)
        .and_then(|metadata| metadata.modified())
        .ok()
}

impl Renderer {
    pub fn reload_shaders(&mut self, world: &mut World) {
        if self.run.is_none() {
            return;
        }
        let Some(registry) = world.get_resource::<engine_registry::Registry>() else {
            return;
        };
        let sources = crate::linker::sources(registry);
        let files = (std::iter::once(crate::shared::core_source()).chain(sources))
            .flat_map(engine_core::Source::modules)
            .filter(|module| !module.file().is_empty());
        let mut changed = Vec::new();
        for (path, file) in files.map(|module| (module.path(), module.file())) {
            let now = modified(file);
            match self
                .watched
                .files
                .iter_mut()
                .find(|(known, ..)| *known == path)
            {
                Some((_, _, seen)) if *seen != now => {
                    *seen = now;
                    changed.push((path, file));
                }
                Some(_) => {}
                None => self.watched.files.push((path, file, now)),
            }
        }
        if changed.is_empty() {
            return;
        }
        for (path, file) in changed {
            match std::fs::read_to_string(file) {
                Ok(text) => {
                    self.watched.edited.insert(path.to_owned(), text);
                }
                Err(error) => {
                    self.watched.errors = vec![failed_with(&format!("$.shaders.{path}"), error)];
                    return;
                }
            }
        }
        self.watched.errors = match self.reinstall_edited(world) {
            Ok(()) => Vec::new(),
            Err(errors) => errors,
        };
    }

    pub fn reload_shaders_now(
        &mut self,
        world: &mut World,
    ) -> Result<Vec<&'static str>, Vec<Error>> {
        if self.run.is_none() {
            return Ok(Vec::new());
        }
        let Some(registry) = world.get_resource::<engine_registry::Registry>() else {
            return Ok(Vec::new());
        };
        let sources = crate::linker::sources(registry);
        let modules: Vec<_> = (std::iter::once(crate::shared::core_source()).chain(sources))
            .flat_map(engine_core::Source::modules)
            .filter(|module| !module.file().is_empty())
            .collect();
        let previous = self.watched.edited.clone();
        let mut changed = Vec::new();
        for module in modules {
            let path = module.path();
            let file = module.file();
            let stamp = modified(file);
            match self
                .watched
                .files
                .iter_mut()
                .find(|(known, ..)| *known == path)
            {
                Some((_, _, seen)) => *seen = stamp,
                None => self.watched.files.push((path, file, stamp)),
            }
            let text = match std::fs::read_to_string(file) {
                Ok(text) => text,
                Err(error) => {
                    self.watched.edited = previous;
                    let errors = vec![failed_with(&format!("$.shaders.{path}"), error)];
                    self.watched.errors = errors.clone();
                    return Err(errors);
                }
            };
            let current = self
                .watched
                .edited
                .get(path)
                .map_or(module.text(), String::as_str);
            if text != current {
                if text == module.text() {
                    self.watched.edited.remove(path);
                } else {
                    self.watched.edited.insert(path.to_owned(), text);
                }
                changed.push(path);
            }
        }
        if changed.is_empty() {
            self.watched.errors.clear();
            return Ok(changed);
        }
        match self.reinstall_edited(world) {
            Ok(()) => {
                self.watched.errors.clear();
                Ok(changed)
            }
            Err(errors) => {
                self.watched.edited = previous;
                self.watched.errors = errors.clone();
                Err(errors)
            }
        }
    }

    fn reinstall_edited(&mut self, world: &mut World) -> Result<(), Vec<Error>> {
        let linking = super::install::Linking {
            edited: self.watched.edited(),
            check: true,
        };
        let mut installed = super::install::install(&self.core.device, world, linking)?;
        if let Some(run) = &self.run {
            installed.machine.shaders = run.machine.shaders.clone();
        }
        let scope = crate::gpu_calls::validation_scope(&self.core.device);
        let machine = &mut installed.machine;
        for built in &machine.pipelines {
            for stage in &built.stages {
                stage.shader(&self.core.device, &mut machine.shaders);
            }
        }
        if let Some(Some(error)) = crate::platform::ready(scope.pop()) {
            return Err(vec![failed_with("$.shaders", error)]);
        }
        self.adopt_installed(installed, world);
        Ok(())
    }
}
