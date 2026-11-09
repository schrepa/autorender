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
        let Some(run) = self.run.as_ref() else {
            return;
        };
        let mut changed = Vec::new();
        for (path, file) in run.machine.linker.files() {
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

    fn reinstall_edited(&mut self, world: &mut World) -> Result<(), Vec<Error>> {
        let mut installed = super::install::install(
            &self.core.device,
            self.core.limits,
            world,
            self.watched.edited(),
        )?;
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
