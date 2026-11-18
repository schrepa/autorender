    pub fn scene<N: PassName>(&mut self) -> PassBuilder<'_, P> {
        let pass = self.core.pass(N::NAME, install::Work::Render);
        self.core.draws_scene(pass, N::NAME, N::PASS);
        PassBuilder {
            core: &mut *self.core,
            pass,
            part: PhantomData,
        }
    }

    pub fn screen<N: PassName>(&mut self) -> PassBuilder<'_, P> {
        if N::PASS.scene() {
            let path = format!("$.passes.{}", N::NAME);
            self.core
                .problems
                .push(Error::new(engine_core::ErrorCode::PassMismatch, path));
        }
        let pass = self.core.pass(N::NAME, install::Work::Render);
        if let Some(declared) = self.core.pass_mut(pass) {
            declared.views = N::PASS.views;
        }
        PassBuilder {
            core: &mut *self.core,
            pass,
            part: PhantomData,
        }
    }

    pub fn drawable(&self, mesh: GeometryHandle, material: Option<MaterialHandle>) -> bool {
        self.core.assets.meshes.get(mesh).is_some()
            && self.core.assets.materials.group(material).is_some()
    }

    pub fn mipped(&self, material: Option<MaterialHandle>) -> bool {
        self.core.assets.materials.mipped(material)
    }

