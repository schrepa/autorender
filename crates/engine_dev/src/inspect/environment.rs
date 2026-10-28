use super::*;

impl DevTools {
    pub(super) fn register_environment(&mut self, schemas: &Value) {
        self.register_resource::<Atmosphere>(
            InspectorDescriptor::generated(schemas, "Atmosphere")
                .with_default(json!(SceneAtmosphere::from(&Atmosphere::default()))),
            |atmosphere| json!(SceneAtmosphere::from(atmosphere)),
        )
        .expect("built-in descriptor is bounded");
    }
}
