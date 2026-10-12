//! Inspectors of the look: the atmosphere, each environment section and the
//! whole environment.

use super::*;

impl DevTools {
    /// The look: the atmosphere, every effect section and the
    /// whole environment.
    pub(super) fn register_environment(&mut self, schemas: &Value) {
        // The environment: each resource serializes as its
        // scene section, so values copy into controls and v5 scenes as is.
        self.register_resource::<Atmosphere>(
            InspectorDescriptor::generated(schemas, "Atmosphere")
                .with_default(json!(SceneAtmosphere::from(&Atmosphere::default()))),
            |atmosphere| json!(SceneAtmosphere::from(atmosphere)),
        )
        .expect("built-in descriptor is bounded");
        // Every effect section of a scene environment as its resource, in
        // the section's shape.
        struct Sections<'t> {
            tools: &'t mut DevTools,
            schemas: &'t Value,
        }
        impl SectionVisitor for Sections<'_> {
            fn section<T: EnvironmentSection>(&mut self, _: &'static str, _: Option<T>) {
                let schema = self
                    .schemas
                    .get(T::NAME)
                    .cloned()
                    .unwrap_or_else(|| panic!("schemas generates no schema named {}", T::NAME));
                // The settings type's doc comment, whole in the schema.
                let description = summary(
                    schema["description"]
                        .as_str()
                        .expect("every environment section's settings type documents itself"),
                );
                self.tools
                    .register_resource::<T>(
                        InspectorDescriptor::new(T::NAME, description, schema)
                            .with_default(json!(T::default())),
                        |settings| json!(settings),
                    )
                    .expect("built-in descriptor is bounded");
                if let Some(feature) = T::FEATURE {
                    self.tools.features.insert(T::NAME.to_owned(), feature);
                }
            }
        }
        SceneEnvironment::default().visit(&mut Sections {
            tools: self,
            schemas,
        });
        // The whole environment's schema naming each section's inspector,
        // which carries its fields, so it stays within bounds as sections
        // grow.
        self.insert(Inspector {
            descriptor: InspectorDescriptor::new("Environment", "The whole current look as a scene environment: paste it into a scene to persist it, or send sections to the environment control. Missing effect resources read as off. Each section's inspector describes its fields.", sections_schema()),
            kind: InspectorKind::Resource,
            has: Box::new(|_, _| true),
            read: Box::new(|world, _, _| Some(json!(SceneEnvironment::current(world)))),
        }).expect("built-in descriptor is bounded");
    }
}

/// The scene environment's schema: each section an object (or null) whose
/// fields the inspector it names describes.
fn sections_schema() -> Value {
    struct Names(Vec<(&'static str, &'static str)>);
    impl SectionVisitor for Names {
        fn section<T: EnvironmentSection>(&mut self, name: &'static str, _: Option<T>) {
            self.0.push((name, T::NAME));
        }
    }
    let mut names = Names(vec![("atmosphere", "Atmosphere")]);
    SceneEnvironment::default().visit(&mut names);
    let properties: serde_json::Map<String, Value> = names
        .0
        .into_iter()
        .map(|(field, inspector)| {
            (
                field.to_owned(),
                json!({
                    "type": ["object", "null"],
                    "description": format!("The {inspector} inspector describes its fields."),
                }),
            )
        })
        .collect();
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SceneEnvironment",
        "type": "object",
        "additionalProperties": false,
        "properties": properties,
    })
}
