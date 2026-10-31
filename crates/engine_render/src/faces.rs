use std::any::Any;

pub const FACE_ROOT: &str = "face";

pub fn face_module(face: &str) -> String {
    format!("{FACE_ROOT}::{face}")
}

pub fn face_of(module: &str) -> Option<&str> {
    let face = module.strip_prefix(FACE_ROOT)?.strip_prefix("::")?;
    (!face.is_empty() && !face.contains("::")).then_some(face)
}

pub struct FaceCast<F: ?Sized + 'static> {
    pub get: for<'a> fn(&'a (dyn Any + 'static)) -> Option<&'a F>,
    pub get_mut: for<'a> fn(&'a mut (dyn Any + 'static)) -> Option<&'a mut F>,
}

pub fn face_name_of<F: ?Sized + 'static>() -> &'static str {
    engine_core::face_name(std::any::type_name::<F>())
}

#[macro_export]
macro_rules! offer_face {
    ($part:ty => dyn $face:path) => {
        ::engine_core::FaceOffer {
            face: ::std::any::TypeId::of::<dyn $face>,
            type_name: ::std::any::type_name::<dyn $face>,
            cast: || {
                ::std::boxed::Box::new($crate::faces::FaceCast::<dyn $face> {
                    get: |part| {
                        part.downcast_ref::<$part>()
                            .map(|part| part as &(dyn $face + 'static))
                    },
                    get_mut: |part| {
                        part.downcast_mut::<$part>()
                            .map(|part| part as &mut (dyn $face + 'static))
                    },
                })
            },
        }
    };
}
