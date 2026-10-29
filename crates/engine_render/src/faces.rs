use std::any::{Any, TypeId};

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

#[derive(Clone, Copy)]
pub struct FaceOffer {
    pub face: fn() -> TypeId,
    pub type_name: fn() -> &'static str,
    pub cast: fn() -> Box<dyn Any + Send + Sync>,
}

impl FaceOffer {
    pub fn name(&self) -> &'static str {
        face_name((self.type_name)())
    }
}

pub fn face_name(type_name: &'static str) -> &'static str {
    let plain = type_name.strip_prefix("dyn ").unwrap_or(type_name);
    let plain = plain.split('<').next().unwrap_or(plain);
    plain.rsplit("::").next().unwrap_or(plain)
}

pub fn face_name_of<F: ?Sized + 'static>() -> &'static str {
    face_name(std::any::type_name::<F>())
}

#[macro_export]
macro_rules! offer_face {
    ($part:ty => dyn $face:path) => {
        $crate::faces::FaceOffer {
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
