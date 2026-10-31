use engine_core::{SmokeFailure, SmokeSession, smoke_eq};

pub fn drawn(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(r#"{"assets":{"sheet":"sprites/sheet.png"},"entities":[{"sprite":{"sheet":"sheet","rect":[0,0,1,1],"tint":[1,1,1,1]}}]}"#);
    smoke_eq!(session.frame(), true, "sprites_frame");
    Ok(())
}
