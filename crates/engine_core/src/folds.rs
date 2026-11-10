use crate::settings;

settings! {
    #[section("fold_order")]
    #[serde(deny_unknown_fields)]
    pub struct FoldOrder {
        pub post: Vec<String> = Vec::new() => any,
        pub display: Vec<String> = Vec::new() => any,
    }
}
