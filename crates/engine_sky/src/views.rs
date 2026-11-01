use serde::Serialize;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StaticTablesView {
    pub tables: Vec<StaticTableRow>,
    pub failed_jobs: u64,
    pub latest_job_failure: Option<String>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StaticTableRow {
    pub kind: String,
    pub parts: u64,
    pub arrived: u64,
    pub showing: bool,
}
