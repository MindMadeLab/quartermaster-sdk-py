//! quartermaster-code-runner SDK for Rust runtime.
//!
//! # Example
//!
//! ```rust
//! use sdk::set_metadata;
//! use serde::Serialize;
//!
//! #[derive(Serialize)]
//! struct Result {
//!     status: String,
//!     count: i32,
//! }
//!
//! fn main() {
//!     let result = Result { status: "success".into(), count: 42 };
//!     set_metadata(&result).unwrap();
//! }
//! ```

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::env;
use std::fs;
use std::io;

const METADATA_FILE: &str = "/metadata/.quartermaster_metadata.json";

/// Set the result metadata to be returned to the backend.
///
/// This separates structured results from stdout/stderr logs.
///
/// # Example
///
/// ```rust
/// use sdk::set_metadata;
/// use serde_json::json;
///
/// 
/// set_metadata(&json!({"status": "success", "count": 42})).unwrap();
/// ```
pub fn set_metadata<T: Serialize>(data: &T) -> io::Result<()> {
    let json =
        serde_json::to_string(data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(METADATA_FILE, json)
}

/// Set metadata from a raw JSON string.
///
/// Use this if you already have JSON as a string.
///
/// # Example
///
/// ```rust
/// sdk::set_metadata_raw(r#"{"status": "success"}"#).unwrap();
/// ```
pub fn set_metadata_raw(json: &str) -> io::Result<()> {
    fs::write(METADATA_FILE, json)
}

/// Get previously set metadata, deserializing into the specified type.
///
/// Returns `None` if no metadata has been set.
pub fn get_metadata<T: DeserializeOwned>() -> io::Result<Option<T>> {
    match fs::read_to_string(METADATA_FILE) {
        Ok(content) => {
            let data = serde_json::from_str(&content)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Some(data))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Get previously set metadata as a raw JSON string.
///
/// Returns `None` if no metadata has been set.
pub fn get_metadata_raw() -> io::Result<Option<String>> {
    match fs::read_to_string(METADATA_FILE) {
        Ok(content) => Ok(Some(content)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

fn webdav_url(path: &str) -> Result<String, io::Error> {
    let base = env::var("QM_WEBDAV_URL").map_err(|_| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "load_file/save_file is only available during flow execution. \
             For test runs, use mounted environments instead.",
        )
    })?;
    Ok(format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    ))
}

/// Load a file from the flow's environment.
///
/// Only available during flow execution, not test runs.
///
/// # Example
///
/// ```rust,no_run
/// let content = sdk::load_file("data/config.json").unwrap();
/// println!("{}", content);
/// ```
pub fn load_file(path: &str) -> io::Result<String> {
    let url = webdav_url(path)?;
    let resp = ureq::get(&url).call().map_err(|e| match &e {
        ureq::Error::Status(404, _) => io::Error::new(
            io::ErrorKind::NotFound,
            format!("File not found: {}", path),
        ),
        _ => io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to load file: {}", e),
        ),
    })?;
    resp.into_string()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

/// Save a file to the flow's environment.
///
/// Only available during flow execution, not test runs.
///
/// # Example
///
/// ```rust,no_run
/// sdk::save_file("output/result.txt", "Hello, world!").unwrap();
/// ```
pub fn save_file(path: &str, content: &str) -> io::Result<()> {
    let url = webdav_url(path)?;
    ureq::put(&url)
        .set("Content-Type", "application/octet-stream")
        .send_string(content)
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to save file: {}", e),
            )
        })?;
    Ok(())
}
