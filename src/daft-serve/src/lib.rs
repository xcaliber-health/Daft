//! Client-server query execution over a streaming columnar RPC protocol.
//!
//! A serving process hosts the local streaming engine behind a single-port
//! endpoint. Thin clients ship complete queries — either a serialized
//! unoptimized logical plan or a textual query — in one round trip, and
//! receive results as a stream of columnar batches. Plans are optimized on
//! the server so scan planning, pruning, and all data access run next to the
//! data; clients never need direct storage access.
//!
//! Module map:
//! - [`wire`] — request envelope and message types
//! - [`codec`] — columnar batch encoding between engine and wire form
//! - [`auth`] — bearer-token authentication
//! - [`admission`] — concurrent-query admission control
//! - [`registry`] — running-query registry for cross-connection cancel
//! - [`error`] — typed errors and their transport mapping
//! - `server` / `execute` — the serving endpoint and query driver
//! - `client` — the connecting side, decoding result streams

pub mod admission;
pub mod auth;
pub mod codec;
pub mod error;
pub mod registry;
pub mod wire;

#[cfg(feature = "python")]
pub mod client;
#[cfg(feature = "python")]
pub mod execute;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub mod server;

#[cfg(feature = "python")]
pub use python::register_modules;
