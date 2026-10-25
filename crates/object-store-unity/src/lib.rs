//! [`object_store`] integration for [Unity Catalog] credential vending.
//!
//! Unity Catalog governs data in cloud storage and hands out short-lived,
//! least-privilege credentials for it. This crate turns those credentials into
//! regular object stores — so anything that takes an `Arc<dyn ObjectStore>`
//! (DataFusion, delta-rs, `parquet`, …) can read and write Unity Catalog
//! tables, volumes, and external locations — and keeps them fresh.
//!
//! # Quickstart
//!
//! ```no_run
//! use object_store::ObjectStoreExt;
//! use object_store_unity::{Operation, UnityStoreFactoryBuilder};
//!
//! # async fn run() -> object_store_unity::Result<()> {
//! // `DATABRICKS_HOST` plus any Databricks auth: a token, OAuth, OIDC, or a
//! // `~/.databrickscfg` profile.
//! let factory = UnityStoreFactoryBuilder::from_env().build()?;
//!
//! let volume = factory
//!     .for_url("uc:///Volumes/main/default/landing/", Operation::Read)
//!     .await?;
//! let bytes = volume.as_dyn().get(&"raw/file.json".into()).await?.bytes().await?;
//! # Ok(()) }
//! ```
//!
//! # Addressing securables
//!
//! [`UnityStoreFactory::for_url`] accepts any [`UnityReference`]:
//!
//! | URL                                                   | Credentials from               |
//! |-------------------------------------------------------|--------------------------------|
//! | `uc:///Volumes/<catalog>/<schema>/<volume>[/<path>]`  | `temporary-volume-credentials` |
//! | `uc:///Tables/<catalog>/<schema>/<table>`             | `temporary-table-credentials`  |
//! | `uc://<catalog>.<schema>.<table>` (deltalake's form)  | `temporary-table-credentials`  |
//! | `s3://`, `abfss://`, `gs://`, `r2://`, …              | `temporary-path-credentials`   |
//! | `file://`                                             | none — local storage           |
//!
//! The typed entry points [`for_table`](UnityStoreFactory::for_table),
//! [`for_volume`](UnityStoreFactory::for_volume) and
//! [`for_path`](UnityStoreFactory::for_path) take the per-endpoint operation
//! instead of the unified [`Operation`].
//!
//! # Credential lifecycle
//!
//! Every [`ScopedStore`] is bound to the securable and operation it was
//! created for. Its credential provider serves the vended credential until it
//! nears expiry, then re-vends exactly that — a refresh can never widen access.
//! Credentials vended without an expiry are kept. Implement
//! [`CredentialVendor`] to back a store with another vending endpoint
//! ([`UnityStoreFactory::for_vendor`]).
//!
//! # Many securables, one registry
//!
//! Object store registries such as DataFusion's resolve stores by
//! `scheme://host`, but each vended credential only covers one securable's
//! sub-path. Register a [`RoutingObjectStore`] under the bucket URL and route
//! each securable to its own store:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use object_store_unity::{RoutingObjectStore, TableOperation, UnityStoreFactory};
//! # async fn run(factory: UnityStoreFactory) -> object_store_unity::Result<()> {
//! let routing = Arc::new(RoutingObjectStore::new());
//! for table in ["main.sales.orders", "main.sales.customers"] {
//!     let store = factory.for_table(table, TableOperation::Read).await?;
//!     routing.register_scoped(&store);
//! }
//! // e.g. `runtime_env.register_object_store(&bucket_url, routing)`
//! # Ok(()) }
//! ```
//!
//! # Features
//!
//! `aws` (also Cloudflare R2), `azure`, and `gcp` enable the respective
//! stores; all are on by default. A credential for a disabled cloud is an
//! [`Error::FeatureDisabled`].
//!
//! [Unity Catalog]: https://www.unitycatalog.io/
#![warn(missing_docs)]

mod builder;
mod client;
mod credential;
mod error;
mod models;
mod reference;
mod routing;
mod store;

pub use builder::{UnityConfigKey, UnityStoreFactoryBuilder};
pub use client::UnityClient;
pub use credential::CredentialVendor;
pub use error::{Error, Result};
pub use models::{
    AwsTemporaryCredentials, AzureAad, AzureUserDelegationSas, GcpOauthToken, Operation,
    PathOperation, R2TemporaryCredentials, TableInfo, TableOperation, TemporaryCredentials,
    VolumeInfo, VolumeOperation,
};
pub use reference::UnityReference;
pub use routing::RoutingObjectStore;
pub use store::{ScopedStore, UnityStoreFactory};
