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
//! // `UNITY_CATALOG_URL` plus `UNITY_*` auth (see below) — or, on Databricks,
//! // `DATABRICKS_HOST` plus any Databricks auth or `~/.databrickscfg` profile.
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
//! # Authentication
//!
//! Unity Catalog runs wherever it is deployed, behind whatever identity
//! provider the deployment uses. [`UnityStoreFactoryBuilder`] authenticates
//! with any of these ([`UnityAuthType`]). Most deployments need no
//! Unity-specific credentials: the workload's cloud identity is picked up from
//! the same `azure_*` / `google_*` keys and `AZURE_*` / `GOOGLE_*` variables
//! that `object_store` reads, once the Unity Catalog scope or audience is
//! known. `unity_*` keys select a different identity for Unity Catalog:
//!
//! | Deployment | Configuration |
//! |------------|---------------|
//! | A token you already have | `unity_token` |
//! | OSS Unity Catalog behind an IdP: exchange an ID token for a UC token | `unity_token_file` / `unity_token_env` (or an identity flow below) + `unity_token_exchange=true` |
//! | Any OAuth 2.0 IdP (Okta, Keycloak, Cognito, Entra, …) | `unity_client_id` + `unity_client_secret` + `unity_oauth_token_url`, optional `unity_oauth_scope` / `unity_oauth_audience` |
//! | The workload's Entra ID identity: service principal, workload identity, user-assigned managed identity, Azure CLI | `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET` / `AZURE_FEDERATED_TOKEN_FILE`, … ([`AzureIdentityKey`]) + `unity_oauth_scope` (known for Azure Databricks) |
//! | The workload's Google service account (ID token) | `GOOGLE_APPLICATION_CREDENTIALS`, … ([`GoogleIdentityKey`]) + `unity_oauth_audience` |
//! | A dedicated Entra ID service principal | `unity_client_id` + `unity_client_secret` + `unity_azure_tenant_id` + `unity_oauth_scope` |
//! | A system-assigned managed identity, the GCP metadata server | `unity_auth_type=azure-managed-identity` + `unity_oauth_scope`, `unity_auth_type=gcp-id-token` |
//! | Databricks | `databricks_*` options, `DATABRICKS_*` variables, or a `.databrickscfg` profile |
//! | Local development | `unity_allow_unauthenticated=true` |
//!
//! The most specific configuration wins: `unity_*` credentials, then
//! Databricks, then the well-known cloud identity; see [`UnityAuthType`].
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
//! stores; all are on by default. Without a configured `aws_region`, the
//! region of a vended S3 bucket is discovered from the bucket itself. A credential for a disabled cloud is an
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

pub use builder::{
    AuthConfigKey, AzureIdentityKey, GoogleIdentityKey, UnityAuthType, UnityConfigKey,
    UnityStoreFactoryBuilder,
};
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
