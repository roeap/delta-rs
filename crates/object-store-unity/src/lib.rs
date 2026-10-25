//! [`object_store`] integration for [Unity Catalog] credential vending.
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
