//! [`object_store`] integration for [Unity Catalog] credential vending.
//!
//! [Unity Catalog]: https://www.unitycatalog.io/
#![warn(missing_docs)]

mod client;
mod error;
mod models;
mod reference;

pub use client::UnityClient;
pub use error::{Error, Result};
pub use models::{
    AwsTemporaryCredentials, AzureAad, AzureUserDelegationSas, GcpOauthToken, Operation,
    PathOperation, R2TemporaryCredentials, TableInfo, TableOperation, TemporaryCredentials,
    VolumeInfo, VolumeOperation,
};
pub use reference::UnityReference;
