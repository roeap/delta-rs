//! Wire models for the Unity Catalog credential-vending endpoints.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Access requested when vending credentials for a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum TableOperation {
    /// Read-only access (`READ`).
    #[serde(rename = "READ")]
    Read,
    /// Read and write access (`READ_WRITE`).
    #[serde(rename = "READ_WRITE")]
    ReadWrite,
}

/// Access requested when vending credentials for a volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum VolumeOperation {
    /// Read-only access (`READ_VOLUME`).
    #[serde(rename = "READ_VOLUME")]
    Read,
    /// Read and write access (`WRITE_VOLUME`).
    #[serde(rename = "WRITE_VOLUME")]
    ReadWrite,
}

/// Access requested when vending credentials for a raw storage path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum PathOperation {
    /// Read-only access (`PATH_READ`).
    #[serde(rename = "PATH_READ")]
    Read,
    /// Read and write access (`PATH_READ_WRITE`).
    #[serde(rename = "PATH_READ_WRITE")]
    ReadWrite,
    /// Access to create a new external table (`PATH_CREATE_TABLE`).
    #[serde(rename = "PATH_CREATE_TABLE")]
    CreateTable,
}

/// A securable-agnostic access level, mapped onto the operation of each
/// vending endpoint.
///
/// | `Operation` | Table        | Volume         | Path              |
/// |-------------|--------------|----------------|-------------------|
/// | `Read`      | `READ`       | `READ_VOLUME`  | `PATH_READ`       |
/// | `ReadWrite` | `READ_WRITE` | `WRITE_VOLUME` | `PATH_READ_WRITE` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Operation {
    /// Read-only access.
    #[default]
    Read,
    /// Read and write access.
    ReadWrite,
}

impl From<Operation> for TableOperation {
    fn from(op: Operation) -> Self {
        match op {
            Operation::Read => TableOperation::Read,
            Operation::ReadWrite => TableOperation::ReadWrite,
        }
    }
}

impl From<Operation> for VolumeOperation {
    fn from(op: Operation) -> Self {
        match op {
            Operation::Read => VolumeOperation::Read,
            Operation::ReadWrite => VolumeOperation::ReadWrite,
        }
    }
}

impl From<Operation> for PathOperation {
    fn from(op: Operation) -> Self {
        match op {
            Operation::Read => PathOperation::Read,
            Operation::ReadWrite => PathOperation::ReadWrite,
        }
    }
}

/// Short-lived cloud credentials vended by Unity Catalog.
///
/// Exactly one cloud credential is expected to be set. `Debug` redacts every
/// secret.
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct TemporaryCredentials {
    /// Expiry in epoch milliseconds. `None` (or a non-positive value) means the
    /// server did not vend an expiry — OSS Unity Catalog returns `null`.
    pub expiration_time: Option<i64>,
    /// The storage location the credential is scoped to.
    pub url: Option<String>,
    /// AWS STS session credentials.
    pub aws_temp_credentials: Option<AwsTemporaryCredentials>,
    /// An Azure user-delegation SAS token.
    pub azure_user_delegation_sas: Option<AzureUserDelegationSas>,
    /// An Azure AD bearer token.
    pub azure_aad: Option<AzureAad>,
    /// A GCP OAuth bearer token.
    pub gcp_oauth_token: Option<GcpOauthToken>,
    /// Cloudflare R2 session credentials.
    pub r2_temp_credentials: Option<R2TemporaryCredentials>,
}

impl TemporaryCredentials {
    /// Build credentials from a generic storage-config map, as returned by the
    /// Unity Catalog Delta (and Iceberg REST) APIs.
    ///
    /// Recognised keys: `s3.access-key-id`, `s3.secret-access-key`,
    /// `s3.session-token`, `azure.sas-token`, and `gcs.oauth-token`. Unknown keys
    /// are ignored. `prefix` is the storage location the credential applies to.
    pub fn from_storage_config(
        prefix: impl Into<String>,
        expiration_time_ms: Option<i64>,
        config: &HashMap<String, String>,
    ) -> Self {
        let get = |key: &str| config.get(key).cloned();
        let aws_temp_credentials = match (get("s3.access-key-id"), get("s3.secret-access-key")) {
            (Some(access_key_id), Some(secret_access_key)) => Some(AwsTemporaryCredentials {
                access_key_id,
                secret_access_key,
                session_token: get("s3.session-token"),
                access_point: None,
            }),
            _ => None,
        };
        Self {
            expiration_time: expiration_time_ms,
            url: Some(prefix.into()),
            aws_temp_credentials,
            azure_user_delegation_sas: get("azure.sas-token")
                .map(|sas_token| AzureUserDelegationSas { sas_token }),
            gcp_oauth_token: get("gcs.oauth-token")
                .map(|oauth_token| GcpOauthToken { oauth_token }),
            ..Default::default()
        }
    }

    /// The cloud this credential was vended for, if any.
    pub(crate) fn kind(&self) -> Option<CredentialKind> {
        if self.azure_user_delegation_sas.is_some() || self.azure_aad.is_some() {
            Some(CredentialKind::Azure)
        } else if self.aws_temp_credentials.is_some() || self.r2_temp_credentials.is_some() {
            Some(CredentialKind::Aws)
        } else if self.gcp_oauth_token.is_some() {
            Some(CredentialKind::Gcp)
        } else {
            None
        }
    }
}

/// The object store family a credential applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CredentialKind {
    Aws,
    Azure,
    Gcp,
}

impl fmt::Debug for TemporaryCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemporaryCredentials")
            .field("expiration_time", &self.expiration_time)
            .field("url", &self.url)
            .field("kind", &self.kind())
            .finish_non_exhaustive()
    }
}

/// AWS STS session credentials.
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct AwsTemporaryCredentials {
    /// The access key id.
    pub access_key_id: String,
    /// The secret access key.
    pub secret_access_key: String,
    /// The session token, when the credential is session-scoped.
    pub session_token: Option<String>,
    /// An S3 access-point ARN the credential is only valid against.
    pub access_point: Option<String>,
}

/// Cloudflare R2 session credentials.
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct R2TemporaryCredentials {
    /// The access key id.
    pub access_key_id: String,
    /// The secret access key.
    pub secret_access_key: String,
    /// The session token.
    pub session_token: Option<String>,
}

/// An Azure user-delegation SAS token (`k1=v1&k2=v2&…`, URL-encoded).
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct AzureUserDelegationSas {
    /// The SAS query string.
    pub sas_token: String,
}

/// An Azure AD bearer token.
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct AzureAad {
    /// The bearer token.
    pub aad_token: String,
}

/// A GCP OAuth bearer token.
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct GcpOauthToken {
    /// The bearer token.
    pub oauth_token: String,
}

macro_rules! redacted_debug {
    ($($ty:ty),*) => {
        $(impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($ty)).finish_non_exhaustive()
            }
        })*
    };
}

redacted_debug!(
    AwsTemporaryCredentials,
    R2TemporaryCredentials,
    AzureUserDelegationSas,
    AzureAad,
    GcpOauthToken
);

/// The subset of a Unity Catalog table needed to vend credentials.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct TableInfo {
    /// The table's unique id, used to vend credentials.
    pub table_id: String,
    /// The table's storage root, when it has one.
    pub storage_location: Option<String>,
}

/// The subset of a Unity Catalog volume needed to vend credentials.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct VolumeInfo {
    /// The volume's unique id, used to vend credentials.
    pub volume_id: String,
    /// The volume's storage root.
    pub storage_location: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TableCredentialsRequest<'a> {
    pub table_id: &'a str,
    pub operation: TableOperation,
}

#[derive(Serialize)]
pub(crate) struct VolumeCredentialsRequest<'a> {
    pub volume_id: &'a str,
    pub operation: VolumeOperation,
}

#[derive(Serialize)]
pub(crate) struct PathCredentialsRequest<'a> {
    pub url: &'a str,
    pub operation: PathOperation,
    pub dry_run: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_null_expiry_and_unknown_fields() {
        let json = r#"{
            "expiration_time": null,
            "url": "s3://bucket/table/",
            "aws_temp_credentials": {
                "access_key_id": "ak", "secret_access_key": "sk", "session_token": "st"
            },
            "some_future_field": 1
        }"#;
        let creds: TemporaryCredentials = serde_json::from_str(json).unwrap();
        assert_eq!(creds.expiration_time, None);
        assert_eq!(creds.kind(), Some(CredentialKind::Aws));
        assert_eq!(creds.url.as_deref(), Some("s3://bucket/table/"));
    }

    #[test]
    fn debug_redacts_secrets() {
        let creds = TemporaryCredentials {
            azure_user_delegation_sas: Some(AzureUserDelegationSas {
                sas_token: "sig=super-secret".to_string(),
            }),
            ..Default::default()
        };
        let debug = format!("{creds:?}");
        assert!(!debug.contains("super-secret"), "secret leaked: {debug}");
        assert!(debug.contains("Azure"));
    }

    #[test]
    fn from_storage_config_maps_known_keys() {
        let config = HashMap::from([
            ("s3.access-key-id".to_string(), "ak".to_string()),
            ("s3.secret-access-key".to_string(), "sk".to_string()),
            ("s3.session-token".to_string(), "st".to_string()),
            ("unrelated".to_string(), "x".to_string()),
        ]);
        let creds = TemporaryCredentials::from_storage_config("s3://b/t/", Some(42), &config);
        let aws = creds.aws_temp_credentials.as_ref().unwrap();
        assert_eq!(aws.access_key_id, "ak");
        assert_eq!(aws.session_token.as_deref(), Some("st"));
        assert_eq!(creds.expiration_time, Some(42));
        assert_eq!(creds.url.as_deref(), Some("s3://b/t/"));

        let gcs = HashMap::from([("gcs.oauth-token".to_string(), "tok".to_string())]);
        let creds = TemporaryCredentials::from_storage_config("gs://b/t/", None, &gcs);
        assert_eq!(creds.kind(), Some(CredentialKind::Gcp));
    }

    #[test]
    fn operations_serialize_to_wire_names() {
        let json = |v: &str| format!("\"{v}\"");
        assert_eq!(
            serde_json::to_string(&TableOperation::ReadWrite).unwrap(),
            json("READ_WRITE")
        );
        assert_eq!(
            serde_json::to_string(&VolumeOperation::ReadWrite).unwrap(),
            json("WRITE_VOLUME")
        );
        assert_eq!(
            serde_json::to_string(&PathOperation::CreateTable).unwrap(),
            json("PATH_CREATE_TABLE")
        );
    }
}
