//! Refreshing [`object_store`] credential providers backed by Unity Catalog
//! credential vending.
//!
//! A [`UnityCredentialProvider`] is bound to one [`CredentialVendor`] — one
//! securable and one operation. It serves the credential vended when the store
//! was built until it nears expiry, then re-vends through the same vendor, so a
//! refresh can never widen the privileges the store was created with.
//! Concurrent callers share a single in-flight refresh.

use std::fmt;
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use url::Url;
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
use {
    chrono::{DateTime, Utc},
    object_store::CredentialProvider,
    olai_http::{TemporaryToken, TokenCache},
};

use crate::client::UnityClient;
use crate::error::Result;
use crate::models::{PathOperation, TableOperation, TemporaryCredentials, VolumeOperation};

/// A source of fresh Unity Catalog credentials for one securable.
///
/// Stores built by this crate call [`vend`](Self::vend) whenever their cached
/// credential nears expiry. Implement it to plug in a vending endpoint this
/// crate does not cover, such as the Unity Catalog Delta API's table
/// credentials (see [`TemporaryCredentials::from_storage_config`]).
#[async_trait]
pub trait CredentialVendor: fmt::Debug + Send + Sync + 'static {
    /// Vend a fresh credential.
    async fn vend(&self) -> Result<TemporaryCredentials>;
}

/// Vends table credentials (`temporary-table-credentials`).
#[derive(Debug)]
pub(crate) struct TableVendor {
    pub client: UnityClient,
    pub table_id: String,
    pub operation: TableOperation,
}

#[async_trait]
impl CredentialVendor for TableVendor {
    async fn vend(&self) -> Result<TemporaryCredentials> {
        self.client
            .temporary_table_credentials(&self.table_id, self.operation)
            .await
    }
}

/// Vends volume credentials (`temporary-volume-credentials`).
#[derive(Debug)]
pub(crate) struct VolumeVendor {
    pub client: UnityClient,
    pub volume_id: String,
    pub operation: VolumeOperation,
}

#[async_trait]
impl CredentialVendor for VolumeVendor {
    async fn vend(&self) -> Result<TemporaryCredentials> {
        self.client
            .temporary_volume_credentials(&self.volume_id, self.operation)
            .await
    }
}

/// Vends path credentials (`temporary-path-credentials`).
#[derive(Debug)]
pub(crate) struct PathVendor {
    pub client: UnityClient,
    pub url: Url,
    pub operation: PathOperation,
    pub dry_run: bool,
}

#[async_trait]
impl CredentialVendor for PathVendor {
    async fn vend(&self) -> Result<TemporaryCredentials> {
        self.client
            .temporary_path_credentials(&self.url, self.operation, self.dry_run)
            .await
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
/// Converts a vended credential into one cloud's `object_store` credential.
pub(crate) type Convert<T> = fn(&TemporaryCredentials) -> Result<T>;

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
/// An [`object_store::CredentialProvider`] that re-vends through its
/// [`CredentialVendor`] when the cached credential nears expiry.
pub(crate) struct UnityCredentialProvider<T> {
    vendor: Arc<dyn CredentialVendor>,
    cache: TokenCache<Arc<T>>,
    convert: Convert<T>,
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
impl<T: Send + Sync> UnityCredentialProvider<T> {
    /// Create a provider seeded with an already-vended credential, so the first
    /// request does not vend again.
    pub(crate) async fn new(
        vendor: Arc<dyn CredentialVendor>,
        seed: &TemporaryCredentials,
        convert: Convert<T>,
    ) -> Result<Self> {
        let cache = TokenCache::default();
        cache
            .get_or_insert_with(|| async { to_token(seed, convert) })
            .await?;
        Ok(Self {
            vendor,
            cache,
            convert,
        })
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[async_trait]
impl<T: fmt::Debug + Send + Sync + 'static> CredentialProvider for UnityCredentialProvider<T> {
    type Credential = T;

    async fn get_credential(&self) -> object_store::Result<Arc<T>> {
        let convert = self.convert;
        let credential = self
            .cache
            .get_or_insert_with(|| async move { to_token(&self.vendor.vend().await?, convert) })
            .await?;
        Ok(credential)
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
impl<T> fmt::Debug for UnityCredentialProvider<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnityCredentialProvider")
            .field("vendor", &self.vendor)
            .finish_non_exhaustive()
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
fn to_token<T>(
    credentials: &TemporaryCredentials,
    convert: Convert<T>,
) -> Result<TemporaryToken<Arc<T>>> {
    Ok(TemporaryToken {
        token: Arc::new(convert(credentials)?),
        expiry: expiry(credentials),
    })
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
/// The cache expiry for a vended credential.
///
/// A missing or non-positive `expiration_time` means the server vended no
/// expiry (OSS Unity Catalog sends `null`): keep the credential. Reading `0` as
/// the Unix epoch would mark it already expired and re-vend on every request.
fn expiry(credentials: &TemporaryCredentials) -> Option<Instant> {
    let millis = credentials.expiration_time.filter(|ms| *ms > 0)?;
    let remaining = DateTime::from_timestamp_millis(millis)
        .and_then(|at| (at - Utc::now()).to_std().ok())
        .unwrap_or_default();
    Some(Instant::now() + remaining)
}

/// Convert a vended credential into an [`AwsCredential`] (AWS or R2).
///
/// [`AwsCredential`]: object_store::aws::AwsCredential
#[cfg(feature = "aws")]
pub(crate) fn aws_credential(
    credentials: &TemporaryCredentials,
) -> Result<object_store::aws::AwsCredential> {
    use crate::error::Error;

    // An empty session token must not be signed as `x-amz-security-token: `.
    let non_empty = |token: &Option<String>| token.clone().filter(|t| !t.is_empty());
    if let Some(aws) = &credentials.aws_temp_credentials {
        return Ok(object_store::aws::AwsCredential {
            key_id: aws.access_key_id.clone(),
            secret_key: aws.secret_access_key.clone(),
            token: non_empty(&aws.session_token),
        });
    }
    if let Some(r2) = &credentials.r2_temp_credentials {
        return Ok(object_store::aws::AwsCredential {
            key_id: r2.access_key_id.clone(),
            secret_key: r2.secret_access_key.clone(),
            token: non_empty(&r2.session_token),
        });
    }
    Err(Error::CredentialMismatch(
        "expected an AWS or R2 credential".to_string(),
    ))
}

/// Convert a vended credential into an [`AzureCredential`] (SAS or AAD).
///
/// [`AzureCredential`]: object_store::azure::AzureCredential
#[cfg(feature = "azure")]
pub(crate) fn azure_credential(
    credentials: &TemporaryCredentials,
) -> Result<object_store::azure::AzureCredential> {
    use crate::error::Error;
    use object_store::azure::AzureCredential;

    if let Some(sas) = &credentials.azure_user_delegation_sas {
        return Ok(AzureCredential::SASToken(split_sas(&sas.sas_token)?));
    }
    if let Some(aad) = &credentials.azure_aad {
        return Ok(AzureCredential::BearerToken(aad.aad_token.clone()));
    }
    Err(Error::CredentialMismatch(
        "expected an Azure credential".to_string(),
    ))
}

/// Split a SAS query string (`k1=v1&k2=v2&…`) into **decoded** pairs.
///
/// Mirrors object_store's own handling of `AzureConfigKey::SasKey`. Unity
/// Catalog vends the token URL-encoded (`sig=…%2B…%3D`), and object_store
/// re-encodes [`SASToken`] pairs when it appends them to each request — so they
/// must be decoded here, or the signature goes out double-encoded and Azure
/// rejects it.
///
/// [`SASToken`]: object_store::azure::AzureCredential::SASToken
#[cfg(feature = "azure")]
pub(crate) fn split_sas(sas: &str) -> Result<Vec<(String, String)>> {
    use crate::error::Error;

    let sas = percent_encoding::percent_decode_str(sas)
        .decode_utf8()
        .map_err(|e| Error::CredentialMismatch(format!("invalid SAS token encoding: {e}")))?;
    sas.trim_start_matches('?')
        .split('&')
        .filter(|pair| !pair.trim().is_empty())
        .map(|pair| {
            pair.trim()
                .split_once('=')
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .ok_or_else(|| Error::CredentialMismatch("malformed SAS token component".into()))
        })
        .collect()
}

/// Convert a vended credential into a [`GcpCredential`].
///
/// [`GcpCredential`]: object_store::gcp::GcpCredential
#[cfg(feature = "gcp")]
pub(crate) fn gcp_credential(
    credentials: &TemporaryCredentials,
) -> Result<object_store::gcp::GcpCredential> {
    use crate::error::Error;

    credentials
        .gcp_oauth_token
        .as_ref()
        .map(|gcp| object_store::gcp::GcpCredential {
            bearer: gcp.oauth_token.clone(),
        })
        .ok_or_else(|| Error::CredentialMismatch("expected a GCP credential".to_string()))
}

#[cfg(all(test, feature = "aws", feature = "azure", feature = "gcp"))]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::models::{AwsTemporaryCredentials, AzureUserDelegationSas};

    /// A vendor that counts how often it is asked and always returns `next`.
    #[derive(Debug)]
    struct CountingVendor {
        calls: AtomicUsize,
        next: TemporaryCredentials,
    }

    #[async_trait]
    impl CredentialVendor for CountingVendor {
        async fn vend(&self) -> Result<TemporaryCredentials> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.next.clone())
        }
    }

    fn aws(key_id: &str, expiration_time: Option<i64>) -> TemporaryCredentials {
        TemporaryCredentials {
            expiration_time,
            aws_temp_credentials: Some(AwsTemporaryCredentials {
                access_key_id: key_id.to_string(),
                secret_access_key: "sk".to_string(),
                session_token: Some(String::new()),
                access_point: None,
            }),
            ..Default::default()
        }
    }

    fn now_millis() -> i64 {
        Utc::now().timestamp_millis()
    }

    async fn provider(
        seed: TemporaryCredentials,
        next: TemporaryCredentials,
    ) -> (
        Arc<CountingVendor>,
        UnityCredentialProvider<object_store::aws::AwsCredential>,
    ) {
        let vendor = Arc::new(CountingVendor {
            calls: AtomicUsize::new(0),
            next,
        });
        let provider = UnityCredentialProvider::new(vendor.clone(), &seed, aws_credential)
            .await
            .unwrap();
        (vendor, provider)
    }

    #[tokio::test]
    async fn missing_or_zero_expiry_never_revends() {
        for expiration_time in [None, Some(0)] {
            let (vendor, provider) =
                provider(aws("seed", expiration_time), aws("fresh", None)).await;
            for _ in 0..3 {
                assert_eq!(provider.get_credential().await.unwrap().key_id, "seed");
            }
            assert_eq!(vendor.calls.load(Ordering::SeqCst), 0);
        }
    }

    #[tokio::test]
    async fn valid_credential_is_served_from_cache() {
        let (vendor, provider) = provider(
            aws("seed", Some(now_millis() + 3_600_000)),
            aws("fresh", None),
        )
        .await;
        provider.get_credential().await.unwrap();
        assert_eq!(vendor.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn expired_credential_revends_once() {
        let (vendor, provider) = provider(
            aws("seed", Some(now_millis() - 1_000)),
            aws("fresh", Some(now_millis() + 3_600_000)),
        )
        .await;
        for _ in 0..2 {
            assert_eq!(provider.get_credential().await.unwrap().key_id, "fresh");
        }
        assert_eq!(vendor.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn empty_session_token_is_dropped() {
        assert!(aws_credential(&aws("ak", None)).unwrap().token.is_none());
    }

    #[test]
    fn mismatched_cloud_is_an_error() {
        let azure = TemporaryCredentials {
            azure_user_delegation_sas: Some(AzureUserDelegationSas {
                sas_token: "sv=1".to_string(),
            }),
            ..Default::default()
        };
        assert!(aws_credential(&azure).is_err());
        assert!(gcp_credential(&azure).is_err());
        assert!(azure_credential(&aws("ak", None)).is_err());
    }

    #[test]
    fn split_sas_decodes_pairs() {
        let pairs = split_sas("?sv=2021-08-06&sp=rl&&sig=ab%2Bcd%2Fef%3D").unwrap();
        assert_eq!(
            pairs,
            vec![
                ("sv".to_string(), "2021-08-06".to_string()),
                ("sp".to_string(), "rl".to_string()),
                ("sig".to_string(), "ab+cd/ef=".to_string()),
            ]
        );
        assert!(split_sas("sv=2021-08-06&garbage").is_err());
    }
}
