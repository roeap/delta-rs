//! Configuration for [`UnityStoreFactory`].

use std::fmt;
use std::str::FromStr;

use std::collections::HashMap;

use olai_http::bearer::{
    AzureManagedIdentity, BearerAuth, BearerCredentialProvider, ClientCredentials,
    GcpServiceAccount, TokenExchange,
};
use olai_http::databricks::{DatabricksBuilder, DatabricksConfigKey};
use olai_http::{ClientConfigKey, ClientOptions, CloudClient, RetryConfig};
use tokio::runtime::Handle;
use url::Url;

use crate::client::UnityClient;
use crate::error::{Error, Result};
use crate::store::UnityStoreFactory;

/// The path of the Unity Catalog REST API below the server root.
const UNITY_CATALOG_API_PATH: &str = "api/2.1/unity-catalog/";

/// The OSS Unity Catalog token-exchange endpoint below the server root.
const UNITY_TOKEN_EXCHANGE_PATH: &str = "api/1.0/unity-control/auth/tokens";

/// The Azure AD scope of Azure Databricks.
const AZURE_DATABRICKS_SCOPE: &str = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default";

/// Domains of Azure Databricks workspaces.
const AZURE_DATABRICKS_DOMAINS: &[&str] = &[
    "azuredatabricks.net",
    "databricks.azure.cn",
    "databricks.azure.us",
];

/// Domains of Databricks workspaces on AWS and GCP.
const DATABRICKS_DOMAINS: &[&str] = &[
    "cloud.databricks.com",
    "cloud.databricks.us",
    "cloud.databricks.mil",
    "gcp.databricks.com",
];

/// Configuration keys for [`UnityStoreFactoryBuilder`].
///
/// Parse one from its string form with [`FromStr`]. Keys are matched exactly
/// (lowercase); environment variables are matched by their lowercased name, so
/// `UNITY_TOKEN` sets [`UnityConfigKey::Token`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnityConfigKey {
    /// The Unity Catalog REST API base URL, e.g.
    /// `https://uc.example.com/api/2.1/unity-catalog/`.
    ///
    /// Supported keys: `unity_catalog_url`, `unity_url`. Defaults to
    /// `<host>/api/2.1/unity-catalog/` when a Databricks host is configured.
    Url,

    /// Region of vended AWS locations.
    ///
    /// Supported keys: `aws_region`, `unity_aws_region`. When unset, the
    /// region is discovered from the bucket, then falls back to the
    /// `AWS_REGION` environment variable.
    AwsRegion,

    /// Send requests unauthenticated when no credentials are configured.
    ///
    /// Only meant for local development against an unauthenticated server.
    ///
    /// Supported keys: `unity_allow_unauthenticated`.
    AllowUnauthenticated,

    /// How to authenticate; see [`UnityAuthType`]. Detected when unset.
    ///
    /// Supported keys: `unity_auth_type`.
    AuthType,

    /// A bearer token for Unity Catalog.
    ///
    /// Supported keys: `unity_token`, `unity_access_token`.
    Token,

    /// The OAuth client id, for client credentials or an Azure service
    /// principal; or the client id of a user-assigned Azure managed identity.
    ///
    /// Supported keys: `unity_client_id`.
    ClientId,

    /// The OAuth client secret.
    ///
    /// Supported keys: `unity_client_secret`.
    ClientSecret,

    /// The OAuth token endpoint for client credentials.
    ///
    /// Supported keys: `unity_oauth_token_url`.
    OAuthTokenUrl,

    /// The OAuth scope to request. Azure flows need one, e.g.
    /// `api://my-catalog/.default`.
    ///
    /// Supported keys: `unity_oauth_scope`.
    OAuthScope,

    /// The OAuth audience to request. GCP ID tokens default to the origin of
    /// the Unity Catalog URL.
    ///
    /// Supported keys: `unity_oauth_audience`.
    OAuthAudience,

    /// The Azure AD tenant of a service principal or workload identity.
    ///
    /// Supported keys: `unity_azure_tenant_id`.
    AzureTenantId,

    /// The Azure AD authority host; defaults to the public cloud.
    ///
    /// Supported keys: `unity_azure_authority_host`.
    AzureAuthorityHost,

    /// A file holding a token, re-read on every refresh — e.g. a projected
    /// Kubernetes service-account token, or the federated token of Azure
    /// workload identity.
    ///
    /// Supported keys: `unity_token_file`.
    TokenFile,

    /// An environment variable holding a token, re-read on every refresh.
    ///
    /// Supported keys: `unity_token_env`.
    TokenEnv,

    /// Exchange the configured token for a Unity Catalog token (RFC 8693), as
    /// OSS Unity Catalog expects for identity-provider tokens.
    ///
    /// `true` uses the server's `/api/1.0/unity-control/auth/tokens`
    /// endpoint; a URL selects another endpoint.
    ///
    /// Supported keys: `unity_token_exchange`.
    TokenExchange,

    /// The RFC 8693 type of the token being exchanged; defaults to an ID
    /// token (`urn:ietf:params:oauth:token-type:id_token`).
    ///
    /// Supported keys: `unity_token_exchange_subject_token_type`.
    TokenExchangeSubjectTokenType,

    /// A well-known Azure identity option; see [`AzureIdentityKey`].
    Azure(AzureIdentityKey),

    /// A well-known Google identity option; see [`GoogleIdentityKey`].
    Google(GoogleIdentityKey),

    /// A Databricks authentication or HTTP client option, resolved by
    /// [`olai_http`]'s Databricks credential chain: `databricks_host`,
    /// `databricks_token`, `databricks_client_id` + `databricks_client_secret`,
    /// `databricks_azure_tenant_id`, `databricks_auth_type`,
    /// `databricks_config_profile`, …, each with or without the `databricks_`
    /// prefix. HTTP client options (`allow_http`, `timeout`, …) also apply to
    /// the other authentication types.
    Databricks(DatabricksConfigKey),
}

impl AsRef<str> for UnityConfigKey {
    fn as_ref(&self) -> &str {
        match self {
            Self::Url => "unity_catalog_url",
            Self::AwsRegion => "aws_region",
            Self::AllowUnauthenticated => "unity_allow_unauthenticated",
            Self::AuthType => "unity_auth_type",
            Self::Token => "unity_token",
            Self::ClientId => "unity_client_id",
            Self::ClientSecret => "unity_client_secret",
            Self::OAuthTokenUrl => "unity_oauth_token_url",
            Self::OAuthScope => "unity_oauth_scope",
            Self::OAuthAudience => "unity_oauth_audience",
            Self::AzureTenantId => "unity_azure_tenant_id",
            Self::AzureAuthorityHost => "unity_azure_authority_host",
            Self::TokenFile => "unity_token_file",
            Self::TokenEnv => "unity_token_env",
            Self::TokenExchange => "unity_token_exchange",
            Self::TokenExchangeSubjectTokenType => "unity_token_exchange_subject_token_type",
            Self::Azure(key) => key.as_ref(),
            Self::Google(key) => key.as_ref(),
            Self::Databricks(key) => key.as_ref(),
        }
    }
}

impl FromStr for UnityConfigKey {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "unity_catalog_url" | "unity_url" => Ok(Self::Url),
            "aws_region" | "unity_aws_region" => Ok(Self::AwsRegion),
            "unity_allow_unauthenticated" => Ok(Self::AllowUnauthenticated),
            "unity_auth_type" => Ok(Self::AuthType),
            "unity_token" | "unity_access_token" => Ok(Self::Token),
            "unity_client_id" => Ok(Self::ClientId),
            "unity_client_secret" => Ok(Self::ClientSecret),
            "unity_oauth_token_url" => Ok(Self::OAuthTokenUrl),
            "unity_oauth_scope" => Ok(Self::OAuthScope),
            "unity_oauth_audience" => Ok(Self::OAuthAudience),
            "unity_azure_tenant_id" => Ok(Self::AzureTenantId),
            "unity_azure_authority_host" => Ok(Self::AzureAuthorityHost),
            "unity_token_file" => Ok(Self::TokenFile),
            "unity_token_env" => Ok(Self::TokenEnv),
            "unity_token_exchange" => Ok(Self::TokenExchange),
            "unity_token_exchange_subject_token_type" => Ok(Self::TokenExchangeSubjectTokenType),
            _ => s
                .parse()
                .map(Self::Azure)
                .or_else(|_| s.parse().map(Self::Google))
                .or_else(|_| s.parse().map(Self::Databricks))
                .map_err(|_| Error::UnknownConfigurationKey(s.to_string())),
        }
    }
}

/// A well-known Azure identity option, named as `object_store`'s
/// `MicrosoftAzureBuilder` names it.
///
/// These describe the identity a deployment provisions for its workloads, and
/// are used when no Unity-specific credentials are set; see [`UnityAuthType`].
/// [`UnityStoreFactoryBuilder::from_env`] reads them from the matching
/// `AZURE_*` variables (and `IDENTITY_ENDPOINT`). Unprefixed aliases such as
/// `client_id` are not accepted: they are ambiguous in an options map shared
/// with other stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AzureIdentityKey {
    /// The client id of a service principal, a workload identity, or a
    /// user-assigned managed identity.
    ///
    /// Supported keys: `azure_client_id`, `azure_storage_client_id`.
    ClientId,

    /// The client secret of a service principal.
    ///
    /// Supported keys: `azure_client_secret`, `azure_storage_client_secret`.
    ClientSecret,

    /// The Azure AD tenant.
    ///
    /// Supported keys: `azure_tenant_id`, `azure_authority_id`,
    /// `azure_storage_tenant_id`, `azure_storage_authority_id`.
    TenantId,

    /// The Azure AD authority host; defaults to the public cloud.
    ///
    /// Supported keys: `azure_authority_host`, `azure_storage_authority_host`.
    AuthorityHost,

    /// The federated token file of a workload identity.
    ///
    /// Supported keys: `azure_federated_token_file`.
    FederatedTokenFile,

    /// The managed-identity token endpoint.
    ///
    /// Supported keys: `azure_msi_endpoint`, `azure_identity_endpoint`.
    MsiEndpoint,

    /// The object id of a user-assigned managed identity.
    ///
    /// Supported keys: `azure_object_id`.
    ObjectId,

    /// The Azure resource id of a user-assigned managed identity.
    ///
    /// Supported keys: `azure_msi_resource_id`.
    MsiResourceId,

    /// Use the signed-in Azure CLI.
    ///
    /// Supported keys: `azure_use_azure_cli`.
    UseAzureCli,
}

impl AsRef<str> for AzureIdentityKey {
    fn as_ref(&self) -> &str {
        match self {
            Self::ClientId => "azure_client_id",
            Self::ClientSecret => "azure_client_secret",
            Self::TenantId => "azure_tenant_id",
            Self::AuthorityHost => "azure_authority_host",
            Self::FederatedTokenFile => "azure_federated_token_file",
            Self::MsiEndpoint => "azure_msi_endpoint",
            Self::ObjectId => "azure_object_id",
            Self::MsiResourceId => "azure_msi_resource_id",
            Self::UseAzureCli => "azure_use_azure_cli",
        }
    }
}

impl FromStr for AzureIdentityKey {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "azure_client_id" | "azure_storage_client_id" => Ok(Self::ClientId),
            "azure_client_secret" | "azure_storage_client_secret" => Ok(Self::ClientSecret),
            "azure_tenant_id"
            | "azure_authority_id"
            | "azure_storage_tenant_id"
            | "azure_storage_authority_id" => Ok(Self::TenantId),
            "azure_authority_host" | "azure_storage_authority_host" => Ok(Self::AuthorityHost),
            "azure_federated_token_file" => Ok(Self::FederatedTokenFile),
            "azure_msi_endpoint" | "azure_identity_endpoint" => Ok(Self::MsiEndpoint),
            "azure_object_id" => Ok(Self::ObjectId),
            "azure_msi_resource_id" => Ok(Self::MsiResourceId),
            "azure_use_azure_cli" => Ok(Self::UseAzureCli),
            _ => Err(Error::UnknownConfigurationKey(s.to_string())),
        }
    }
}

/// A well-known Google identity option, named as `object_store`'s
/// `GoogleCloudStorageBuilder` names it.
///
/// A service-account key mints Google ID tokens for the Unity Catalog
/// audience; see [`UnityAuthType::GcpIdToken`].
/// [`UnityStoreFactoryBuilder::from_env`] reads them from the matching
/// `GOOGLE_*` variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GoogleIdentityKey {
    /// The path of a service-account key file.
    ///
    /// Supported keys: `google_service_account`, `google_service_account_path`.
    ServiceAccount,

    /// A service-account key, as JSON.
    ///
    /// Supported keys: `google_service_account_key`.
    ServiceAccountKey,

    /// The path of an application-default credentials file, which must hold
    /// a service-account key.
    ///
    /// Supported keys: `google_application_credentials`.
    ApplicationCredentials,
}

impl AsRef<str> for GoogleIdentityKey {
    fn as_ref(&self) -> &str {
        match self {
            Self::ServiceAccount => "google_service_account",
            Self::ServiceAccountKey => "google_service_account_key",
            Self::ApplicationCredentials => "google_application_credentials",
        }
    }
}

impl FromStr for GoogleIdentityKey {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "google_service_account" | "google_service_account_path" => Ok(Self::ServiceAccount),
            "google_service_account_key" => Ok(Self::ServiceAccountKey),
            "google_application_credentials" => Ok(Self::ApplicationCredentials),
            _ => Err(Error::UnknownConfigurationKey(s.to_string())),
        }
    }
}

/// How [`UnityStoreFactoryBuilder`] authenticates to Unity Catalog.
///
/// Set it with [`UnityConfigKey::AuthType`] (string form in parentheses).
/// When unset, it is detected from the configuration, most specific first:
///
/// 1. Unity-specific credentials: a token (`bearer`); a client id + secret
///    with a token URL (`oauth-client-credentials`) or an Azure tenant
///    (`azure-client-secret`); a token file (`token-file`) or variable
///    (`token-env`).
/// 2. Databricks options, or — for a Databricks workspace — the Databricks
///    environment or profile (`databricks`).
/// 3. The deployment's cloud identity, from the well-known
///    [`AzureIdentityKey`]s and [`GoogleIdentityKey`]s: an Azure service
///    principal, workload identity, the Azure CLI, or a managed identity; a
///    Google service account (`gcp-id-token`). Only used once the identity
///    can be scoped to Unity Catalog — an Azure scope
///    ([`UnityConfigKey::OAuthScope`], known for Azure Databricks) or a
///    Google audience ([`UnityConfigKey::OAuthAudience`]) — so an identity
///    provisioned for other services is not sent to an arbitrary server.
/// 4. Unauthenticated, if allowed (`none`).
///
/// Values resolve the same way: a `unity_*` key wins over its well-known
/// counterpart, e.g. `unity_azure_tenant_id` over `azure_tenant_id`. A
/// client id and secret are taken as a pair, never one from each.
///
/// An identity without configuration — a system-assigned managed identity, or
/// the GCP metadata server — is only used when selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnityAuthType {
    /// A static bearer token (`bearer`): [`UnityConfigKey::Token`].
    Bearer,
    /// OAuth 2.0 client credentials (`oauth-client-credentials`): client id,
    /// secret and token URL; scope and audience optional.
    OAuthClientCredentials,
    /// An Azure AD service principal (`azure-client-secret`): client id,
    /// secret, tenant and scope.
    AzureClientSecret,
    /// The host's Azure managed identity (`azure-managed-identity`): scope;
    /// client id for a user-assigned identity.
    AzureManagedIdentity,
    /// Azure workload identity (`azure-workload-identity`): client id, tenant,
    /// scope, and the federated token file (`AZURE_FEDERATED_TOKEN_FILE` when
    /// no token file is configured).
    AzureWorkloadIdentity,
    /// The signed-in Azure CLI (`azure-cli`): scope.
    AzureCli,
    /// A Google ID token (`gcp-id-token`) for the audience (defaults to the
    /// origin of the Unity Catalog URL), from a service-account key if one is
    /// configured, else from the metadata server.
    GcpIdToken,
    /// A token read from a file (`token-file`).
    TokenFile,
    /// A token read from an environment variable (`token-env`).
    TokenEnv,
    /// The Databricks credential chain (`databricks`).
    Databricks,
    /// No authentication (`none`).
    None,
}

impl FromStr for UnityAuthType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bearer" | "pat" => Ok(Self::Bearer),
            "oauth-client-credentials" => Ok(Self::OAuthClientCredentials),
            "azure-client-secret" => Ok(Self::AzureClientSecret),
            "azure-managed-identity" | "azure-msi" => Ok(Self::AzureManagedIdentity),
            "azure-workload-identity" => Ok(Self::AzureWorkloadIdentity),
            "azure-cli" => Ok(Self::AzureCli),
            "gcp-id-token" => Ok(Self::GcpIdToken),
            "token-file" => Ok(Self::TokenFile),
            "token-env" => Ok(Self::TokenEnv),
            "databricks" => Ok(Self::Databricks),
            "none" => Ok(Self::None),
            other => Err(Error::InvalidConfig(format!(
                "unknown `unity_auth_type` `{other}`"
            ))),
        }
    }
}

/// Builder for [`UnityStoreFactory`].
///
/// ```no_run
/// # use object_store_unity::{UnityConfigKey, UnityStoreFactoryBuilder};
/// # fn main() -> object_store_unity::Result<()> {
/// // Everything from the environment: `UNITY_*` variables, then Databricks
/// // (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, OAuth, a `~/.databrickscfg` profile).
/// let factory = UnityStoreFactoryBuilder::from_env().build()?;
///
/// // An OSS Unity Catalog behind an identity provider: exchange the pod's
/// // projected service-account token for a Unity Catalog token.
/// let factory = UnityStoreFactoryBuilder::new()
///     .with_url("https://uc.example.com/api/2.1/unity-catalog/")
///     .with_options([
///         ("unity_token_file", "/var/run/secrets/tokens/uc-token"),
///         ("unity_token_exchange", "true"),
///     ])
///     .build()?;
/// # Ok(()) }
/// ```
#[derive(Clone, Default)]
pub struct UnityStoreFactoryBuilder {
    url: Option<String>,
    aws_region: Option<String>,
    allow_unauthenticated: bool,
    auth_type: Option<String>,
    token: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    oauth_token_url: Option<String>,
    oauth_scope: Option<String>,
    oauth_audience: Option<String>,
    azure_tenant_id: Option<String>,
    azure_authority_host: Option<String>,
    token_file: Option<String>,
    token_env: Option<String>,
    token_exchange: Option<String>,
    token_exchange_subject_token_type: Option<String>,
    azure: HashMap<AzureIdentityKey, String>,
    google: HashMap<GoogleIdentityKey, String>,
    databricks: Vec<(DatabricksConfigKey, String)>,
    databricks_from_env: bool,
    client_options: ClientOptions,
    retry: RetryConfig,
    client: Option<CloudClient>,
    io_runtime: Option<Handle>,
}

impl UnityStoreFactoryBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder configured from the environment.
    ///
    /// Variables named after a [`UnityConfigKey`] (`UNITY_CATALOG_URL`,
    /// `UNITY_TOKEN`, `UNITY_CLIENT_ID`, …), an [`AzureIdentityKey`]
    /// (`AZURE_CLIENT_ID`, `AZURE_FEDERATED_TOKEN_FILE`, …, and
    /// `IDENTITY_ENDPOINT`) or a [`GoogleIdentityKey`]
    /// (`GOOGLE_APPLICATION_CREDENTIALS`, …) are applied as configuration.
    /// For a Databricks workspace, the Databricks credential chain also reads
    /// `DATABRICKS_*` variables and, beneath them, the selected
    /// `.databrickscfg` profile. Without an explicit URL, it is derived from
    /// `DATABRICKS_HOST`.
    ///
    /// Options set afterwards take precedence over the environment.
    pub fn from_env() -> Self {
        let mut builder = Self {
            databricks_from_env: true,
            ..Self::default()
        };
        for (name, value) in std::env::vars() {
            let lower = name.to_ascii_lowercase();
            let key = if name == "IDENTITY_ENDPOINT" {
                Some(UnityConfigKey::Azure(AzureIdentityKey::MsiEndpoint))
            } else if name.starts_with("UNITY_") {
                lower.parse().ok()
            } else if name.starts_with("AZURE_") {
                lower.parse().ok().map(UnityConfigKey::Azure)
            } else if name.starts_with("GOOGLE_") {
                lower.parse().ok().map(UnityConfigKey::Google)
            } else {
                None
            };
            if let Some(key) = key {
                builder = builder.with_config(key, value);
            }
        }
        builder
    }

    /// Set the Unity Catalog REST API base URL, e.g.
    /// `https://uc.example.com/api/2.1/unity-catalog/`.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Authenticate with a bearer token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Set the region of vended AWS locations; see [`UnityConfigKey::AwsRegion`].
    pub fn with_aws_region(mut self, region: impl Into<String>) -> Self {
        self.aws_region = Some(region.into());
        self
    }

    /// Send requests unauthenticated when no credentials are configured; see
    /// [`UnityConfigKey::AllowUnauthenticated`].
    pub fn with_allow_unauthenticated(mut self, allow: bool) -> Self {
        self.allow_unauthenticated = allow;
        self
    }

    /// Set the HTTP client options for Unity Catalog and token requests.
    ///
    /// Client options set through [`UnityConfigKey::Databricks`] keys
    /// (`allow_http`, `timeout`, …) are applied on top.
    pub fn with_client_options(mut self, options: ClientOptions) -> Self {
        self.client_options = options;
        self
    }

    /// Set the retry configuration for Unity Catalog and token requests.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// Use `client` for every Unity Catalog request.
    ///
    /// This bypasses all authentication configuration on this builder. Use it
    /// to supply a [`CloudClient`] with custom signing or a decorated
    /// [`HttpService`](olai_http::service::HttpService), e.g. to forward a
    /// trusted identity header.
    pub fn with_client(mut self, client: CloudClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Spawn all HTTP I/O — Unity Catalog requests, credential refreshes, and
    /// the built object stores' requests — on `handle`.
    ///
    /// Common in DataFusion deployments, which keep network I/O off the runtime
    /// that runs CPU-bound query work.
    pub fn with_io_runtime(mut self, handle: Handle) -> Self {
        self.io_runtime = Some(handle);
        self
    }

    /// Set a configuration value.
    pub fn with_config(mut self, key: UnityConfigKey, value: impl Into<String>) -> Self {
        let value = value.into();
        match key {
            UnityConfigKey::Url => self.url = Some(value),
            UnityConfigKey::AwsRegion => self.aws_region = Some(value),
            UnityConfigKey::AllowUnauthenticated => {
                self.allow_unauthenticated = str_is_truthy(&value)
            }
            UnityConfigKey::AuthType => self.auth_type = Some(value),
            UnityConfigKey::Token => self.token = Some(value),
            UnityConfigKey::ClientId => self.client_id = Some(value),
            UnityConfigKey::ClientSecret => self.client_secret = Some(value),
            UnityConfigKey::OAuthTokenUrl => self.oauth_token_url = Some(value),
            UnityConfigKey::OAuthScope => self.oauth_scope = Some(value),
            UnityConfigKey::OAuthAudience => self.oauth_audience = Some(value),
            UnityConfigKey::AzureTenantId => self.azure_tenant_id = Some(value),
            UnityConfigKey::AzureAuthorityHost => self.azure_authority_host = Some(value),
            UnityConfigKey::TokenFile => self.token_file = Some(value),
            UnityConfigKey::TokenEnv => self.token_env = Some(value),
            UnityConfigKey::TokenExchange => self.token_exchange = Some(value),
            UnityConfigKey::TokenExchangeSubjectTokenType => {
                self.token_exchange_subject_token_type = Some(value)
            }
            UnityConfigKey::Azure(key) => {
                self.azure.insert(key, value);
            }
            UnityConfigKey::Google(key) => {
                self.google.insert(key, value);
            }
            UnityConfigKey::Databricks(key) => {
                self.databricks.retain(|(existing, _)| existing != &key);
                self.databricks.push((key, value));
            }
        }
        self
    }

    /// Set configuration values from string keys, ignoring keys that are not a
    /// [`UnityConfigKey`].
    ///
    /// Lenient on purpose: callers can pass a whole storage-options map that
    /// also configures other stores.
    pub fn with_options<I, K, V>(self, options: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        options
            .into_iter()
            .fold(self, |builder, (key, value)| match key.as_ref().parse() {
                Ok(key) => builder.with_config(key, value),
                Err(_) => builder,
            })
    }

    /// Get a configuration value, if set.
    pub fn get_config_value(&self, key: &UnityConfigKey) -> Option<String> {
        let value = match key {
            UnityConfigKey::Url => &self.url,
            UnityConfigKey::AwsRegion => &self.aws_region,
            UnityConfigKey::AllowUnauthenticated => {
                return Some(self.allow_unauthenticated.to_string());
            }
            UnityConfigKey::AuthType => &self.auth_type,
            UnityConfigKey::Token => &self.token,
            UnityConfigKey::ClientId => &self.client_id,
            UnityConfigKey::ClientSecret => &self.client_secret,
            UnityConfigKey::OAuthTokenUrl => &self.oauth_token_url,
            UnityConfigKey::OAuthScope => &self.oauth_scope,
            UnityConfigKey::OAuthAudience => &self.oauth_audience,
            UnityConfigKey::AzureTenantId => &self.azure_tenant_id,
            UnityConfigKey::AzureAuthorityHost => &self.azure_authority_host,
            UnityConfigKey::TokenFile => &self.token_file,
            UnityConfigKey::TokenEnv => &self.token_env,
            UnityConfigKey::TokenExchange => &self.token_exchange,
            UnityConfigKey::TokenExchangeSubjectTokenType => {
                &self.token_exchange_subject_token_type
            }
            UnityConfigKey::Azure(key) => return self.azure.get(key).cloned(),
            UnityConfigKey::Google(key) => return self.google.get(key).cloned(),
            UnityConfigKey::Databricks(key) => {
                return self.databricks_value(*key).map(str::to_string);
            }
        };
        value.clone()
    }

    /// Build the factory.
    ///
    /// No request is sent yet; credentials are resolved when the first store
    /// is requested.
    pub fn build(self) -> Result<UnityStoreFactory> {
        let url = self.resolve_url()?;
        let client = self.transport(&url)?;
        Ok(UnityStoreFactory::new(
            UnityClient::new(client, url),
            self.aws_region,
            self.io_runtime,
        ))
    }

    /// Build just the authenticated client for Unity Catalog requests, and the
    /// REST API base URL it is for.
    pub fn build_client(&self) -> Result<(CloudClient, Url)> {
        let url = self.resolve_url()?;
        Ok((self.transport(&url)?, url))
    }

    fn databricks_value(&self, key: DatabricksConfigKey) -> Option<&str> {
        self.databricks
            .iter()
            .find(|(existing, _)| *existing == key)
            .map(|(_, value)| value.as_str())
    }

    /// The UC REST base URL: explicit, else derived from the Databricks host.
    fn resolve_url(&self) -> Result<Url> {
        let url = match &self.url {
            Some(url) => url.clone(),
            None => {
                let host = self
                    .databricks_value(DatabricksConfigKey::Host)
                    .map(str::to_string)
                    .or_else(|| {
                        self.databricks_from_env
                            .then(|| std::env::var("DATABRICKS_HOST").ok())
                            .flatten()
                    })
                    .ok_or_else(|| {
                        Error::InvalidConfig(
                            "missing Unity Catalog URL: set `unity_catalog_url` or a Databricks host"
                                .to_string(),
                        )
                    })?;
                let host = if host.contains("://") {
                    host
                } else {
                    format!("https://{host}")
                };
                format!("{}/{UNITY_CATALOG_API_PATH}", host.trim_end_matches('/'))
            }
        };
        Ok(Url::parse(&url)?)
    }

    /// HTTP client options: the base options plus client keys given as
    /// configuration, allowing HTTP when Unity Catalog itself is plain HTTP.
    fn http_options(&self, url: &Url) -> ClientOptions {
        let mut options = self.client_options.clone();
        for (key, value) in &self.databricks {
            if let DatabricksConfigKey::Client(key) = key {
                options = options.with_config(*key, value);
            }
        }
        if url.scheme() == "http" && self.databricks_value(allow_http_key()).is_none() {
            options = options.with_allow_http(true);
        }
        options
    }

    /// The authenticated transport for Unity Catalog requests.
    fn transport(&self, url: &Url) -> Result<CloudClient> {
        if let Some(client) = &self.client {
            return Ok(client.clone());
        }
        let auth_type = match &self.auth_type {
            Some(auth_type) => Some(auth_type.parse::<UnityAuthType>()?),
            None => None,
        };
        let auth_type = match auth_type {
            Some(auth_type) => auth_type,
            None => match self.detect_auth_type(url) {
                Some(auth_type) => auth_type,
                None if self.allow_unauthenticated => UnityAuthType::None,
                None => {
                    return Err(Error::InvalidConfig(
                        "no Unity Catalog credentials configured: set `unity_token`, \
                         `unity_client_id` + `unity_client_secret` (with `unity_oauth_token_url` \
                         or `unity_azure_tenant_id`), `unity_token_file`, `unity_auth_type`, \
                         Databricks credentials, an Azure or Google identity with \
                         `unity_oauth_scope` or `unity_oauth_audience`, or \
                         `unity_allow_unauthenticated`"
                            .to_string(),
                    ));
                }
            },
        };

        let exchange = self.token_exchange_endpoint(url)?;
        let options = self.http_options(url);
        match auth_type {
            UnityAuthType::Databricks | UnityAuthType::None if exchange.is_some() => {
                Err(Error::InvalidConfig(
                    "`unity_token_exchange` needs a token to exchange: configure a token, \
                     token file, or identity flow"
                        .to_string(),
                ))
            }
            UnityAuthType::Databricks => self.databricks_client(url),
            UnityAuthType::None => {
                let client = CloudClient::new_unauthenticated();
                Ok(match &self.io_runtime {
                    Some(handle) => client.with_runtime(handle.clone()),
                    None => client,
                })
            }
            _ => {
                let auth = self.bearer_auth(options.clone());
                let mut provider = self.bearer_provider(&auth, auth_type, url)?;
                if let Some(endpoint) = exchange {
                    provider = auth.token_exchange(TokenExchange {
                        endpoint,
                        subject: provider,
                        subject_token_type: self.token_exchange_subject_token_type.clone(),
                        requested_token_type: None,
                        audience: None,
                        scope: None,
                    })?;
                }
                Ok(CloudClient::new_with_bearer(
                    provider,
                    &options,
                    self.retry.clone(),
                    self.io_runtime.as_ref(),
                )?)
            }
        }
    }

    /// The authentication type implied by the configuration, if any; see
    /// [`UnityAuthType`] for the order.
    fn detect_auth_type(&self, url: &Url) -> Option<UnityAuthType> {
        let has_client = self.client_id.is_some() && self.client_secret.is_some();
        if self.token.is_some() {
            Some(UnityAuthType::Bearer)
        } else if has_client && self.oauth_token_url.is_some() {
            Some(UnityAuthType::OAuthClientCredentials)
        } else if has_client && self.tenant_id().is_some() {
            Some(UnityAuthType::AzureClientSecret)
        } else if self.token_file.is_some() {
            Some(UnityAuthType::TokenFile)
        } else if self.token_env.is_some() {
            Some(UnityAuthType::TokenEnv)
        } else if self.databricks_configured(url) {
            Some(UnityAuthType::Databricks)
        } else {
            self.detect_cloud_identity(url)
        }
    }

    /// The deployment's Azure or Google identity, once it can be scoped to
    /// Unity Catalog.
    fn detect_cloud_identity(&self, url: &Url) -> Option<UnityAuthType> {
        use AzureIdentityKey as Azure;

        if self.oauth_scope(url).is_some() {
            let client_id = self.client_id().is_some();
            let tenant_id = self.tenant_id().is_some();
            if client_id && tenant_id && self.client_secret().is_some() {
                return Some(UnityAuthType::AzureClientSecret);
            }
            if client_id && tenant_id && self.azure.contains_key(&Azure::FederatedTokenFile) {
                return Some(UnityAuthType::AzureWorkloadIdentity);
            }
            if self
                .azure
                .get(&Azure::UseAzureCli)
                .is_some_and(|value| str_is_truthy(value))
            {
                return Some(UnityAuthType::AzureCli);
            }
            let user_assigned = [Azure::ObjectId, Azure::MsiResourceId, Azure::MsiEndpoint]
                .iter()
                .any(|key| self.azure.contains_key(key));
            if client_id || user_assigned {
                return Some(UnityAuthType::AzureManagedIdentity);
            }
        }
        if self.oauth_audience.is_some() && !self.google.is_empty() {
            return Some(UnityAuthType::GcpIdToken);
        }
        None
    }

    /// Whether Databricks credentials are configured: explicitly, or — for a
    /// Databricks workspace — through `DATABRICKS_*` variables or a profile.
    fn databricks_configured(&self, url: &Url) -> bool {
        let explicit = self.databricks.iter().any(|(key, _)| {
            !matches!(
                key,
                DatabricksConfigKey::Client(_) | DatabricksConfigKey::Host
            )
        });
        explicit
            || (self.databricks_from_env
                && self.databricks_target(url)
                && self.databricks_client(url).is_ok())
    }

    /// Whether Unity Catalog is served by a Databricks workspace.
    ///
    /// The Databricks environment is only consulted for one, as the chain
    /// also reacts to variables of other services, such as
    /// `GOOGLE_APPLICATION_CREDENTIALS`.
    fn databricks_target(&self, url: &Url) -> bool {
        self.databricks_value(DatabricksConfigKey::Host).is_some()
            || host_in(url, DATABRICKS_DOMAINS)
            || host_in(url, AZURE_DATABRICKS_DOMAINS)
            || (self.databricks_from_env
                && std::env::vars().any(|(name, _)| name.starts_with("DATABRICKS_")))
    }

    /// The client id: Unity-specific, else the Azure one — unless only a
    /// Unity-specific secret is set, which must not pair with another
    /// identity's id.
    fn client_id(&self) -> Option<String> {
        self.client_id.clone().or_else(|| {
            self.client_secret
                .is_none()
                .then(|| self.azure.get(&AzureIdentityKey::ClientId).cloned())
                .flatten()
        })
    }

    /// The client secret, paired like [`client_id`](Self::client_id).
    fn client_secret(&self) -> Option<String> {
        self.client_secret.clone().or_else(|| {
            self.client_id
                .is_none()
                .then(|| self.azure.get(&AzureIdentityKey::ClientSecret).cloned())
                .flatten()
        })
    }

    fn tenant_id(&self) -> Option<String> {
        self.azure_tenant_id
            .clone()
            .or_else(|| self.azure.get(&AzureIdentityKey::TenantId).cloned())
    }

    fn authority_host(&self) -> Option<String> {
        self.azure_authority_host
            .clone()
            .or_else(|| self.azure.get(&AzureIdentityKey::AuthorityHost).cloned())
    }

    /// The Azure AD scope of Unity Catalog: configured, or that of Azure
    /// Databricks for one of its workspaces.
    fn oauth_scope(&self, url: &Url) -> Option<String> {
        self.oauth_scope.clone().or_else(|| {
            host_in(url, AZURE_DATABRICKS_DOMAINS).then(|| AZURE_DATABRICKS_SCOPE.to_string())
        })
    }

    fn token_exchange_endpoint(&self, url: &Url) -> Result<Option<String>> {
        let Some(value) = &self.token_exchange else {
            return Ok(None);
        };
        if value.contains("://") {
            return Ok(Some(value.clone()));
        }
        if !str_is_truthy(value) {
            return Ok(None);
        }
        // The endpoint sits beside the REST API, below the server root.
        let root = url
            .as_str()
            .strip_suffix(UNITY_CATALOG_API_PATH)
            .map(str::to_string)
            .unwrap_or_else(|| format!("{}/", url.origin().ascii_serialization()));
        Ok(Some(format!("{root}{UNITY_TOKEN_EXCHANGE_PATH}")))
    }

    fn bearer_auth(&self, options: ClientOptions) -> BearerAuth {
        let auth = BearerAuth::new()
            .with_client_options(options)
            .with_retry(self.retry.clone());
        match &self.io_runtime {
            Some(handle) => auth.with_runtime(handle.clone()),
            None => auth,
        }
    }

    fn bearer_provider(
        &self,
        auth: &BearerAuth,
        auth_type: UnityAuthType,
        url: &Url,
    ) -> Result<BearerCredentialProvider> {
        let required = |value: Option<String>, keys: &[UnityConfigKey]| {
            value.ok_or_else(|| {
                let keys = keys
                    .iter()
                    .map(|key| format!("`{}`", key.as_ref()))
                    .collect::<Vec<_>>()
                    .join(" or ");
                Error::InvalidConfig(format!("`unity_auth_type` `{auth_type:?}` requires {keys}"))
            })
        };
        use AzureIdentityKey as Azure;
        let client_id_keys = [
            UnityConfigKey::ClientId,
            UnityConfigKey::Azure(Azure::ClientId),
        ];
        let client_secret_keys = [
            UnityConfigKey::ClientSecret,
            UnityConfigKey::Azure(Azure::ClientSecret),
        ];
        let tenant_keys = [
            UnityConfigKey::AzureTenantId,
            UnityConfigKey::Azure(Azure::TenantId),
        ];
        let scope = || required(self.oauth_scope(url), &[UnityConfigKey::OAuthScope]);
        let azure = |key| self.azure.get(&key).cloned();

        let provider = match auth_type {
            UnityAuthType::Bearer => {
                BearerAuth::static_token(required(self.token.clone(), &[UnityConfigKey::Token])?)
            }
            UnityAuthType::OAuthClientCredentials => {
                auth.client_credentials(ClientCredentials {
                    token_url: required(
                        self.oauth_token_url.clone(),
                        &[UnityConfigKey::OAuthTokenUrl],
                    )?,
                    client_id: required(self.client_id.clone(), &[UnityConfigKey::ClientId])?,
                    client_secret: required(
                        self.client_secret.clone(),
                        &[UnityConfigKey::ClientSecret],
                    )?,
                    scope: self.oauth_scope.clone(),
                    audience: self.oauth_audience.clone(),
                })?
            }
            UnityAuthType::AzureClientSecret => auth.azure_client_secret(
                &required(self.tenant_id(), &tenant_keys)?,
                required(self.client_id(), &client_id_keys)?,
                required(self.client_secret(), &client_secret_keys)?,
                &scope()?,
                self.authority_host(),
            )?,
            UnityAuthType::AzureManagedIdentity => {
                let scope = scope()?;
                let resource = scope.strip_suffix("/.default").unwrap_or(&scope);
                auth.azure_managed_identity(AzureManagedIdentity {
                    resource: resource.to_string(),
                    client_id: self.client_id(),
                    object_id: azure(Azure::ObjectId),
                    msi_resource_id: azure(Azure::MsiResourceId),
                    endpoint: azure(Azure::MsiEndpoint),
                })?
            }
            UnityAuthType::AzureWorkloadIdentity => auth.azure_workload_identity(
                &required(self.tenant_id(), &tenant_keys)?,
                required(self.client_id(), &client_id_keys)?,
                required(
                    self.token_file
                        .clone()
                        .or_else(|| azure(Azure::FederatedTokenFile)),
                    &[
                        UnityConfigKey::TokenFile,
                        UnityConfigKey::Azure(Azure::FederatedTokenFile),
                    ],
                )?,
                &scope()?,
                self.authority_host(),
            )?,
            UnityAuthType::AzureCli => auth.azure_cli(scope()?),
            UnityAuthType::GcpIdToken => {
                let audience = self
                    .oauth_audience
                    .clone()
                    .unwrap_or_else(|| url.origin().ascii_serialization());
                let google = |key| self.google.get(&key).cloned();
                let account = google(GoogleIdentityKey::ServiceAccountKey)
                    .map(GcpServiceAccount::Key)
                    .or_else(|| {
                        google(GoogleIdentityKey::ServiceAccount)
                            .or_else(|| google(GoogleIdentityKey::ApplicationCredentials))
                            .map(GcpServiceAccount::File)
                    });
                match account {
                    Some(account) => auth.gcp_service_account_id_token(account, audience)?,
                    None => auth.gcp_id_token(audience, None)?,
                }
            }
            UnityAuthType::TokenFile => auth.token_file(required(
                self.token_file.clone(),
                &[UnityConfigKey::TokenFile],
            )?),
            UnityAuthType::TokenEnv => auth.token_env(required(
                self.token_env.clone(),
                &[UnityConfigKey::TokenEnv],
            )?),
            UnityAuthType::Databricks | UnityAuthType::None => {
                unreachable!("not a bearer flow")
            }
        };
        Ok(provider)
    }

    fn databricks_client(&self, url: &Url) -> Result<CloudClient> {
        let mut builder = if self.databricks_from_env {
            DatabricksBuilder::from_env()
        } else {
            DatabricksBuilder::new()
        };
        builder = builder
            .with_client_options(self.http_options(url))
            .with_retry(self.retry.clone());
        for (key, value) in &self.databricks {
            if !matches!(key, DatabricksConfigKey::Client(_)) {
                builder = builder.with_config(*key, value);
            }
        }
        if self.databricks_value(DatabricksConfigKey::Host).is_none() {
            // OAuth flows resolve their token endpoint against the workspace
            // host, which is the origin of the Unity Catalog URL.
            builder = builder.with_config(
                DatabricksConfigKey::Host,
                url.origin().ascii_serialization(),
            );
        }
        Ok(CloudClient::new_databricks_from_builder(
            builder,
            self.io_runtime.as_ref(),
        )?)
    }
}

/// Whether the host of `url` is one of `domains` or below one.
fn host_in(url: &Url, domains: &[&str]) -> bool {
    url.host_str().is_some_and(|host| {
        domains.iter().any(|domain| {
            host == *domain
                || host
                    .strip_suffix(domain)
                    .is_some_and(|prefix| prefix.ends_with('.'))
        })
    })
}

fn allow_http_key() -> DatabricksConfigKey {
    DatabricksConfigKey::Client(ClientConfigKey::AllowHttp)
}

impl fmt::Debug for UnityStoreFactoryBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = |value: &Option<String>| value.as_ref().map(|_| "***");
        // Secrets (tokens, client secrets, option values) are never printed.
        f.debug_struct("UnityStoreFactoryBuilder")
            .field("url", &self.url)
            .field("aws_region", &self.aws_region)
            .field("allow_unauthenticated", &self.allow_unauthenticated)
            .field("auth_type", &self.auth_type)
            .field("token", &set(&self.token))
            .field("client_id", &self.client_id)
            .field("client_secret", &set(&self.client_secret))
            .field("oauth_token_url", &self.oauth_token_url)
            .field("oauth_scope", &self.oauth_scope)
            .field("oauth_audience", &self.oauth_audience)
            .field("azure_tenant_id", &self.azure_tenant_id)
            .field("token_file", &self.token_file)
            .field("token_env", &self.token_env)
            .field("token_exchange", &self.token_exchange)
            .field(
                "azure",
                &self
                    .azure
                    .keys()
                    .map(|key| key.as_ref())
                    .collect::<Vec<_>>(),
            )
            .field(
                "google",
                &self
                    .google
                    .keys()
                    .map(|key| key.as_ref())
                    .collect::<Vec<_>>(),
            )
            .field(
                "databricks",
                &self
                    .databricks
                    .iter()
                    .map(|(key, _)| key.as_ref())
                    .collect::<Vec<_>>(),
            )
            .field("databricks_from_env", &self.databricks_from_env)
            .field("client", &self.client.as_ref().map(|_| "CloudClient"))
            .field("io_runtime", &self.io_runtime.is_some())
            .finish()
    }
}

fn str_is_truthy(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "yes" | "y"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_keys_roundtrip() {
        for key in [
            UnityConfigKey::Url,
            UnityConfigKey::AwsRegion,
            UnityConfigKey::AllowUnauthenticated,
            UnityConfigKey::AuthType,
            UnityConfigKey::Token,
            UnityConfigKey::ClientId,
            UnityConfigKey::ClientSecret,
            UnityConfigKey::OAuthTokenUrl,
            UnityConfigKey::OAuthScope,
            UnityConfigKey::OAuthAudience,
            UnityConfigKey::AzureTenantId,
            UnityConfigKey::AzureAuthorityHost,
            UnityConfigKey::TokenFile,
            UnityConfigKey::TokenEnv,
            UnityConfigKey::TokenExchange,
            UnityConfigKey::TokenExchangeSubjectTokenType,
            UnityConfigKey::Azure(AzureIdentityKey::ClientId),
            UnityConfigKey::Azure(AzureIdentityKey::ClientSecret),
            UnityConfigKey::Azure(AzureIdentityKey::TenantId),
            UnityConfigKey::Azure(AzureIdentityKey::AuthorityHost),
            UnityConfigKey::Azure(AzureIdentityKey::FederatedTokenFile),
            UnityConfigKey::Azure(AzureIdentityKey::MsiEndpoint),
            UnityConfigKey::Azure(AzureIdentityKey::ObjectId),
            UnityConfigKey::Azure(AzureIdentityKey::MsiResourceId),
            UnityConfigKey::Azure(AzureIdentityKey::UseAzureCli),
            UnityConfigKey::Google(GoogleIdentityKey::ServiceAccount),
            UnityConfigKey::Google(GoogleIdentityKey::ServiceAccountKey),
            UnityConfigKey::Google(GoogleIdentityKey::ApplicationCredentials),
            UnityConfigKey::Databricks(DatabricksConfigKey::ClientId),
        ] {
            assert_eq!(key.as_ref().parse::<UnityConfigKey>().unwrap(), key);
        }
        // object_store's prefixed aliases select the well-known identity.
        for (alias, key) in [
            ("azure_storage_client_id", AzureIdentityKey::ClientId),
            ("azure_storage_tenant_id", AzureIdentityKey::TenantId),
            ("azure_authority_id", AzureIdentityKey::TenantId),
            ("azure_identity_endpoint", AzureIdentityKey::MsiEndpoint),
        ] {
            assert_eq!(
                alias.parse::<UnityConfigKey>().unwrap(),
                UnityConfigKey::Azure(key)
            );
        }
        assert_eq!(
            "google_service_account_path"
                .parse::<UnityConfigKey>()
                .unwrap(),
            UnityConfigKey::Google(GoogleIdentityKey::ServiceAccount)
        );
        assert_eq!(
            "token".parse::<UnityConfigKey>().unwrap(),
            UnityConfigKey::Databricks(DatabricksConfigKey::Token)
        );
        assert!(matches!(
            "definitely_not_a_key".parse::<UnityConfigKey>(),
            Err(Error::UnknownConfigurationKey(_))
        ));
    }

    #[test]
    fn with_options_ignores_unknown_keys() {
        let builder = UnityStoreFactoryBuilder::new().with_options([
            ("unity_catalog_url", "https://ws/api/2.1/unity-catalog/"),
            ("aws_region", "eu-west-1"),
            ("some_other_store_option", "x"),
        ]);
        assert_eq!(
            builder
                .get_config_value(&UnityConfigKey::AwsRegion)
                .as_deref(),
            Some("eu-west-1")
        );
        assert_eq!(
            builder.get_config_value(&UnityConfigKey::Url).as_deref(),
            Some("https://ws/api/2.1/unity-catalog/")
        );
    }

    #[test]
    fn url_defaults_to_the_databricks_host() {
        for host in [
            "https://ws.cloud.databricks.com/",
            "ws.cloud.databricks.com",
        ] {
            let builder = UnityStoreFactoryBuilder::new()
                .with_config(UnityConfigKey::Databricks(DatabricksConfigKey::Host), host);
            assert_eq!(
                builder.resolve_url().unwrap().as_str(),
                "https://ws.cloud.databricks.com/api/2.1/unity-catalog/"
            );
        }
        assert!(UnityStoreFactoryBuilder::new().resolve_url().is_err());
    }

    #[test]
    fn missing_credentials_error_unless_unauthenticated_allowed() {
        let builder = UnityStoreFactoryBuilder::new()
            .with_url("http://localhost:8080/api/2.1/unity-catalog/");
        assert!(builder.clone().build().is_err());
        assert!(builder.with_allow_unauthenticated(true).build().is_ok());
    }

    #[test]
    fn auth_type_detection_prefers_explicit_unity_config() {
        let url = Url::parse("https://uc.example.com/api/2.1/unity-catalog/").unwrap();
        let detect = |builder: UnityStoreFactoryBuilder| builder.detect_auth_type(&url);
        let builder = UnityStoreFactoryBuilder::new().with_options([
            ("databricks_client_id", "db"),
            ("databricks_client_secret", "x"),
        ]);
        assert_eq!(detect(builder.clone()), Some(UnityAuthType::Databricks));
        assert_eq!(
            detect(builder.with_token("t")),
            Some(UnityAuthType::Bearer),
            "an explicit Unity Catalog token beats Databricks options"
        );
        let client = UnityStoreFactoryBuilder::new()
            .with_options([("unity_client_id", "app"), ("unity_client_secret", "s")]);
        assert_eq!(
            detect(client.clone()),
            None,
            "a client needs a token URL or a tenant"
        );
        assert_eq!(
            detect(
                client
                    .clone()
                    .with_options([("unity_oauth_token_url", "https://idp/token")])
            ),
            Some(UnityAuthType::OAuthClientCredentials)
        );
        assert_eq!(
            detect(client.with_options([("unity_azure_tenant_id", "t")])),
            Some(UnityAuthType::AzureClientSecret)
        );
        // Only client options are not Databricks credentials.
        let options = UnityStoreFactoryBuilder::new().with_options([("allow_http", "true")]);
        assert_eq!(detect(options), None);
    }

    #[test]
    fn cloud_identities_need_a_unity_catalog_scope_or_audience() {
        let oss = Url::parse("https://uc.example.com/api/2.1/unity-catalog/").unwrap();
        let sp = UnityStoreFactoryBuilder::new().with_options([
            ("azure_client_id", "app"),
            ("azure_client_secret", "s"),
            ("azure_tenant_id", "t"),
        ]);
        assert_eq!(
            sp.detect_auth_type(&oss),
            None,
            "an Azure identity is not sent to a server without a scope"
        );
        let scoped = |builder: UnityStoreFactoryBuilder| {
            builder
                .with_options([("unity_oauth_scope", "api://uc/.default")])
                .detect_auth_type(&oss)
        };
        assert_eq!(scoped(sp.clone()), Some(UnityAuthType::AzureClientSecret));
        assert_eq!(
            scoped(UnityStoreFactoryBuilder::new().with_options([
                ("azure_client_id", "app"),
                ("azure_tenant_id", "t"),
                ("azure_federated_token_file", "/var/run/token"),
            ])),
            Some(UnityAuthType::AzureWorkloadIdentity)
        );
        assert_eq!(
            scoped(UnityStoreFactoryBuilder::new().with_options([("azure_use_azure_cli", "true")])),
            Some(UnityAuthType::AzureCli)
        );
        assert_eq!(
            scoped(UnityStoreFactoryBuilder::new().with_options([("azure_client_id", "uami")])),
            Some(UnityAuthType::AzureManagedIdentity)
        );
        assert_eq!(
            scoped(UnityStoreFactoryBuilder::new()),
            None,
            "a system-assigned identity is only used when selected"
        );

        // Azure Databricks has a well-known scope.
        let adb = Url::parse("https://adb-1.2.azuredatabricks.net/api/2.1/unity-catalog/").unwrap();
        assert_eq!(
            sp.detect_auth_type(&adb),
            Some(UnityAuthType::AzureClientSecret)
        );
        assert_eq!(
            sp.oauth_scope(&adb).as_deref(),
            Some(AZURE_DATABRICKS_SCOPE)
        );

        let gcp = UnityStoreFactoryBuilder::new()
            .with_options([("google_application_credentials", "/etc/sa.json")]);
        assert_eq!(gcp.detect_auth_type(&oss), None);
        assert_eq!(
            gcp.with_options([("unity_oauth_audience", "https://uc.example.com")])
                .detect_auth_type(&oss),
            Some(UnityAuthType::GcpIdToken)
        );
    }

    #[test]
    fn unity_specific_credentials_win_over_cloud_identities() {
        let url = Url::parse("https://uc.example.com/api/2.1/unity-catalog/").unwrap();
        let builder = UnityStoreFactoryBuilder::new().with_options([
            ("azure_client_id", "deployment-app"),
            ("azure_client_secret", "deployment-secret"),
            ("azure_tenant_id", "deployment-tenant"),
            ("unity_oauth_scope", "api://uc/.default"),
        ]);
        assert_eq!(
            builder.clone().with_token("t").detect_auth_type(&url),
            Some(UnityAuthType::Bearer)
        );
        let unity = builder.clone().with_options([
            ("unity_client_id", "uc-app"),
            ("unity_client_secret", "uc-secret"),
        ]);
        assert_eq!(unity.client_id().as_deref(), Some("uc-app"));
        assert_eq!(unity.client_secret().as_deref(), Some("uc-secret"));
        assert_eq!(
            unity.tenant_id().as_deref(),
            Some("deployment-tenant"),
            "the tenant falls back to the well-known key"
        );
        assert_eq!(
            unity
                .with_options([("unity_azure_tenant_id", "uc-tenant")])
                .tenant_id()
                .as_deref(),
            Some("uc-tenant")
        );

        // A client id and secret never mix identities.
        let id_only = builder.with_options([("unity_client_id", "uami")]);
        assert_eq!(id_only.client_id().as_deref(), Some("uami"));
        assert_eq!(id_only.client_secret(), None);
        assert_eq!(
            id_only.detect_auth_type(&url),
            Some(UnityAuthType::AzureManagedIdentity)
        );
    }

    #[test]
    fn databricks_target_is_recognised_by_host() {
        let builder = UnityStoreFactoryBuilder::new();
        for (url, expected) in [
            (
                "https://dbc-1.cloud.databricks.com/api/2.1/unity-catalog/",
                true,
            ),
            (
                "https://adb-1.2.azuredatabricks.net/api/2.1/unity-catalog/",
                true,
            ),
            (
                "https://1.2.gcp.databricks.com/api/2.1/unity-catalog/",
                true,
            ),
            ("https://uc.example.com/api/2.1/unity-catalog/", false),
            (
                "https://notazuredatabricks.net/api/2.1/unity-catalog/",
                false,
            ),
        ] {
            let url = Url::parse(url).unwrap();
            assert_eq!(builder.databricks_target(&url), expected, "{url}");
        }
        let with_host = builder.with_config(
            UnityConfigKey::Databricks(DatabricksConfigKey::Host),
            "https://dbc.private.example.com",
        );
        assert!(
            with_host.databricks_target(&Url::parse("https://dbc.private.example.com/").unwrap())
        );
    }

    #[test]
    fn forced_auth_types_report_missing_config() {
        for (auth_type, missing) in [
            ("azure-client-secret", "unity_azure_tenant_id"),
            ("azure-managed-identity", "unity_oauth_scope"),
            ("oauth-client-credentials", "unity_oauth_token_url"),
            ("token-file", "unity_token_file"),
        ] {
            let result = UnityStoreFactoryBuilder::new()
                .with_url("https://uc.example.com/api/2.1/unity-catalog/")
                .with_options([
                    ("unity_auth_type", auth_type),
                    ("unity_client_id", "app"),
                    ("unity_client_secret", "s"),
                ])
                .build();
            let Err(err) = result else {
                panic!("{auth_type} should fail without {missing}")
            };
            assert!(err.to_string().contains(missing), "{auth_type}: {err}");
        }
        assert!(
            UnityStoreFactoryBuilder::new()
                .with_url("https://uc.example.com/api/2.1/unity-catalog/")
                .with_options([("unity_auth_type", "kerberos")])
                .build()
                .is_err()
        );
    }

    #[test]
    fn token_exchange_endpoint_sits_beside_the_rest_api() {
        let url = Url::parse("https://uc.example.com/uc/api/2.1/unity-catalog/").unwrap();
        let endpoint = |value: &str| {
            UnityStoreFactoryBuilder::new()
                .with_options([("unity_token_exchange", value)])
                .token_exchange_endpoint(&url)
                .unwrap()
        };
        assert_eq!(
            endpoint("true").as_deref(),
            Some("https://uc.example.com/uc/api/1.0/unity-control/auth/tokens")
        );
        assert_eq!(
            endpoint("https://sts.example.com/token").as_deref(),
            Some("https://sts.example.com/token")
        );
        assert_eq!(endpoint("false"), None);
    }

    #[test]
    fn debug_redacts_secrets() {
        let builder = UnityStoreFactoryBuilder::new()
            .with_token("super-secret-token")
            .with_options([
                ("unity_client_secret", "super-secret-client"),
                ("databricks_client_secret", "super-secret-value"),
                ("azure_client_secret", "super-secret-azure"),
                ("google_service_account_key", "super-secret-google"),
            ]);
        let debug = format!("{builder:?}");
        assert!(!debug.contains("super-secret"), "secret leaked: {debug}");
        assert!(debug.contains("databricks_client_secret"));
    }
}
