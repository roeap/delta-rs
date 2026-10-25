//! Configuration for [`UnityStoreFactory`].

use std::fmt;
use std::str::FromStr;

use olai_http::CloudClient;
use olai_http::databricks::{DatabricksBuilder, DatabricksConfigKey};
use tokio::runtime::Handle;
use url::Url;

use crate::client::UnityClient;
use crate::error::{Error, Result};
use crate::store::UnityStoreFactory;

/// The path of the Unity Catalog REST API on a Databricks workspace.
const UNITY_CATALOG_API_PATH: &str = "api/2.1/unity-catalog/";

/// Configuration keys for [`UnityStoreFactoryBuilder`].
///
/// Parse one from its string form with [`FromStr`]. Keys are matched exactly
/// (lowercase); environment variables are matched by their lowercased name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnityConfigKey {
    /// The Unity Catalog REST API base URL, e.g.
    /// `https://<workspace>/api/2.1/unity-catalog/`.
    ///
    /// Supported keys: `unity_catalog_url`, `unity_url`. Defaults to
    /// `<host>/api/2.1/unity-catalog/` when a Databricks host is configured.
    Url,

    /// Region for vended AWS credentials, when the location does not imply one.
    ///
    /// Supported keys: `aws_region`, `unity_aws_region`. Falls back to the
    /// `AWS_REGION` environment variable, then to object_store's default.
    AwsRegion,

    /// Send requests unauthenticated when no credentials are configured.
    ///
    /// Only meant for local development against an unauthenticated server.
    ///
    /// Supported keys: `unity_allow_unauthenticated`.
    AllowUnauthenticated,

    /// A Databricks authentication or HTTP client option, resolved by
    /// [`olai_http`]'s Databricks credential chain: `databricks_host`,
    /// `databricks_token` (PAT), `databricks_client_id` +
    /// `databricks_client_secret` (OAuth M2M), `databricks_oidc_token_env` /
    /// `databricks_oidc_token_filepath` (workload identity federation),
    /// `databricks_azure_resource_id` (Azure managed identity),
    /// `databricks_auth_type`, `databricks_config_profile`, and the client
    /// options (`allow_http`, `timeout`, …), each with or without the
    /// `databricks_` prefix.
    Databricks(DatabricksConfigKey),
}

impl AsRef<str> for UnityConfigKey {
    fn as_ref(&self) -> &str {
        match self {
            Self::Url => "unity_catalog_url",
            Self::AwsRegion => "aws_region",
            Self::AllowUnauthenticated => "unity_allow_unauthenticated",
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
            _ => s
                .parse()
                .map(Self::Databricks)
                .map_err(|_| Error::UnknownConfigurationKey(s.to_string())),
        }
    }
}

/// Builder for [`UnityStoreFactory`].
///
/// ```no_run
/// # use object_store_unity::{UnityConfigKey, UnityStoreFactoryBuilder};
/// # fn main() -> object_store_unity::Result<()> {
/// // Everything from the environment: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`,
/// // OAuth / OIDC variables, or a `~/.databrickscfg` profile.
/// let factory = UnityStoreFactoryBuilder::from_env().build()?;
///
/// // Or configure explicitly, e.g. with string options.
/// let factory = UnityStoreFactoryBuilder::new()
///     .with_url("https://my-workspace.cloud.databricks.com/api/2.1/unity-catalog/")
///     .with_options([
///         ("databricks_client_id", "<service principal id>"),
///         ("databricks_client_secret", "<secret>"),
///     ])
///     .build()?;
/// # Ok(()) }
/// ```
#[derive(Clone, Default)]
pub struct UnityStoreFactoryBuilder {
    url: Option<String>,
    aws_region: Option<String>,
    allow_unauthenticated: bool,
    databricks: Vec<(DatabricksConfigKey, String)>,
    databricks_from_env: bool,
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
    /// Databricks authentication comes from [`DatabricksBuilder::from_env`]:
    /// `DATABRICKS_*` variables and, beneath them, the selected
    /// `.databrickscfg` profile. Variables named after a [`UnityConfigKey`]
    /// (`UNITY_CATALOG_URL`, `UNITY_ALLOW_UNAUTHENTICATED`, …) are applied too.
    /// Without an explicit URL, it is derived from `DATABRICKS_HOST`.
    pub fn from_env() -> Self {
        let mut builder = Self {
            databricks_from_env: true,
            ..Self::default()
        };
        for (name, value) in std::env::vars() {
            if !name.starts_with("UNITY_") {
                continue;
            }
            if let Ok(key) = name.to_ascii_lowercase().parse() {
                builder = builder.with_config(key, value);
            }
        }
        builder
    }

    /// Set the Unity Catalog REST API base URL, e.g.
    /// `https://<workspace>/api/2.1/unity-catalog/`.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Authenticate with a personal access token.
    pub fn with_token(self, token: impl Into<String>) -> Self {
        self.with_config(
            UnityConfigKey::Databricks(DatabricksConfigKey::Token),
            token,
        )
    }

    /// Set the region for vended AWS credentials; see [`UnityConfigKey::AwsRegion`].
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
        match key {
            UnityConfigKey::Url => self.url.clone(),
            UnityConfigKey::AwsRegion => self.aws_region.clone(),
            UnityConfigKey::AllowUnauthenticated => Some(self.allow_unauthenticated.to_string()),
            UnityConfigKey::Databricks(key) => self.databricks_value(*key).map(str::to_string),
        }
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

    /// The authenticated transport for Unity Catalog requests.
    fn transport(&self, url: &Url) -> Result<CloudClient> {
        if let Some(client) = &self.client {
            return Ok(client.clone());
        }
        match self.databricks_client(url) {
            Ok(client) => Ok(client),
            Err(_) if self.allow_unauthenticated => {
                let client = CloudClient::new_unauthenticated();
                Ok(match &self.io_runtime {
                    Some(handle) => client.with_runtime(handle.clone()),
                    None => client,
                })
            }
            Err(e) => Err(e),
        }
    }

    fn databricks_client(&self, url: &Url) -> Result<CloudClient> {
        let mut builder = if self.databricks_from_env {
            DatabricksBuilder::from_env()
        } else {
            DatabricksBuilder::new()
        };
        for (key, value) in &self.databricks {
            builder = builder.with_config(*key, value);
        }
        if self.databricks_value(DatabricksConfigKey::Host).is_none() {
            // OAuth flows resolve their token endpoint against the workspace
            // host, which is the origin of the Unity Catalog URL.
            builder = builder.with_config(
                DatabricksConfigKey::Host,
                url.origin().ascii_serialization(),
            );
        }
        let allow_http = DatabricksConfigKey::Client(olai_http::ClientConfigKey::AllowHttp);
        if url.scheme() == "http" && self.databricks_value(allow_http).is_none() {
            // The caller chose plain HTTP for Unity Catalog itself (local
            // development); the token endpoint lives on the same host.
            builder = builder.with_config(allow_http, "true");
        }
        Ok(CloudClient::new_databricks_from_builder(
            builder,
            self.io_runtime.as_ref(),
        )?)
    }
}

impl fmt::Debug for UnityStoreFactoryBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Option values may be secrets (tokens, client secrets): keys only.
        f.debug_struct("UnityStoreFactoryBuilder")
            .field("url", &self.url)
            .field("aws_region", &self.aws_region)
            .field("allow_unauthenticated", &self.allow_unauthenticated)
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
            UnityConfigKey::Databricks(DatabricksConfigKey::ClientId),
        ] {
            assert_eq!(key.as_ref().parse::<UnityConfigKey>().unwrap(), key);
        }
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
    fn debug_redacts_option_values() {
        let builder = UnityStoreFactoryBuilder::new()
            .with_token("super-secret-token")
            .with_options([("databricks_client_secret", "super-secret-value")]);
        let debug = format!("{builder:?}");
        assert!(!debug.contains("super-secret"), "secret leaked: {debug}");
        assert!(debug.contains("databricks_client_secret"));
    }
}
