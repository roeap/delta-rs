/// Errors raised while resolving, vending, or building Unity Catalog stores.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A request to the Unity Catalog REST API failed.
    #[error("Unity Catalog request failed: {0}")]
    Request(#[from] olai_http::Error),

    /// Unity Catalog returned a response this crate could not interpret.
    #[error("invalid Unity Catalog response: {0}")]
    InvalidResponse(String),

    /// The builder was configured inconsistently or incompletely.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// A configuration key is not recognised.
    #[error("unknown configuration key `{0}`")]
    UnknownConfigurationKey(String),

    /// A securable or storage URL could not be parsed.
    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    /// A vended response carried no cloud credential.
    #[error("Unity Catalog vended no cloud credential")]
    MissingCredential,

    /// A vended credential does not match the storage it was requested for.
    #[error("credential mismatch: {0}")]
    CredentialMismatch(String),

    /// The credential targets a cloud whose crate feature is disabled.
    #[error("{cloud} storage requires the `{feature}` feature of object_store_unity")]
    FeatureDisabled {
        /// The storage provider the credential was vended for.
        cloud: &'static str,
        /// The crate feature that enables it.
        feature: &'static str,
    },
}

/// A specialized `Result` for this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

const STORE: &str = "UnityCatalog";

impl From<Error> for object_store::Error {
    fn from(err: Error) -> Self {
        object_store::Error::Generic {
            store: STORE,
            source: Box::new(err),
        }
    }
}

impl From<url::ParseError> for Error {
    fn from(err: url::ParseError) -> Self {
        Error::InvalidUrl(err.to_string())
    }
}
