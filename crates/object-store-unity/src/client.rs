//! A minimal Unity Catalog REST client covering credential vending.

use std::fmt;

use olai_http::CloudClient;
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use serde::de::DeserializeOwned;
use url::Url;

use crate::error::{Error, Result};
use crate::models::{
    PathCredentialsRequest, PathOperation, TableCredentialsRequest, TableInfo, TableOperation,
    TemporaryCredentials, VolumeCredentialsRequest, VolumeInfo, VolumeOperation,
};

/// Characters escaped in a single URL path segment (RFC 3986 `pchar` minus
/// sub-delimiters), so a securable name can never introduce a new segment.
const PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}');

/// Client for the Unity Catalog endpoints this crate needs: securable lookups
/// and the `temporary-*-credentials` vending endpoints.
///
/// Authentication, retries and the I/O runtime are owned by the wrapped
/// [`CloudClient`].
#[derive(Clone)]
pub struct UnityClient {
    client: CloudClient,
    base_url: Url,
}

impl UnityClient {
    /// Create a client for the Unity Catalog REST API rooted at `base_url`
    /// (e.g. `https://<workspace>/api/2.1/unity-catalog/`).
    pub fn new(client: CloudClient, mut base_url: Url) -> Self {
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path()));
        }
        Self { client, base_url }
    }

    /// The Unity Catalog REST API base URL.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Vend credentials for a table by id.
    pub async fn temporary_table_credentials(
        &self,
        table_id: &str,
        operation: TableOperation,
    ) -> Result<TemporaryCredentials> {
        let request = TableCredentialsRequest {
            table_id,
            operation,
        };
        self.post("temporary-table-credentials", &request).await
    }

    /// Vend credentials for a volume by id.
    pub async fn temporary_volume_credentials(
        &self,
        volume_id: &str,
        operation: VolumeOperation,
    ) -> Result<TemporaryCredentials> {
        let request = VolumeCredentialsRequest {
            volume_id,
            operation,
        };
        self.post("temporary-volume-credentials", &request).await
    }

    /// Vend credentials for a raw storage URL. With `dry_run`, the server only
    /// checks that the credential could be issued.
    pub async fn temporary_path_credentials(
        &self,
        url: &Url,
        operation: PathOperation,
        dry_run: bool,
    ) -> Result<TemporaryCredentials> {
        let request = PathCredentialsRequest {
            url: url.as_str(),
            operation,
            dry_run,
        };
        self.post("temporary-path-credentials", &request).await
    }

    /// Look up a table by its three-level `<catalog>.<schema>.<table>` name.
    pub async fn get_table(&self, full_name: &str) -> Result<TableInfo> {
        self.get("tables", full_name).await
    }

    /// Look up a volume by its three-level `<catalog>.<schema>.<volume>` name.
    pub async fn get_volume(&self, full_name: &str) -> Result<VolumeInfo> {
        self.get("volumes", full_name).await
    }

    async fn get<T: DeserializeOwned>(&self, collection: &str, name: &str) -> Result<T> {
        let name = utf8_percent_encode(name, PATH_SEGMENT);
        let url = self.base_url.join(&format!("{collection}/{name}"))?;
        let response = self.client.get(url).send().await?;
        decode(response).await
    }

    async fn post<B, T>(&self, endpoint: &str, body: &B) -> Result<T>
    where
        B: serde::Serialize + ?Sized,
        T: DeserializeOwned,
    {
        let url = self.base_url.join(endpoint)?;
        let response = self.client.post(url).json(body).send().await?;
        decode(response).await
    }
}

async fn decode<T: DeserializeOwned>(response: reqwest::Response) -> Result<T> {
    response
        .json()
        .await
        .map_err(|e| Error::InvalidResponse(e.to_string()))
}

impl fmt::Debug for UnityClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnityClient")
            .field("base_url", &self.base_url.as_str())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;

    use super::*;

    fn client(server: &MockServer) -> UnityClient {
        let base = Url::parse(&server.url("/api/2.1/unity-catalog")).unwrap();
        UnityClient::new(CloudClient::new_with_token("pat"), base)
    }

    #[tokio::test]
    async fn vends_table_credentials() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/2.1/unity-catalog/temporary-table-credentials")
                    .header("authorization", "Bearer pat")
                    .json_body(serde_json::json!({"table_id": "t-1", "operation": "READ"}));
                then.status(200).json_body(serde_json::json!({
                    "expiration_time": 1,
                    "url": "s3://bucket/t/",
                    "aws_temp_credentials": {"access_key_id": "ak", "secret_access_key": "sk"}
                }));
            })
            .await;

        let creds = client(&server)
            .temporary_table_credentials("t-1", TableOperation::Read)
            .await
            .unwrap();
        assert_eq!(creds.url.as_deref(), Some("s3://bucket/t/"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn vends_path_credentials_with_dry_run() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/2.1/unity-catalog/temporary-path-credentials")
                    .json_body(serde_json::json!({
                        "url": "abfss://c@a.dfs.core.windows.net/p",
                        "operation": "PATH_READ_WRITE",
                        "dry_run": true
                    }));
                then.status(200)
                    .json_body(serde_json::json!({"azure_aad": {"aad_token": "tok"}}));
            })
            .await;

        let url = Url::parse("abfss://c@a.dfs.core.windows.net/p").unwrap();
        client(&server)
            .temporary_path_credentials(&url, PathOperation::ReadWrite, true)
            .await
            .unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn get_table_encodes_name_as_one_segment() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/api/2.1/unity-catalog/tables/main.default.a%2Fb");
                then.status(200).json_body(serde_json::json!({
                    "table_id": "t-1",
                    "storage_location": "s3://bucket/t"
                }));
            })
            .await;

        let table = client(&server).get_table("main.default.a/b").await.unwrap();
        assert_eq!(table.table_id, "t-1");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn not_found_is_a_request_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/api/2.1/unity-catalog/volumes/main.default.v");
                then.status(404)
                    .json_body(serde_json::json!({"error_code": "NOT_FOUND"}));
            })
            .await;

        let err = client(&server)
            .get_volume("main.default.v")
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Request(_)), "unexpected error: {err}");
    }
}
