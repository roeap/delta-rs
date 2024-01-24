//! Delta table checkpoint creation
//!
//! A checkpoint contains the complete replay of all actions, up to and including
//! the checkpointed table version, with invalid actions removed. Invalid actions
//! are those that have been canceled out by subsequent ones (for example removing
//! a file that has been added), using the [rules for reconciliation]. In addition to
//! above, checkpoint also contains the [remove tombstones] until they are expired.
//!
//! Checkpoints allow readers to short-cut the cost of reading the log up-to a given
//! point in order to reconstruct a snapshot, and they also allow [metadata cleanup]
//! to delete expired JSON Delta log entries.
//!
//! [rules for reconciliation]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#Action-Reconciliation
//! [remove tombstones]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#add-file-and-remove-file
//! [metadata cleanup]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#metadata-cleanup

use crate::table::config::CheckpointPolicy;

use super::{DeltaResult, Snapshot};

#[derive(thiserror::Error, Debug)]
#[allow(missing_docs)]
pub enum CheckpointError {
    #[error("Invalid checkpoint config: {0}")]
    InvalidConfig(&'static str),
}

/// Checkpoint specification
pub enum CheckpointSpec {
    /// Checkpoint accrding to the [V1 spec]
    ///
    /// [V1 spec]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#classic-checkpoint
    V1,
    /// Checkpoint accrding to the [V2 spec]
    ///
    /// [V2 spec]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#v2-spec
    V2,
}

/// Checkpoint naming scheme
///
/// The [naming scheme] defines how the checkpoint files are named.
///
/// [naming scheme]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#checkpoint-naming-scheme
pub enum CheckpointNamingScheme {
    UUID,
    Classic,
}

pub(crate) struct CheckpointBuilder<'a> {
    snapshot: &'a Snapshot,
    spec: CheckpointSpec,
    naming_scheme: CheckpointNamingScheme,
    write_sidecars: bool,
}

impl<'a> CheckpointBuilder<'a> {
    /// Create a new [`CheckpointBuilder`].
    pub fn new(snapshot: &'a Snapshot) -> Self {
        let (spec, naming_scheme) = match snapshot.table_config().checkpoint_policy() {
            CheckpointPolicy::V2 => (CheckpointSpec::V2, CheckpointNamingScheme::UUID),
            _ => (CheckpointSpec::V1, CheckpointNamingScheme::Classic),
        };
        Self {
            snapshot,
            spec,
            naming_scheme,
            write_sidecars: false,
        }
    }

    /// Set the specification version the checkpoint is to be created with.
    pub fn with_spec(mut self, spec: CheckpointSpec) -> Self {
        self.spec = spec;
        self
    }

    /// Set the naming scheme the checkpoint is to be created with.
    pub fn with_naming_scheme(mut self, naming_scheme: CheckpointNamingScheme) -> Self {
        self.naming_scheme = naming_scheme;
        self
    }

    /// Write add and remove actions to [sidecar files] (requires V2 checkpoint).
    ///
    /// [sidecar files]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#sidecar-files
    pub fn with_write_sidecars(mut self, write_sidecars: bool) -> Self {
        self.write_sidecars = write_sidecars;
        self
    }

    /// Validate that the checkpoint configuration is consistent accoring to the Delta protocol.
    fn validate_config(&self) -> DeltaResult<()> {
        match self.spec {
            CheckpointSpec::V1 => {
                if self.write_sidecars {
                    return Err(CheckpointError::InvalidConfig(
                        "V1 checkpoint does not support sidecars",
                    )
                    .into());
                }
                if matches!(self.naming_scheme, CheckpointNamingScheme::UUID) {
                    return Err(CheckpointError::InvalidConfig(
                        "V1 checkpoint does not support UUID naming scheme",
                    )
                    .into());
                }
            }
            // https://github.com/delta-io/delta/blob/master/PROTOCOL.md#v2-checkpoint-table-feature
            CheckpointSpec::V2 => {
                if (
                    self.snapshot.protocol().min_reader_version,
                    self.snapshot.protocol().min_writer_version,
                ) < (3, 7)
                {
                    return Err(CheckpointError::InvalidConfig(
                        "V2 checkpoint requires minReaderVersion >= 3 and minWriterVersion >= 7",
                    )
                    .into());
                }
            }
        }
        Ok(())
    }
}
