//! A served query's lifecycle events, as a query run in-process reports them.
//!
//! A served query reports that it was accepted and is waiting for an execution
//! slot, then that it started once its plan exists, its optimization, its
//! execution (from the engine), and that it finished, on every exit: completed,
//! failed, cancelled, timed out, or abandoned by its client. Events go to the
//! server process's subscribers under the query id the client chose.
//!
//! Reporting never fails a query: a subscriber's failure is logged and the
//! query goes on.

use std::{sync::Arc, time::Instant};

use common_metrics::{QueryEndState, QueryID};
use daft_context::subscribers::{QueryMetadata, QueryResult};
use daft_logical_plan::LogicalPlanBuilder;

use crate::{
    error::ServeError,
    wire::{QueryPayload, QueryRequest},
};

/// Label naming the engine that ran a served query in its started event.
const RUNNER: &str = "Serve (Swordfish)";

/// The lane a request arrived on, and a host request's kind.
fn lane(payload: &QueryPayload) -> (&'static str, Option<String>) {
    match payload {
        QueryPayload::Plan(_) => ("plan", None),
        QueryPayload::Sql(_) => ("sql", None),
        QueryPayload::Request { kind, .. } => ("request", Some(kind.clone())),
    }
}

/// The end state and message a query's outcome is reported with.
fn ending(outcome: &Result<(), ServeError>) -> (QueryEndState, String) {
    match outcome {
        Ok(()) => (QueryEndState::Finished, "Query finished".to_string()),
        Err(ServeError::Cancelled(_)) => (QueryEndState::Canceled, "Query canceled".to_string()),
        Err(err) => (QueryEndState::Failed, err.to_string()),
    }
}

/// Reports the events of planning a served query, which runs on a blocking
/// thread apart from the query's own lifecycle.
#[derive(Debug, Clone)]
pub struct PlanningEvents {
    query_id: QueryID,
    queued_seconds: f64,
}

impl PlanningEvents {
    /// Reports that the query started, once its plan exists, and that its
    /// optimization began.
    pub fn started(&self, builder: &LogicalPlanBuilder) {
        let ctx = daft_context::get_context();
        let unoptimized_plan = match builder.repr_json(false) {
            Ok(plan) => plan,
            Err(err) => {
                log::warn!("query {}: could not render its plan: {err}", self.query_id);
                return;
            }
        };
        let metadata = Arc::new(QueryMetadata {
            output_schema: builder.schema(),
            unoptimized_plan: unoptimized_plan.into(),
            runner: RUNNER.to_string(),
            ray_dashboard_url: None,
            entrypoint: None,
            python_version: None,
            daft_version: Some(common_version::VERSION.to_string()),
            ray_version: None,
            queued_seconds: Some(self.queued_seconds),
        });
        if let Err(err) = ctx.notify_query_start(self.query_id.clone(), metadata) {
            log::warn!("query {}: could not report its start: {err}", self.query_id);
        }
        if let Err(err) = ctx.notify_optimization_start(self.query_id.clone()) {
            log::warn!(
                "query {}: could not report its optimization: {err}",
                self.query_id
            );
        }
    }

    /// Reports that the query's optimization completed, with its plan.
    pub fn optimized(&self, builder: &LogicalPlanBuilder) {
        let ctx = daft_context::get_context();
        match builder.repr_json(false) {
            Ok(plan) => {
                if let Err(err) = ctx.notify_optimization_end(self.query_id.clone(), plan.into()) {
                    log::warn!(
                        "query {}: could not report its optimization: {err}",
                        self.query_id
                    );
                }
            }
            Err(err) => log::warn!("query {}: could not render its plan: {err}", self.query_id),
        }
    }
}

/// One served query's lifecycle, from acceptance to its end.
///
/// Created when the server accepts a request, which reports it queued. Its end
/// is reported exactly once: by [`QueryLifecycle::finish`], or when it is
/// dropped unfinished, which reports it cancelled (its task was abandoned).
#[derive(Debug)]
pub struct QueryLifecycle {
    query_id: QueryID,
    /// Whether any subscriber listens; when none does, nothing is rendered or
    /// reported.
    listening: bool,
    arrived: Instant,
    queued_seconds: f64,
    rows_returned: u64,
    bytes_returned: u64,
    ended: bool,
}

impl QueryLifecycle {
    /// Reports a request accepted by the server and waiting for a slot.
    #[must_use]
    pub fn accepted(request: &QueryRequest, caller: Option<&str>) -> Self {
        let ctx = daft_context::get_context();
        let listening = !ctx.subscribers().is_empty();
        let query_id: QueryID = request.query_id.as_str().into();
        if listening {
            let (lane, kind) = lane(&request.payload);
            if let Err(err) = ctx.notify_query_queued(
                query_id.clone(),
                lane.to_string(),
                kind,
                caller.map(str::to_string),
            ) {
                log::warn!("query {query_id}: could not report it queued: {err}");
            }
        }
        Self {
            query_id,
            listening,
            arrived: Instant::now(),
            queued_seconds: 0.0,
            rows_returned: 0,
            bytes_returned: 0,
            ended: false,
        }
    }

    /// Records that the query was admitted to an execution slot.
    pub fn admitted(&mut self) {
        self.queued_seconds = self.arrived.elapsed().as_secs_f64();
    }

    /// The reporter for the query's planning, or `None` when nobody listens.
    #[must_use]
    pub fn planning(&self) -> Option<PlanningEvents> {
        self.listening.then(|| PlanningEvents {
            query_id: self.query_id.clone(),
            queued_seconds: self.queued_seconds,
        })
    }

    /// Counts rows handed to the query's reader, and their in-memory size.
    pub fn returned(&mut self, rows: u64, bytes: u64) {
        self.rows_returned += rows;
        self.bytes_returned += bytes;
    }

    /// Reports the query's end with its outcome.
    pub fn finish(mut self, outcome: &Result<(), ServeError>) {
        let (end_state, message) = ending(outcome);
        self.report_end(end_state, message);
    }

    fn report_end(&mut self, end_state: QueryEndState, message: String) {
        self.ended = true;
        if !self.listening {
            return;
        }
        daft_context::get_context().notify_query_end(
            self.query_id.clone(),
            QueryResult {
                end_state,
                error_message: Some(message),
                rows_returned: Some(self.rows_returned),
                bytes_returned: Some(self.bytes_returned),
            },
        );
    }
}

impl Drop for QueryLifecycle {
    fn drop(&mut self) {
        if !self.ended {
            self.report_end(
                QueryEndState::Canceled,
                "Query canceled: its task was abandoned".to_string(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::finished(Ok(()), QueryEndState::Finished)]
    #[case::cancelled(Err(ServeError::Cancelled("q".into())), QueryEndState::Canceled)]
    #[case::timed_out(
        Err(ServeError::Timeout { query_id: "q".into(), limit_secs: 1 }),
        QueryEndState::Failed
    )]
    #[case::at_capacity(
        Err(ServeError::AtCapacity { max_concurrent: 1, waited_secs: 1 }),
        QueryEndState::Failed
    )]
    fn an_outcome_ends_in_its_state(
        #[case] outcome: Result<(), ServeError>,
        #[case] expected: QueryEndState,
    ) {
        let (state, _) = ending(&outcome);

        assert_eq!(format!("{state:?}"), format!("{expected:?}"));
    }

    #[rstest]
    #[case::plan(QueryPayload::Plan(vec![]), "plan", None)]
    #[case::sql(QueryPayload::Sql("select 1".into()), "sql", None)]
    #[case::request(
        QueryPayload::Request { kind: "k".into(), body: vec![] },
        "request",
        Some("k".to_string())
    )]
    fn a_payload_names_its_lane(
        #[case] payload: QueryPayload,
        #[case] expected_lane: &str,
        #[case] expected_kind: Option<String>,
    ) {
        assert_eq!(lane(&payload), (expected_lane, expected_kind));
    }
}
