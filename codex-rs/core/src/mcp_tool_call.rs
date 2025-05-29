use std::time::Duration;

use tracing::{error, warn};

use mcp_types::{
    CallToolResult, CallToolResultContent, EmbeddedResourceResource, ImageContent,
    ReadResourceResultContents,
};

use crate::codex::Session;
use crate::models::FunctionCallOutputPayload;
use crate::models::ResponseInputItem;
use crate::protocol::Event;
use crate::protocol::EventMsg;
use crate::protocol::McpToolCallBeginEvent;
use crate::protocol::McpToolCallEndEvent;

/// Timeout when fetching resources referenced by tool call results.
const READ_RESOURCE_TIMEOUT: Duration = Duration::from_secs(10);

/// Handles the specified tool call dispatches the appropriate
/// `McpToolCallBegin` and `McpToolCallEnd` events to the `Session`.
pub(crate) async fn handle_mcp_tool_call(
    sess: &Session,
    sub_id: &str,
    call_id: String,
    server: String,
    tool_name: String,
    arguments: String,
    timeout: Option<Duration>,
) -> ResponseInputItem {
    // Parse the `arguments` as JSON. An empty string is OK, but invalid JSON
    // is not.
    let arguments_value = if arguments.trim().is_empty() {
        None
    } else {
        match serde_json::from_str::<serde_json::Value>(&arguments) {
            Ok(value) => Some(value),
            Err(e) => {
                error!("failed to parse tool call arguments: {e}");
                return ResponseInputItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output: FunctionCallOutputPayload {
                        content: format!("err: {e}"),
                        success: Some(false),
                    },
                };
            }
        }
    };

    let tool_call_begin_event = EventMsg::McpToolCallBegin(McpToolCallBeginEvent {
        call_id: call_id.clone(),
        server: server.clone(),
        tool: tool_name.clone(),
        arguments: arguments_value.clone(),
    });
    notify_mcp_tool_call_event(sess, sub_id, tool_call_begin_event).await;

    // Perform the tool call.
    let result = sess
        .call_tool(&server, &tool_name, arguments_value, timeout)
        .await
        .map_err(|e| format!("tool call error: {e}"));

    let event_result = match &result {
        Ok(res) => match inline_image_resource(sess, &server, res).await {
            Some(inlined) => Ok(inlined),
            None => Ok(res.clone()),
        },
        Err(e) => Err(e.clone()),
    };

    let tool_call_end_event = EventMsg::McpToolCallEnd(McpToolCallEndEvent {
        call_id: call_id.clone(),
        result: event_result,
    });

    notify_mcp_tool_call_event(sess, sub_id, tool_call_end_event.clone()).await;

    ResponseInputItem::McpToolCallOutput { call_id, result }
}

async fn inline_image_resource(
    sess: &Session,
    server: &str,
    result: &CallToolResult,
) -> Option<CallToolResult> {
    let first = result.content.first()?;
    let CallToolResultContent::EmbeddedResource(embedded) = first else {
        return None;
    };

    let EmbeddedResourceResource::BlobResourceContents(blob) = &embedded.resource else {
        return None;
    };

    let mime_type = blob
        .mime_type
        .as_deref()
        .filter(|m| m.starts_with("image/"))?;

    let read_res = match sess
        .read_resource(server, blob.uri.clone(), Some(READ_RESOURCE_TIMEOUT))
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("failed to fetch image resource: {e}");
            return None;
        }
    };

    let Some(ReadResourceResultContents::BlobResourceContents(contents)) =
        read_res.contents.into_iter().next()
    else {
        return None;
    };

    Some(CallToolResult {
        content: vec![CallToolResultContent::ImageContent(ImageContent {
            annotations: embedded.annotations.clone(),
            data: contents.blob,
            mime_type: contents.mime_type.unwrap_or_else(|| mime_type.to_string()),
            r#type: "image".to_string(),
        })],
        is_error: result.is_error,
    })
}

async fn notify_mcp_tool_call_event(sess: &Session, sub_id: &str, event: EventMsg) {
    sess.send_event(Event {
        id: sub_id.to_string(),
        msg: event,
    })
    .await;
}
