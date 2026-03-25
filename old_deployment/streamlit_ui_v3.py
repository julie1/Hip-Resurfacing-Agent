from __future__ import annotations
from typing import Literal, TypedDict, Dict, Any, Optional
import asyncio
import os
import tiktoken
import uuid
from datetime import datetime

import streamlit as st
import json
import logfire
from qdrant_client import QdrantClient
from openai import AsyncOpenAI
from pinecone import Pinecone
import csv
#import datetime
from pathlib import Path
import json
from io import StringIO

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Import the updated Langfuse configuration and dashboard
from configure_langfuse_v3 import configure_langfuse
from langfuse_dashboard_v3 import LangfuseDashboard, FeedbackCollector

# Import the combined agent
from combine_agents_langfuse import hip_rag_agent, CombinedDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
llm_model = st.secrets.get("LLM_MODEL", "gpt-4o-mini")
TOKEN_LIMIT = 100000  # Set a conservative token limit

# Initialize clients
openai_client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

# Initialize Pinecone client
pinecone_client = Pinecone(
    api_key=st.secrets["PINECONE_API_KEY"]
)

# Configure logfire
logfire.configure(send_to_logfire='never')

# Initialize Langfuse ONCE and reuse it
langfuse_config = configure_langfuse()
langfuse_dashboard = LangfuseDashboard(langfuse_config)
feedback_collector = FeedbackCollector(langfuse_config, langfuse_dashboard)
ENABLE_RAG_EVALUATION = os.getenv('ENABLE_RAG_EVALUATION', 'true').lower() == 'true'

# Initialize session state for tracking user sessions
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = f"anonymous_{uuid.uuid4().hex[:8]}"
if "langfuse_trace_id" not in st.session_state:
    st.session_state.langfuse_trace_id = {}

# Start an initial trace for the session and store it
initial_trace_id = feedback_collector.start_interaction(st.session_state.session_id)
st.session_state.langfuse_trace_id = {}

# Start an initial trace for the session and store it
initial_trace_id = feedback_collector.start_interaction(st.session_state.session_id)
st.session_state.langfuse_trace_id[st.session_state.session_id] = initial_trace_id


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str


def log_feedback_to_csv(session_id: str, feedback_type: str, score: int,
                       normalized_score: float, comment: str = "",
                       trace_id: str = "", source: str = ""):
    """Simple function to log feedback to CSV"""
    csv_path = "feedback_log.csv"

    print(f"🔍 CSV logging attempt: {feedback_type}, score: {score}, source: {source}")

    try:
        # Check if file exists
        file_exists = os.path.exists(csv_path)
        print(f"🔍 CSV file exists: {file_exists}")

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    'timestamp', 'session_id', 'trace_id', 'feedback_type',
                    'score_1_to_5', 'score_normalized', 'comment', 'source'
                ])
                print("🔍 CSV header written")

            # Write feedback data
            row_data = [
                datetime.now().isoformat(),
                session_id,
                trace_id,
                feedback_type,
                score,
                normalized_score,
                comment,
                source
            ]

            writer.writerow(row_data)
            print(f"🔍 CSV row written: {row_data}")

        print(f"✅ Feedback logged to CSV: {os.path.abspath(csv_path)}")
        return True

    except Exception as e:
        print(f"❌ CSV logging failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def render_sidebar_feedback(st, feedback_collector, session_id, selected_source):
    """Render feedback UI in sidebar with proper reset functionality"""

    st.markdown("---")
    st.subheader("Feedback")

    # Initialize feedback submission counter
    if "feedback_submissions" not in st.session_state:
        st.session_state.feedback_submissions = 0

    # Show submission count if > 0
    if st.session_state.feedback_submissions > 0:
        st.caption(f"✅ {st.session_state.feedback_submissions} feedback(s) submitted this session")

    # Use session state to control form values instead of relying on widget reset
    if "current_feedback_type" not in st.session_state:
        st.session_state.current_feedback_type = 0
    if "current_feedback_score" not in st.session_state:
        st.session_state.current_feedback_score = 4
    if "current_feedback_comment" not in st.session_state:
        st.session_state.current_feedback_comment = ""

    st.write("Your feedback helps us improve the quality of answers.")

    # Initialize feedback form state
    if "feedback_form_key" not in st.session_state:
        st.session_state.feedback_form_key = 0

    # Use form for proper reset
    form_key = f"feedback_form_{st.session_state.feedback_form_key}"

    with st.form(key=form_key):
        # Feedback type selection
        feedback_options = ["Relevance", "Accuracy", "Helpfulness", "Overall Experience"]
        feedback_type_index = st.selectbox(
            "What aspect would you like to rate?",
            options=range(len(feedback_options)),
            format_func=lambda x: feedback_options[x],
            index=0
        )

        # Rating slider
        score = st.slider(
            "Rating",
            min_value=1,
            max_value=5,
            value=st.session_state.current_feedback_score,
            help="1 = Poor, 5 = Excellent",
            key="rating_slider_sidebar"
        )

        # Comment text area
        comment = st.text_area(
            "Additional Comments (Optional)",
            value=st.session_state.current_feedback_comment,
            height=100,
            key="comment_sidebar"
        )

        # Submit button - CHANGED: Use st.form_submit_button() instead of st.button()
        submitted = st.form_submit_button("Submit Feedback")

    # Handle form submission outside the form context
    if submitted:
        # Get selected feedback type name
        feedback_type_name = feedback_options[feedback_type_index]
        normalized_score = score / 5.0

        # Get current trace ID
        current_trace_id = st.session_state.langfuse_trace_id.get(session_id, "")

        # Log to CSV
        csv_success = log_feedback_to_csv(
            session_id=session_id,
            feedback_type=feedback_type_name,
            score=score,
            normalized_score=normalized_score,
            comment=comment,
            trace_id=current_trace_id,
            source=selected_source
        )

        # Log to Langfuse (original functionality)
        langfuse_success = False
        try:
            feedback_collector.record_feedback(
                session_id=session_id,
                score_name=feedback_type_name.lower(),
                score_value=normalized_score,
                comment=comment,
                trace_id=current_trace_id
            )
            langfuse_success = True
        except Exception as e:
            print(f"Langfuse feedback failed: {e}")

        # Show success message
        if csv_success and langfuse_success:
            st.success(f"✅ Feedback recorded! You rated {feedback_type_name.lower()} as {score}/5.")
        elif csv_success:
            st.success(f"✅ Feedback recorded locally! You rated {feedback_type_name.lower()} as {score}/5.")
            st.warning("⚠️ Langfuse logging failed, but feedback saved locally.")
        else:
            st.error("❌ Failed to record feedback")

        # Reset form values in session state
        st.session_state.current_feedback_type = 0
        st.session_state.current_feedback_score = 4
        st.session_state.current_feedback_comment = ""

        # Increment submission counter
        st.session_state.feedback_submissions += 1

        # Reset the form by incrementing the key
        st.session_state.feedback_form_key += 1

        # Force rerun to reset form
        st.rerun()





def count_tokens(messages):
    """Count tokens in a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(llm_model)
        total_tokens = 0

        for msg in messages:
            if isinstance(msg, (ModelRequest, ModelResponse)):
                for part in msg.parts:
                    if hasattr(part, 'content'):
                        total_tokens += len(encoding.encode(part.content))

        return total_tokens
    except Exception as e:
        # Fallback to approximate counting
        st.error(f"Error counting tokens: {e}")
        total_chars = sum(len(part.content) for msg in messages
                       for part in msg.parts
                       if isinstance(msg, (ModelRequest, ModelResponse)) and hasattr(part, 'content'))
        return total_chars // 4  # Rough approximation

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

def check_pinecone_status():
    """Check if Pinecone is connected and the forum-pages index exists with hip-forum namespace."""
    try:
        st.session_state.langfuse_trace_id = {}

# Start an initial trace for the session and store it
        initial_trace_id = feedback_collector.start_interaction(st.session_state.session_id)
        print("Starting Pinecone status check...")

        # Create a trace for this operation
        with langfuse_config.langfuse.start_as_current_span(
            name="dashboard_load"
        ) as span:
            span.update(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                metadata={"type": "dashboard_load"}
            )

        try:
            deps = CombinedDeps(
                qdrant_client=qdrant_client,
                pinecone_client=pinecone_client,
                openai_client=openai_client,
                search_source="pinecone"
            )

            # Try to access index directly
            index = deps.pinecone_client.Index("forum-pages")
            stats = index.describe_index_stats()

            # If we got here, the index exists
            print("Successfully accessed index 'forum-pages' directly")

            # Check for namespace
            namespaces = stats.get('namespaces', {})
            print(f"Available namespaces: {list(namespaces.keys())}")

            has_namespace = "hip-forum" in namespaces
            print(f"Namespace 'hip-forum' exists: {has_namespace}")

            span.update(
                status="success",
                metadata={
                    "index_found": True,
                    "has_namespace": has_namespace,
                    "namespaces": list(namespaces.keys())
                }
            )
            span.end()

            # Return True only if both index and namespace exist
            return has_namespace

        except Exception as direct_err:
            print(f"Failed to access index directly: {direct_err}")

            # Try fallback methods
            all_indexes = deps.pinecone_client.list_indexes()
            print(f"Raw indexes response: {all_indexes}")

            # Try multiple approaches to find the index
            index_found = False

            # Approach 1: Check if it's a list of dicts with 'name' field
            if isinstance(all_indexes, list) and all(isinstance(idx, dict) for idx in all_indexes if idx):
                index_names = [idx.get('name') for idx in all_indexes if idx and 'name' in idx]
                print(f"Extracted index names (approach 1): {index_names}")
                index_found = "forum-pages" in index_names

            # Approach 2: Check if it's a list of strings
            elif isinstance(all_indexes, list) and all(isinstance(idx, str) for idx in all_indexes if idx):
                print(f"Index names (approach 2): {all_indexes}")
                index_found = "forum-pages" in all_indexes

            # Approach 3: Check if it's a dict with keys as index names
            elif isinstance(all_indexes, dict):
                print(f"Index dict keys (approach 3): {list(all_indexes.keys())}")
                index_found = "forum-pages" in all_indexes

            if not index_found:
                print("Index 'forum-pages' not found through any check method")
                span.update(
                    status="error",
                    metadata={"error": "Index not found", "direct_error": str(direct_err)}
                )
                span.end()
                return False

            # Try again with direct access if we found it in the list
            try:
                index = deps.pinecone_client.Index("forum-pages")
                stats = index.describe_index_stats()
                namespaces = stats.get('namespaces', {})
                has_namespace = "hip-forum" in namespaces

                span.update(
                    status="success",
                    metadata={
                        "recovered": True,
                        "has_namespace": has_namespace
                    }
                )
                span.end()
                return has_namespace

            except Exception as retry_err:
                print(f"Failed on retry: {retry_err}")
                span.update(
                    status="error",
                    metadata={
                        "direct_error": str(direct_err),
                        "retry_error": str(retry_err)
                    }
                )
                span.end()
                return False

    except Exception as e:
        print(f"Pinecone connection error: {e}")
        import traceback
        traceback.print_exc()

        # Update span status and metadata
        span.update(
            status="error",
            metadata={"error": str(e)}
        )
        span.end()

        return False

def check_qdrant_status():
    """Check if Qdrant is connected and the site_pages collection exists."""
    try:
        # Create a trace for this check
        with langfuse_config.langfuse.start_as_current_span(
            name="dashboard_init"
        ) as span:
            span.update(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                metadata={"type": "dashboard_init"}
            )

            try:
                # Attempt to get the collection
                result = qdrant_client.get_collection(collection_name="site_pages")

                span.update(
                    status="success",
                    metadata={"collection_found": True}
                )
                return True

            except Exception as e:
                # If there's an error, log it in the span and end it
                error_message = str(e)
                print(f"Qdrant connection error: {error_message}")

                span.update(
                    status="error",
                    metadata={
                        "error": error_message,
                        "collection_found": False
                    }
                )
                return False

            finally:
                span.end()

    except Exception as langfuse_error:
        # Handle any issues with Langfuse itself
        print(f"Langfuse error during Qdrant status check: {langfuse_error}")

        # Still try to check Qdrant even if Langfuse fails
        try:
            qdrant_client.get_collection(collection_name="site_pages")
            return True
        except Exception as e:
            print(f"Qdrant connection error: {e}")
            return False

def clear_conversation(auto=False):
    """
    Clear the conversation history.
    If auto=True, this was triggered automatically due to token limits.
    """
    with langfuse_config.langfuse.start_as_current_span(
        name="clear_conversation"
    ) as span:
        span.update(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            metadata={"auto_triggered": auto}
        )
        trace_id = langfuse_config.langfuse.get_current_trace_id()

        st.session_state.messages = []
        if auto:
            # Add a system message explaining the reset
            st.session_state.messages.append(
                ModelResponse(parts=[
                    SystemPromptPart(content="Conversation has been reset due to token limits. You can continue asking questions.")
                ])
            )

        # Only perform rerun for manual clears to avoid disturbing the UI during auto clears
        if not auto:
            st.rerun()

async def run_agent_with_streaming(user_input: str, search_source: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Debug log for the source
    print(f"=== RUNNING AGENT WITH SOURCE: {search_source} ===")

    # Create a root span for this query
    with langfuse_config.langfuse.start_as_current_span(
        name="dashboard_root"
    ) as root_span:
        trace = root_span
        span = root_span
        span.update(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            metadata={"type": "dashboard_root"}
        )
        trace_id = root_span.trace_id

    # Register this trace with the feedback collector for later reference
    feedback_collector.register_trace(st.session_state.session_id, trace_id)

    # Store the trace ID in session state for future reference
    st.session_state.langfuse_trace_id[st.session_state.session_id] = trace_id

    # Create a span for token counting
    with langfuse_config.langfuse.start_as_current_span(
        name="token_counting"
    ) as token_span:
        # Check if we need to clear the conversation due to token limits
        current_tokens = count_tokens(st.session_state.messages)

        # End the token counting span
        token_span.update(
            metadata={"token_count": current_tokens, "token_limit": TOKEN_LIMIT}
        )
        token_span.end()

    if current_tokens > TOKEN_LIMIT:
        # Create a span for conversation clearing
        with langfuse_config.langfuse.start_as_current_span(
            name="conversation_clearing"
        ) as clear_span:
            clear_conversation(auto=True)
            # Add a notification that conversation was cleared
            st.session_state.conversation_cleared = True
            st.info("Previous conversation history has been cleared to optimize performance. Your question will still receive an accurate answer.")

            clear_span.update(
                metadata={"reason": "token_limit_exceeded"}
            )
            clear_span.end()

    # Add source context to the user input
    source_context = f"The user is querying from the {search_source} source. Only use information from this source."
    enhanced_prompt = f"{source_context}\n\nUser query: {user_input}"

    # Create a span for agent execution (including evaluation)
    with langfuse_config.langfuse.start_as_current_span(
        name="agent_execution"
    ) as agent_span:
        partial_text = ""

        try:

            # Prepare dependencies
            deps = CombinedDeps(
                qdrant_client=qdrant_client,
                pinecone_client=pinecone_client,
                openai_client=openai_client,
                search_source=search_source,
                langfuse_client=langfuse_config.langfuse,
                langfuse_tracer=trace,
                tracer=trace
            )


            # Use the async context manager properly with async with
            async with hip_rag_agent.run_stream(
                enhanced_prompt,
                deps=deps,
                message_history=st.session_state.messages[:-1],
            ) as result:
                message_placeholder = st.empty()

                # Stream the text chunks as they come in
                async for chunk in result.stream_text(delta=True):
                    partial_text += chunk
                    message_placeholder.markdown(partial_text)

                # Access new_messages() on the result object directly
                # Add new messages from this run, excluding user-prompt messages
                filtered_messages = [msg for msg in result.new_messages()
                                   if not (hasattr(msg, 'parts') and
                                          any(part.part_kind == 'user-prompt' for part in msg.parts))]
                st.session_state.messages.extend(filtered_messages)

                # Add the final response to the messages
                st.session_state.messages.append(
                    ModelResponse(parts=[TextPart(content=partial_text)])
                )

            # Run evaluation BEFORE ending the agent span
            if ENABLE_RAG_EVALUATION:
                await _run_post_response_evaluation_as_tool(
                    trace=trace,
                    agent_span=agent_span,
                    deps=deps,
                    user_query=user_input,
                    llm_response=partial_text
                )

            # End the agent span with success (after evaluation completes)
            agent_span.update(
                status="success",
                output=partial_text[:1000] if partial_text else "",
                metadata={
                    "response_length": len(partial_text) if partial_text else 0,
                    "evaluation_enabled": ENABLE_RAG_EVALUATION
                }
            )
            agent_span.end()

        except Exception as e:
            error_message = f"Error during agent execution: {str(e)}"
            print(error_message)

            # End the span with error
            agent_span.update(
                status="error",
                metadata={"error": str(e)}
            )
            agent_span.end()

            # Update trace with error status
            trace.update(
                output={"error": str(e)},
                level="ERROR",
                status_message=error_message
            )

            # Display error to user
            st.error("An error occurred while processing your request. Please try again.")

            import traceback
            traceback.print_exc()






async def _run_post_response_evaluation_as_tool(trace, agent_span, deps, user_query: str, llm_response: str):
    """Helper function to run evaluation as a proper agent tool call with file download."""
    # Create evaluation span as a child of the agent span
    with langfuse_config.langfuse.start_as_current_span(
        name="retrieval_evaluation_tool"
    ) as evaluation_span:

        # Prepare data for download
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "trace_id": getattr(trace, 'id', None),
            "span_id": getattr(evaluation_span, 'id', None),
            "user_query": user_query,
            "llm_response": llm_response,
            "query_length": len(user_query),
            "response_length": len(llm_response)
        }

        try:
            # Call the evaluation tool through the agent - this is the correct way
            evaluation_result = await hip_rag_agent.run(
                f"Please evaluate the retrieval quality for this query and response using the evaluate_retrieval tool:\n"
                f"Query: {user_query}\n"
                f"Response: {llm_response[:500]}...",  # Truncate for the prompt
                deps=deps
            )

            # Add evaluation result to record
            evaluation_record["evaluation_result"] = str(evaluation_result)
            evaluation_record["status"] = "success"

            evaluation_span.update(
                metadata={
                    "evaluation_completed": True,
                    "query_length": len(user_query),
                    "response_length": len(llm_response)
                }
            )
            evaluation_span.end()
            print(f"✅ Evaluation completed via tool call")

        except Exception as eval_error:
            print(f"❌ Evaluation failed: {str(eval_error)}")

            # Add error to record
            evaluation_record["status"] = "error"
            evaluation_record["error"] = str(eval_error)

            evaluation_span.update(
                status="error",
                metadata={"evaluation_error": str(eval_error)}
            )
            evaluation_span.end()

        # Automatically save to local file
        try:
            # Save to evaluations.json in the same directory as your project
            filename = "evaluations.json"

            # Load existing evaluations or create empty list
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
            else:
                evaluations = []

            # Add new evaluation record
            evaluations.append(evaluation_record)

            # Save back to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(evaluations, f, indent=2, ensure_ascii=False)

            print(f"💾 Evaluation record saved to {filename}")

        except Exception as save_error:
            print(f"❌ Failed to save evaluation record: {save_error}")





async def main():
    st.title("Hip Resurfacing Search Engine")
    st.write("""Select Surface Hippy Forum to ask any question to retrieve information from the archived posts of members of the hip resurfacing support group
    https://surfacehippy.info/hiptalk/.   Select Hip Resurfacing Forum to retrieve information from https://groups.io/g/Hipresurfacingsite archived posts.
    Select Both Sources to query both archives at the same time.   Please join the groups to post new messages.""")

    with st.sidebar:
        st.title("Options")

        # Add source selection
        data_source = st.radio(
            "Select data source:",
            ["Surface Hippy Forum (Pinecone)",
             "Hip Resurfacing Group Messages (Qdrant)",
             "Both Sources"],
            index=0
        )

        # Map UI selection to code values
        source_mapping = {
            "Surface Hippy Forum (Pinecone)": "pinecone",
            "Hip Resurfacing Group Messages (Qdrant)": "qdrant",
            "Both Sources": "both"
        }

        selected_source = source_mapping[data_source]
        print(f"DEBUG: Selected source from UI: {data_source} -> mapped to: {selected_source}")

        # Clear conversation button with a unique key
        if st.button("Clear Conversation", key="clear_conversation_btn"):
            clear_conversation()

        # Add token usage indicator
        if "messages" in st.session_state and st.session_state.messages:
            current_tokens = count_tokens(st.session_state.messages)
            st.progress(min(1.0, current_tokens / TOKEN_LIMIT))
            st.caption(f"Token usage: {current_tokens}/{TOKEN_LIMIT}")

        # Add status indicators for both sources
        st.subheader("Data Source Status")

        # Check Qdrant status
        qdrant_status = "Connected" if check_qdrant_status() else "Disconnected"
        st.markdown(f"**Qdrant**: {qdrant_status}")

        # Check Pinecone status
        pinecone_status = "Connected" if check_pinecone_status() else "Disconnected"
        st.markdown(f"**Pinecone**: {pinecone_status}")

        render_sidebar_feedback(st, feedback_collector, st.session_state.session_id, selected_source)


    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize conversation_cleared flag if not present
    if "conversation_cleared" not in st.session_state:
        st.session_state.conversation_cleared = False

    # Show notification if conversation was cleared
    if st.session_state.conversation_cleared:
        st.info("Previous conversation history has been cleared to optimize performance. Your new questions will still receive accurate answers.")
        # Reset the flag after displaying the message
        st.session_state.conversation_cleared = False

    # Display all messages from the conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about hip resurfacing?")

    # Only process if there's user input
    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Run the agent with the user input and selected source
            await run_agent_with_streaming(user_input, selected_source)

if __name__ == "__main__":
    asyncio.run(main())
