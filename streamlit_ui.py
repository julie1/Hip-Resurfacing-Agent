from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import tiktoken

import streamlit as st
import json
import logfire
from qdrant_client import QdrantClient
from openai import AsyncOpenAI
from pinecone import Pinecone

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

# Import the combined agent instead of the original
from combine_agents import hip_rag_agent, CombinedDeps

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

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


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
        print("Starting Pinecone status check...")

        # Create temporary deps just for this check
        deps = CombinedDeps(
            qdrant_client=qdrant_client,
            pinecone_client=pinecone_client,
            openai_client=openai_client,
            search_source="pinecone"
        )

        # Direct access approach - try to directly use the index
        try:
            # Skip the check for existence and just try to access it directly
            index = deps.pinecone_client.Index("forum-pages")
            stats = index.describe_index_stats()

            # If we got here, the index exists
            print("Successfully accessed index 'forum-pages' directly")

            # Check for namespace
            namespaces = stats.get('namespaces', {})
            print(f"Available namespaces: {list(namespaces.keys())}")

            has_namespace = "hip-forum" in namespaces
            print(f"Namespace 'hip-forum' exists: {has_namespace}")

            # Return True only if both index and namespace exist
            return has_namespace
        except Exception as direct_err:
            print(f"Failed to access index directly: {direct_err}")

            # Fall back to listing and checking
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
                return False

            # Try again with direct access if we found it in the list
            try:
                index = deps.pinecone_client.Index("forum-pages")
                stats = index.describe_index_stats()
                namespaces = stats.get('namespaces', {})
                has_namespace = "hip-forum" in namespaces
                return has_namespace
            except Exception as retry_err:
                print(f"Failed on retry: {retry_err}")
                return False
    except Exception as e:
        print(f"Pinecone connection error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_qdrant_status():
    """Check if Qdrant is connected and the site_pages collection exists."""
    try:
        qdrant_client.get_collection(collection_name="site_pages")
        return True
    except Exception as e:
        print(f"Qdrant connection error: {e}")
        return False

async def run_agent_with_streaming(user_input: str, search_source: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Debug log for the source
    print(f"=== RUNNING AGENT WITH SOURCE: {search_source} ===")

    # Check if we need to clear the conversation due to token limits
    current_tokens = count_tokens(st.session_state.messages)
    if current_tokens > TOKEN_LIMIT:
        clear_conversation(auto=True)
        # Add a notification that conversation was cleared
        st.session_state.conversation_cleared = True
        st.info("Previous conversation history has been cleared to optimize performance. Your question will still receive an accurate answer.")

    # Add source context to the user input
    source_context = f"The user is querying from the {search_source} source. Only use information from this source."
    enhanced_prompt = f"{source_context}\n\nUser query: {user_input}"

    # Prepare dependencies
    deps = CombinedDeps(
        qdrant_client=qdrant_client,
        pinecone_client=pinecone_client,
        openai_client=openai_client,
        search_source=search_source  # Pass the source as is
    )

    # Run the agent in a stream
    async with hip_rag_agent.run_stream(
        enhanced_prompt,  # Use the enhanced prompt with source context
        deps=deps,
        message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # Rest of the function stays the same...

        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages()
                            if not (hasattr(msg, 'parts') and
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )



def clear_conversation(auto=False):
    """
    Clear the conversation history.
    If auto=True, this was triggered automatically due to token limits.
    """
    st.session_state.messages = []
    if auto:
        # Add a system message explaining the reset
        st.session_state.messages.append(
            ModelResponse(parts=[
                SystemPromptPart(content="Conversation has been reset due to token limits. You can continue asking questions.")
            ])
        )
    else:
        # Manual clear by user
        st.rerun()

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
