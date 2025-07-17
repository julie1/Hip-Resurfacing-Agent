from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import tiktoken  # Add this import for token counting

import streamlit as st
import json
import logfire
from qdrant_client import QdrantClient
from openai import AsyncOpenAI

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
from hip_agent_qdrant import hip_agent_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
llm_model = st.secrets.get("LLM_MODEL", "gpt-4o-mini")
TOKEN_LIMIT = 100000  # Set a conservative token limit (below the 128K max)

openai_client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def count_tokens(messages):
    """
    Count tokens in a list of messages.
    """
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
        # Fallback to approximate counting if tiktoken fails
        st.error(f"Error counting tokens: {e}")
        total_chars = sum(len(part.content) for msg in messages 
                       for part in msg.parts 
                       if isinstance(msg, (ModelRequest, ModelResponse)) and hasattr(part, 'content'))
        return total_chars // 4  # Rough approximation: 1 token ≈ 4 characters


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


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Check if we need to clear the conversation due to token limits
    current_tokens = count_tokens(st.session_state.messages)
    if current_tokens > TOKEN_LIMIT:
        clear_conversation(auto=True)
        # Add a notification that conversation was cleared
        st.session_state.conversation_cleared = True
        st.info("Previous conversation history has been cleared to optimize performance. Your question will still receive an accurate answer.")
    
    # Prepare dependencies
    deps = PydanticAIDeps(
        qdrant_client=qdrant_client,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with hip_agent_expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
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
    st.write("""Ask any question to retrieve information from the archived posts of members of the hip resurfacing support group
    https://groups.io/g/Hipresurfacingsite. Please join the group to post new messages.  Also see https://www.hipresurfacingsite.com/.""")

    # Add clear conversation button in the sidebar
    with st.sidebar:
        st.title("Options")
        if st.button("Clear Conversation"):
            clear_conversation()
        
        # Add token usage indicator
        if "messages" in st.session_state and st.session_state.messages:
            current_tokens = count_tokens(st.session_state.messages)
            st.progress(min(1.0, current_tokens / TOKEN_LIMIT))
            st.caption(f"Token usage: {current_tokens}/{TOKEN_LIMIT}")

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
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about hip resurfacing?")

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
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
