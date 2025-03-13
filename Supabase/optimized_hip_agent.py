from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
from functools import lru_cache
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client
from typing import List, Dict, Tuple

load_dotenv()

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

# Optimize model initialization
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

# Refined system prompt for more concise outputs
system_prompt = """
You are an expert at retrieving and summarizing information from hip resurfacing group posts.
Provide clear, focused answers in 2-3 paragraphs maximum.
Include only directly relevant information to the query.
Always note information dates when mentioning experiences or outcomes.
Do not include any system information or metadata in your responses.
Do not ask follow-up questions unless critical information is missing.
"""

hip_agent_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=1  # Reduce retries to improve speed
)

# Cache embeddings to avoid recomputing
@lru_cache(maxsize=1000)
async def get_cached_embedding(text: str) -> Tuple[float, ...]:
    """Cached version of embedding generation."""
    try:
        # Create separate client for embeddings
        async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return tuple(response.data[0].embedding)  # Convert to tuple for caching
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise

@hip_agent_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Optimized RAG retrieval with better filtering and formatting.
    """
    try:
        # Get cached embedding
        query_embedding = list(await get_cached_embedding(user_query))

        # Optimized Supabase query with better filtering
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,  # Reduced from 5
                'similarity_threshold': 0.7,  # Add minimum similarity threshold
                'filter': {'source': 'hip resurfacing group posts'}
            }
        ).execute()

        if not result.data:
            return "No relevant documentation found."

        # Improved result formatting
        formatted_chunks = []
        seen_content = set()  # Avoid duplicate content

        for doc in sorted(result.data,
                         key=lambda x: x.get('metadata', {}).get('started_date', ''),
                         reverse=True):

            content = doc.get('content', '').strip()
            if content and content not in seen_content:
                seen_content.add(content)
                date_info = doc.get('metadata', {}).get('started_date', 'Unknown date')

                chunk_text = f"""
Source ({date_info}):
{content}
"""
                formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks) if formatted_chunks else "No relevant content found."

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@hip_agent_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Optimized page listing with caching.
    """
    try:
        # Add caching at the database level
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'hip_messages') \
            .execute()

        if not result.data:
            return []

        # Use set for deduplication then convert to sorted list
        return sorted(set(doc['url'] for doc in result.data))

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@hip_agent_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Optimized page content retrieval with better chunk handling.
    """
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number, metadata') \
            .eq('url', url) \
            .eq('metadata->>source', 'hip_messages') \
            .order('chunk_number') \
            .execute()

        if not result.data:
            return f"No content found for URL: {url}"

        # Improved content formatting
        page_title = result.data[0]['title'].split(' - ')[0]
        metadata = result.data[0].get('metadata', {})
        date_info = metadata.get('started_date', 'Unknown date')

        formatted_content = [
            f"# {page_title}",
            f"Date: {date_info}\n"
        ]

        # Combine chunks efficiently
        content_chunks = []
        for chunk in result.data:
            chunk_content = chunk['content'].strip()
            if chunk_content:  # Skip empty chunks
                content_chunks.append(chunk_content)

        formatted_content.extend(content_chunks)

        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

# Optional: Add a response formatter to ensure consistent output
async def format_response(response: str) -> str:
    """Clean and format the response for consistency."""
    # Remove any system prompt leakage
    if "You are an expert" in response:
        response = response.split("You are an expert")[0]

    # Remove any tool calls or metadata
    if "Tool call:" in response:
        response = response.split("Tool call:")[0]

    return response.strip()
