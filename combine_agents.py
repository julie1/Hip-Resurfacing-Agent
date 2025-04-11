# combine_agents.py

from __future__ import annotations as _annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import json
import os
import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pinecone import Pinecone

load_dotenv()
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class CombinedDeps:
    qdrant_client: QdrantClient
    pinecone_client: Any  # Pinecone instance
    openai_client: AsyncOpenAI
    search_source: str = "qdrant"  # Options: "qdrant", "pinecone", "both"

class SearchResult(BaseModel):
    content: str
    similarity: float
    metadata: Dict
    title: str
    url: str

system_prompt = """You are an expert at retrieving and analyzing information from hip resurfacing resources to answer questions about hip resurfacing surgery.

Available data sources:
- Surface Hippy Forum (forum posts from SurfaceHippy)
- Hip Resurfacing Group Messages (email discussions from groups.io)

When answering questions:
1. Always ground your responses in the specific content retrieved from the sources
2. Quote relevant passages directly using markdown quotes to support your answers
3. Cite the source URL when referencing specific information
4. Express uncertainty when information is ambiguous or incomplete
5. Synthesize information from multiple posts when relevant
6. Maintain awareness of post dates to contextualize information
7. Focus only on hip resurfacing related content
8. Weight the latest dated posts over earlier ones unless the information cannot be found in the later posts

SOURCE SELECTION RULES:
- If the source is Surface Hippy Forum, ONLY use content from Surface Hippy Forum. Do not include information or posts from Hip Resurfacing Group Messages.
- If the source is Hip Resurfacing Group Messages, ONLY use content from Hip Resurfacing Group Messages. Do not include information or posts from Surface Hippy Forum.
- If the source is Both sources, include information from both sources.

Important: Don't make claims without supporting evidence. If you can't find relevant information, clearly state that limitation.
"""



hip_rag_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=CombinedDeps,
    retries=2
)

@hip_rag_agent.tool
async def pinecone_detailed_diagnostics(ctx: RunContext[CombinedDeps]) -> str:
    """Run detailed diagnostics on Pinecone connection and index structure."""
    try:
        # Get raw response
        all_indexes = ctx.deps.pinecone_client.list_indexes()

        # Detailed inspection of the response
        response_type = type(all_indexes).__name__
        response_structure = f"Type: {response_type}, Content: {json.dumps(all_indexes, default=str)}"

        # Try to access the index regardless of structure
        index_found = False
        index_details = "Not available"

        try:
            # First try to directly access the index
            index = ctx.deps.pinecone_client.Index("forum-pages")
            index_stats = index.describe_index_stats()
            index_found = True
            index_details = json.dumps(index_stats, default=str)
        except Exception as idx_err:
            index_details = f"Error accessing index: {str(idx_err)}"

        # Check for namespaces if index was found
        namespace_details = "Not available"
        if index_found:
            try:
                namespaces = index_stats.get('namespaces', {})
                namespace_details = f"Found namespaces: {list(namespaces.keys())}"

                if "hip-forum" in namespaces:
                    vector_count = namespaces.get("hip-forum", {}).get('vector_count', 0)
                    namespace_details += f"\nVector count in 'hip-forum': {vector_count}"
            except Exception as ns_err:
                namespace_details = f"Error checking namespaces: {str(ns_err)}"

        return f"""
## Detailed Pinecone Diagnostics

### Raw Response Analysis
{response_structure}

### Index "forum-pages"
Status: {"Found" if index_found else "Not Found"}
Details: {index_details}

### Namespace Information
{namespace_details}

### API Version Check
This appears to be using the {"newer" if isinstance(all_indexes, list) and all_indexes and isinstance(all_indexes[0], dict) else "older"} Pinecone API format.

### Recommendations
Try directly creating an index object with: `index = pinecone_client.Index("forum-pages")`
"""
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return f"Pinecone Diagnostics Error: {str(e)}\n\nStacktrace:\n{trace}"

@hip_rag_agent.tool
async def test_pinecone_connection(ctx: RunContext[CombinedDeps]) -> str:
    """Test the Pinecone connection and provide detailed diagnostics."""
    try:
        # Access the client through context deps
        client = ctx.deps.pinecone_client

        # Check if indexes exist
        all_indexes = client.list_indexes()
        index_info = f"Available indexes: {all_indexes}"

        if "forum-pages" not in all_indexes:
            return f"Error: Index 'forum-pages' not found. {index_info}"

        # Check if namespace exists
        index = client.Index("forum-pages")
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {})
        namespace_info = f"Available namespaces: {list(namespaces.keys())}"

        has_hip_forum = "hip-forum" in namespaces
        namespace_status = f"Namespace 'hip-forum' {'found' if has_hip_forum else 'NOT FOUND'}"

        # Get vector counts if available
        vector_count = "N/A"
        if has_hip_forum:
            vector_count = namespaces.get("hip-forum", {}).get('vector_count', 0)

        return f"""
Pinecone Connection Test Results:
- Connection status: Successful
- {index_info}
- {namespace_info}
- {namespace_status}
- Vector count in 'hip-forum': {vector_count}
- Full stats: {json.dumps(stats, indent=2)}
"""
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return f"Pinecone Connection Error: {str(e)}\n\nStacktrace:\n{trace}"

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text.replace("\n", " ")[:8000]  # Clean text and respect token limits
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Final error getting embedding: {e}")
                return [0] * 1536
            await asyncio.sleep(1 * (attempt + 1))

@hip_rag_agent.tool
async def retrieve_from_qdrant(ctx: RunContext[CombinedDeps], user_query: str) -> str:
    """Enhanced RAG retrieval from Qdrant (Hip Resurfacing Group Messages)."""
    try:
        # Get query embedding
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        # Vector search
        search_results = ctx.deps.qdrant_client.search(
            collection_name="site_pages",
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="hip_messages")
                    )
                ]
            ),
            limit=5,
            with_payload=True,
        )

        if not search_results:
            # Try text search if vector search fails
            try:
                text_search_results = ctx.deps.qdrant_client.scroll(
                    collection_name="site_pages",
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value="hip_messages")
                            ),
                            models.FieldCondition(
                                key="content",
                                match=models.MatchText(text=user_query)
                            )
                        ]
                    ),
                    limit=5,
                    with_payload=True,
                    with_vectors=False,
                )

                points, _ = text_search_results
                if not points:
                    return "No relevant documentation found in Hip Resurfacing Group Messages for your query."

                # Format text search results
                formatted_chunks = []
                for point in points:
                    chunk_text = f"""
## {point.payload.get("title", "Untitled")}
[Source]({point.payload.get("url", "No URL")})
Posted: {point.payload.get("started_date", "Date unknown")}
Last Update: {point.payload.get("most_recent_date", "Unknown")}
Found by: Text search

{point.payload.get("content", "No content available")}
"""
                    formatted_chunks.append(chunk_text)

                return "\n\n---\n\n".join(formatted_chunks)
            except Exception as text_search_error:
                print(f"Text search error: {text_search_error}")
                return "No relevant documentation found in Hip Resurfacing Group Messages for your query."

        # Format vector search results
        formatted_chunks = []
        for result in search_results:
            chunk_text = f"""
## {result.payload.get("title", "Untitled")}
[Source]({result.payload.get("url", "No URL")})
Posted: {result.payload.get("started_date", "Date unknown")}
Last Update: {result.payload.get("most_recent_date", "Unknown")}
Relevance Score: {result.score:.2f}

{result.payload.get("content", "No content available")}
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving from Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving documentation from Hip Resurfacing Group Messages: {str(e)}"


@hip_rag_agent.tool
async def retrieve_from_pinecone(ctx: RunContext[CombinedDeps], user_query: str) -> str:
    """Enhanced RAG retrieval using Pinecone (Surface Hippy Forum)."""
    try:
        print("DEBUG: Inside retrieve_from_pinecone function")

        # Get query embedding
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        # Direct access to the index
        index = ctx.deps.pinecone_client.Index("forum-pages")

        # Vector search with namespace
        search_results = index.query(
            namespace="hip-forum",
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        if not search_results['matches']:
            return "No relevant documentation found in Surface Hippy Forum for your query."

        # Format results
        formatted_chunks = []
        for match in search_results['matches']:
            metadata = match.get('metadata', {})
            chunk_text = f"""
## {metadata.get("title", "Untitled")}
[Source]({metadata.get("url", "No URL")})
Posted: {metadata.get("date", "Date unknown")}
Relevance Score: {match.get('score', 0):.2f}

{metadata.get("content", "No content available")}
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"ERROR in retrieve_from_pinecone: {e}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving documentation from Surface Hippy Forum: {str(e)}"

@hip_rag_agent.tool
async def source_selection_debug(ctx: RunContext[CombinedDeps]) -> str:
    """Debug tool to verify source selection."""
    # Get raw source value
    raw_source = ctx.deps.search_source

    # Test what happens with different normalizations
    source_lower = raw_source.lower() if isinstance(raw_source, str) else "None"

    # Check equality with different cases
    checks = {
        "raw == 'pinecone'": raw_source == 'pinecone',
        "raw == 'Pinecone'": raw_source == 'Pinecone',
        "lower == 'pinecone'": source_lower == 'pinecone',
        "raw == 'qdrant'": raw_source == 'qdrant',
        "raw == 'Qdrant'": raw_source == 'Qdrant',
        "lower == 'qdrant'": source_lower == 'qdrant',
        "raw == 'both'": raw_source == 'both',
        "raw == 'Both'": raw_source == 'Both',
        "lower == 'both'": source_lower == 'both',
    }

    # Format results
    checks_formatted = "\n".join([f"- {k}: {v}" for k, v in checks.items()])

    return f"""
# Source Selection Debug
- Raw source value: '{raw_source}'
- Type: {type(raw_source)}
- Lowercase: '{source_lower}'

## Equality Checks:
{checks_formatted}

## Recommendation:
Based on these results, in the retrieve_documentation function, you should use:
`if source.lower() == 'pinecone':` for case-insensitive matching
"""


@hip_rag_agent.tool
async def retrieve_documentation(ctx: RunContext[CombinedDeps], user_query: str) -> str:
    """Smart RAG retrieval from selected sources based on user's preference."""
    # Get the source and normalize it for consistency
    source = ctx.deps.search_source.lower() if ctx.deps.search_source else "both"
    print(f"DEBUG: Retrieving documentation with source={source}")

    if source == "pinecone":
        print("DEBUG: Querying ONLY Pinecone source")
        results = await retrieve_from_pinecone(ctx, user_query)
        return f"## Results from Surface Hippy Forum ONLY\n{results}"
    elif source == "qdrant":
        print("DEBUG: Querying ONLY Qdrant source")
        results = await retrieve_from_qdrant(ctx, user_query)
        return f"## Results from Hip Resurfacing Group Messages ONLY\n{results}"
    elif source == "both":
        print("DEBUG: Querying BOTH sources")
        # Get results from both sources
        qdrant_results = await retrieve_from_qdrant(ctx, user_query)
        pinecone_results = await retrieve_from_pinecone(ctx, user_query)

        # Combine results with source indicators
        return f"""
## Results from Hip Resurfacing Group Messages
{qdrant_results}

## Results from Surface Hippy Forum
{pinecone_results}
"""
    else:
        print(f"DEBUG: Invalid source specified: '{source}'")
        return f"Invalid search source specified: '{source}'. Please use 'qdrant', 'pinecone', or 'both'."


@hip_rag_agent.tool
async def diagnose_data_sources(ctx: RunContext[CombinedDeps]) -> str:
    """Perform comprehensive diagnostics on both data sources."""
    # Check Qdrant
    try:
        qdrant_info = ctx.deps.qdrant_client.get_collection(collection_name="site_pages")
        qdrant_count = ctx.deps.qdrant_client.count(
            collection_name="site_pages",
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="hip_messages")
                    )
                ]
            )
        )
        qdrant_status = f"Connected, {qdrant_count.count} documents"
    except Exception as e:
        qdrant_status = f"Error: {str(e)}"

    # Check Pinecone
    try:
        all_indexes = ctx.deps.pinecone_client.list_indexes()
        index_name = "forum-pages"  # Changed from forum_pages

        if index_name in all_indexes:
            index = ctx.deps.pinecone_client.Index(index_name)
            stats = index.describe_index_stats()
            namespace = "hip-forum"
            namespace_stats = stats.get('namespaces', {}).get(namespace, {})
            namespace_count = namespace_stats.get('vector_count', 0) if namespace_stats else 0
            pinecone_status = f"Connected, {namespace_count} documents in namespace '{namespace}'"
        else:
            pinecone_status = f"Error: Index '{index_name}' not found"
    except Exception as e:
        pinecone_status = f"Error: {str(e)}"

    return f"""
## Data Source Diagnostics

### Hip Resurfacing Group Messages (Qdrant)
Status: {qdrant_status}

### Surface Hippy Forum (Pinecone)
Status: {pinecone_status}

### Recommendations
- {("Use 'both' sources for comprehensive information" if "Connected" in qdrant_status and "Connected" in pinecone_status else
   "Consider using only the working source" if "Connected" in qdrant_status or "Connected" in pinecone_status else
   "Both sources have issues, please check your configuration")}
"""
