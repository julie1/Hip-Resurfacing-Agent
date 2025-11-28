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
from qdrant_client.models import QueryRequest
from qdrant_client.http import models
from pinecone import Pinecone
from configure_langfuse_v3 import configure_langfuse
import time
from datetime import datetime


load_dotenv()

# Configure Langfuse for agent observability
# Configure Langfuse for agent observability
langfuse_config = configure_langfuse()
tracer = langfuse_config.tracer
langfuse_client = langfuse_config.langfuse


llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)


logfire.configure(send_to_logfire='if-token-present')


@dataclass
class CombinedDeps:
    qdrant_client: QdrantClient
    pinecone_client: Any  # Pinecone instance
    openai_client: AsyncOpenAI
    search_source: str = "qdrant"  # Options: "qdrant", "pinecone", "both"
    langfuse_client: Optional[Any] = None  # Langfuse client
    tracer: Optional[Any] = None  # OpenTelemetry tracer
    langfuse_tracer: Optional[Any] = None  # LangfuseTracer helper



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

# Decorate hip_rag_agent with Langfuse observation
hip_rag_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=CombinedDeps,
    retries=2,
    instrument=True
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

async def get_embedding(text: str, openai_client: AsyncOpenAI, deps: CombinedDeps = None, run_context=None, trace=None) -> List[float]:
    """Get embedding vector from OpenAI with retry logic and proper tracing."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Use Langfuse v3 API
            with deps.langfuse_client.start_as_current_span(name="text_embedding") as span:
                span.update(
                    metadata={
                        "text_length": len(text),
                        "text_preview": text[:100] + "..." if len(text) > 100 else text
                    }
                )

                # Call the embedding API
                response = await openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text.replace("\n", " ")[:8000]  # Clean text and respect token limits
                )

                span.update(
                    metadata={
                        "dimensions": len(response.data[0].embedding),
                        "status": "success",
                        "attempt": attempt + 1
                    }
                )
                return response.data[0].embedding

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Final error getting embedding: {e}")
                # Create a span for the error
                with deps.langfuse_client.start_as_current_span(name="text_embedding_error") as error_span:
                    error_span.update(
                        metadata={
                            "error": str(e),
                            "attempt": attempt + 1,
                            "status": "error"
                        }
                    )
                return [0] * 1536  # Return zero vector on final failure

            print(f"Embedding attempt {attempt+1} failed: {e}. Retrying...")
            await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff




@hip_rag_agent.tool
async def retrieve_from_qdrant(ctx: RunContext[CombinedDeps], user_query: str) -> str:
    """Enhanced RAG retrieval from Qdrant (Hip Resurfacing Group Messages) with date prioritization."""
    try:
        print("DEBUG: Inside retrieve_from_qdrant function")

        # Start a new span for the retrieval operation
        with ctx.deps.langfuse_client.start_as_current_span(name="qdrant_retrieval") as span:
            span.update(
                metadata={
                    "langfuse.user.id": "user",
                    "source": "qdrant",
                    "query": user_query
                }
            )

            # Get query embedding with trace
            with ctx.deps.langfuse_client.start_as_current_span(name="get_embedding_wrapper") as embedding_wrapper_span:
                query_embedding = await get_embedding(text=user_query,
                openai_client=ctx.deps.openai_client,
                deps=ctx.deps)

            # 1. Standard vector search (for relevance)
            with ctx.deps.langfuse_client.start_as_current_span(name="standard_vector_search") as search_span:
                search_span.update(
                    metadata={
                        "namespace": "hip_pages",
                        "top_k": 5
                    }
                )

                # Vector search
                search_results = ctx.deps.qdrant_client.query_points(
                    collection_name="site_pages",
                    query=query_embedding,  # Note: 'query' not 'query_vector'
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
                search_span.update(metadata={"match_count": len(search_results.points)})


            # 2. Search for recent documents (2024-2025)
            recent_results = []
            years_to_check = ["2025", "2024"]
            for year in years_to_check:
                with ctx.deps.langfuse_client.start_as_current_span(name=f"year_search_{year}") as year_span:
                    year_span.update(
                        metadata={
                            "year": year
                        }
                    )
                    try:
                        # Get a larger sample for date filtering
                        year_results = ctx.deps.qdrant_client.query_points(
                            collection_name="site_pages",
                            query=query_embedding,
                            query_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="source",
                                        match=models.MatchValue(value="hip_messages")
                                    ),
                                    # Filter by year using text match with a prefix
                                    models.FieldCondition(
                                        key="started_date",
                                        match=models.MatchText(text=f"{year}")
                                    )
                                ]
                            ),
                            limit=10,  # Increased for better coverage
                            with_payload=True,
                        )
                        print(f"Found {len(year_results.points)} matches for {year}")  # Fixed here
                        year_span.update(
                            metadata={
                                "matches_found": len(year_results.points)
                            }
                        )
                        recent_results.extend(year_results.points)
                    except Exception as e:
                        print(f"Error filtering for {year}: {e}")
                        year_span.update(
                            metadata={
                                "filter_result": "failed",
                                "error": str(e)
                            }
                        )
            

        # Combine and deduplicate results
        all_results = {}

        # Add standard results
        for result in search_results.points:
            if hasattr(result, 'id'):
                all_results[result.id] = result

        # Add recent results, prioritizing them over standard results
        for result in recent_results:
            if hasattr(result, 'id'):
                all_results[result.id] = result

        # Convert back to list and sort by date
        combined_results = list(all_results.values())

        # Sort by date if available
        combined_results.sort(
            key=lambda x: x.payload.get("started_date", "0000-00-00"),
            reverse=True  # Most recent first
        )

        # Update span with result count
        span.update(
            metadata={
                "total_results": len(combined_results)
            }
        )

        # If no results found, try text search as fallback
        if not combined_results:
            with ctx.deps.langfuse_client.start_as_current_span(name="text_search_fallback") as text_search_span:
                text_search_span.update(
                    metadata={
                        "query": user_query
                    }
                )

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
                    text_search_span.update(metadata={"result_count": len(points)})

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
                    print("Text search error: %s" % text_search_error)
                    return "No relevant documentation found in Hip Resurfacing Group Messages for your query."

        # Format results
        with ctx.deps.langfuse_client.start_as_current_span(name="format_results") as format_span:
            formatted_chunks = []
            for result in combined_results[:7]:  # Limit to 7 most relevant/recent results
                score = float(result.payload.get('score', 0.0))
                chunk_text = """
## %s
[Source](%s)
Posted: %s
Last Update: %s
Relevance Score: %.2f

%s
""" % (
                    result.payload.get("title", "Untitled"),
                    result.payload.get("url", "No URL"),
                    result.payload.get("started_date", "Date unknown"),
                    result.payload.get("most_recent_date", "Unknown"),
                    score,
                    result.payload.get("content", "No content available")
                )
                formatted_chunks.append(chunk_text)

            return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print("Error retrieving from Qdrant: %s" % e)
        print(f"Error retrieving from Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving documentation from Hip Resurfacing Group Messages: {str(e)}"



@hip_rag_agent.tool
async def retrieve_from_pinecone(ctx: RunContext[CombinedDeps], user_query: str) -> str:
    """Enhanced RAG retrieval using Pinecone with proper date handling."""
    try:
        print("DEBUG: Inside retrieve_from_pinecone function")

        # Start a new span for the retrieval operation using Langfuse v3
        with ctx.deps.langfuse_client.start_as_current_span(name="pinecone_retrieval") as span:
            # Update span with input data (Langfuse v3 way)
            span.update(
                input={
                    "user_query": user_query,
                    "source": "pinecone"
                },
                metadata={
                    "operation": "retrieval_started",
                    "user_id": "user"
                }
            )

            # Get the embedding for the query
            with ctx.deps.langfuse_client.start_as_current_span(name="get_embedding_for_retrieval") as embedding_span:
                embedding_span.update(
                    input={"text": user_query},
                    metadata={"operation": "embedding_generation"}
                )

                # Call get_embedding with Langfuse tracer
                query_embedding = await get_embedding(
                    text=user_query,
                    openai_client=ctx.deps.openai_client,
                    deps=ctx.deps
                )

                embedding_span.update(
                    output={"embedding_length": len(query_embedding) if query_embedding else 0},
                    metadata={"status": "completed"}
                )

            # Direct access to the index
            with ctx.deps.langfuse_client.start_as_current_span(name="index_access") as index_span:
                index = ctx.deps.pinecone_client.Index("forum-pages")
                index_span.update(
                    metadata={"index_name": "forum-pages", "operation": "index_access"}
                )

            # 1. Standard vector search (for relevance)
            with ctx.deps.langfuse_client.start_as_current_span(name="standard_vector_search") as search_span:
                search_span.update(
                    input={
                        "namespace": "hip-forum",
                        "top_k": 7
                    },
                    metadata={"search_type": "standard_vector"}
                )

                standard_results = index.query(
                    namespace="hip-forum",
                    vector=query_embedding,
                    top_k=7,
                    include_metadata=True
                )

                search_span.update(
                    output={"matches_found": len(standard_results.get('matches', []))},
                    metadata={"status": "completed"}
                )

            # 2. Search for recent documents (2024-2025)
            recent_results = []
            years_to_check = ["2025", "2024"]
            for year in years_to_check:
                with ctx.deps.langfuse_client.start_as_current_span(name=f"year_search_{year}") as year_span:
                    year_span.update(
                        input={"year": year, "filter_type": "post_query_filter"},
                        metadata={"operation": "year_specific_search"}
                    )
                    try:
                        # Get more results without date filter
                        all_results = index.query(
                            namespace="hip-forum",
                            vector=query_embedding,
                            top_k=50,  # Get more to filter from
                            include_metadata=True
                        )

                        # Filter by year in Python
                        from datetime import datetime

                        # Filter by year in Python - most reliable

                        year_matches = []
                        for match in all_results.get('matches', []):
                            started_date = match.get('metadata', {}).get('started_date', '')
                            # Check if date starts with year followed by a dash
                            if started_date.startswith(f"{year}-"):
                                year_matches.append(match)
                                if len(year_matches) >= 10:
                                    break



                        matches_count = len(year_matches)
                        print(f"Found {matches_count} matches for {year}")
                        year_span.update(
                            output={"matches_found": matches_count},
                            metadata={"status": "success"}
                        )
                        recent_results.extend(year_matches)

                    except Exception as e:
                        print(f"Error filtering for {year}: {e}")
                        year_span.update(
                            output={"error": str(e)},
                            metadata={"status": "error"}
                        )
            # recent_results = []
            # years_to_check = ["2025", "2024"]
            #
            # for year in years_to_check:
            #     with ctx.deps.langfuse_client.start_as_current_span(name=f"year_search_{year}") as year_span:
            #         year_span.update(
            #             input={"year": year, "filter_type": "date_regex"},
            #             metadata={"operation": "year_specific_search"}
            #         )
            #
            #         try:
            #             # Get a larger sample for date filtering
            #             # Use Pinecone's native filter syntax
            #             year_results = index.query(
            #                 namespace="hip-forum",
            #                 vector=query_embedding,
            #                 filter={"started_date": {"$regex": f"^{year}"}},
            #                 top_k=10,  # Increased for better coverage
            #                 include_metadata=True
            #             )
            #
            #             matches_count = len(year_results.get('matches', []))
            #             print(f"Found {matches_count} matches for {year}")
            #
            #             year_span.update(
            #                 output={"matches_found": matches_count},
            #                 metadata={"status": "success"}
            #             )
            #             recent_results.extend(year_results.get('matches', []))
            #
            #         except Exception as e:
            #             print(f"Error filtering for {year}: {e}")
            #             year_span.update(
            #                 output={"error": str(e)},
            #                 metadata={"status": "error"}
            #             )

            # Combine results, removing duplicates
            with ctx.deps.langfuse_client.start_as_current_span(name="combine_results") as combine_span:
                all_matches = {}

                # Add standard results
                for match in standard_results.get('matches', []):
                    if 'id' in match:
                        all_matches[match['id']] = match

                # Add recent results, prioritizing them over standard results
                for match in recent_results:
                    if 'id' in match:
                        all_matches[match['id']] = match

                # Convert back to list and sort by date
                combined_results = list(all_matches.values())

                # Sort by date if available
                combined_results.sort(
                    key=lambda x: x.get('metadata', {}).get('started_date', '0000-00-00'),
                    reverse=True  # Most recent first
                )

                combine_span.update(
                    output={"total_results": len(combined_results)},
                    metadata={
                        "standard_results": len(standard_results.get('matches', [])),
                        "recent_results": len(recent_results),
                        "unique_results": len(combined_results)
                    }
                )

            # Format results
            result = "No relevant documentation found in Surface Hippy Forum for your query."
            if combined_results:
                formatted_chunks = []
                for match in combined_results[:7]:  # Limit to 7 most relevant results
                    metadata = match.get('metadata', {})
                    date_info = metadata.get("started_date", "Unknown date")
                    score = float(match.get('score', 0.0))
                    title = metadata.get("title", "Untitled")
                    url = metadata.get("url", "No URL")
                    content = metadata.get("content", "No content available")
                    most_recent_date = metadata.get("most_recent_date", "Date unknown")

                    chunk_text = "## " + title + "\n" + \
                                  "[Source](" + url + ")\n" + \
                                  "Posted: " + date_info + "\n" + \
                                  "Last Updated: " + most_recent_date + "\n" + \
                                  "Relevance Score: " + "%.2f" % score + "\n\n" + \
                                  content
                    formatted_chunks.append(chunk_text)

                result = "\n\n---\n\n".join(formatted_chunks)

                # Update main span with successful results
                span.update(
                    output={
                        "result_type": "success",
                        "chunks_formatted": len(formatted_chunks),
                        "total_characters": len(result)
                    },
                    metadata={"status": "completed_successfully"}
                )
            else:
                # Update main span with no results found
                span.update(
                    output={
                        "result_type": "no_results",
                        "standard_results_count": len(standard_results.get('matches', [])),
                        "recent_results_count": len(recent_results)
                    },
                    metadata={"status": "no_matches_found"}
                )

            return result

    except Exception as e:
        # Log the error with tracing using Langfuse v3
        try:
            with ctx.deps.langfuse_client.start_as_current_span("retrieval_error") as error_span:
                error_span.update(
                    input={"user_query": user_query},
                    output={"error_message": str(e)},
                    metadata={
                        "status": "error",
                        "error_type": type(e).__name__
                    }
                )
        except:
            # Fallback if even error logging fails
            pass

        print(f"ERROR in retrieve_from_pinecone: {e}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving documentation from Surface Hippy Forum: {str(e)}"


@hip_rag_agent.tool
async def check_recent_entries(ctx: RunContext[CombinedDeps]) -> str:
    """Check for recent entries in Pinecone by date prefix."""
    try:
        # Direct access to the index
        index = ctx.deps.pinecone_client.Index("forum-pages")

        # Get current stats
        stats = index.describe_index_stats()
        namespace_count = stats.get('namespaces', {}).get('hip-forum', {}).get('vector_count', 0)

        # Use dummy vector for metadata-only queries
        dummy_vector = [0.0] * 1536

        # Get a larger sample of entries to analyze locally
        all_results = index.query(
            namespace="hip-forum",
            vector=dummy_vector,
            top_k=500,  # Get a good sample
            include_metadata=True
        )

        # Analyze dates locally
        entries_2025 = []
        entries_2024 = []
        all_dates = []

        for match in all_results['matches']:
            date_str = match.get('metadata', {}).get('started_date', '')
            if date_str and date_str != 'unknown':
                all_dates.append(date_str)

                if date_str.startswith('2025'):
                    entries_2025.append(match)
                elif date_str.startswith('2024'):
                    entries_2024.append(match)

        # Sort dates
        all_dates.sort(reverse=True)

        # Format some examples of entries
        examples_2025 = []
        for i, entry in enumerate(entries_2025[:3]):  # Show first 3
            metadata = entry.get('metadata', {})
            examples_2025.append(f"{i+1}. {metadata.get('started_date')} - {metadata.get('title', 'No title')[:50]}...")

        examples_2024 = []
        for i, entry in enumerate(entries_2024[:3]):  # Show first 3
            metadata = entry.get('metadata', {})
            examples_2024.append(f"{i+1}. {metadata.get('started_date')} - {metadata.get('title', 'No title')[:50]}...")

        return f"""
## Pinecone Date Analysis Report

Total vectors in hip-forum namespace: {namespace_count}
Sample size analyzed: {len(all_results['matches'])} entries

### 2025 Entries
Found: {len(entries_2025)} entries

Examples:
{"\\n".join(examples_2025) if examples_2025 else "None found"}

### 2024 Entries
Found: {len(entries_2024)} entries

Examples:
{"\\n".join(examples_2024) if examples_2024 else "None found"}

### 10 Most Recent Dates
{", ".join(all_dates[:10]) if all_dates else "No valid dates found"}

### Recommendation for Queries
Use prefix matching with startsWith operator or manual filtering for date searches.
        """
    except Exception as e:
        return f"Error checking recent entries: {str(e)}"

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

    with tracer.start_as_current_span("rag_retrieval") as span:
        # Set standard attributes
        span.set_attribute("langfuse.user.id", "user")
        span.set_attribute("source", source)
        span.set_attribute("query", user_query)
        span.set_attribute("operation", "documentation_retrieval")

        try:
            if source == "pinecone":
                print("DEBUG: Querying ONLY Pinecone source")
                span.set_attribute("retrieval_strategy", "pinecone_only")

                with tracer.start_as_current_span("pinecone_retrieval") as pinecone_span:
                    pinecone_span.set_attribute("langfuse.user.id", "user")
                    pinecone_span.set_attribute("source", "pinecone")
                    pinecone_span.set_attribute("query", user_query)

                    results = await retrieve_from_pinecone(ctx, user_query)

                    pinecone_span.set_attribute("results_length", len(results) if results else 0)
                    pinecone_span.set_attribute("status", "success")

                span.set_attribute("documents_retrieved", 1)
                span.set_attribute("status", "success")
                return f"## Results from Surface Hippy Forum ONLY\n{results}"

            elif source == "qdrant":
                print("DEBUG: Querying ONLY Qdrant source")
                span.set_attribute("retrieval_strategy", "qdrant_only")

                with tracer.start_as_current_span("qdrant_retrieval") as qdrant_span:
                    qdrant_span.set_attribute("langfuse.user.id", "user")
                    qdrant_span.set_attribute("source", "qdrant")
                    qdrant_span.set_attribute("query", user_query)

                    results = await retrieve_from_qdrant(ctx, user_query)

                    qdrant_span.set_attribute("results_length", len(results) if results else 0)
                    qdrant_span.set_attribute("status", "success")

                span.set_attribute("documents_retrieved", 1)
                span.set_attribute("status", "success")
                return f"## Results from Hip Resurfacing Group Messages ONLY\n{results}"

            elif source == "both":
                print("DEBUG: Querying BOTH sources")
                span.set_attribute("retrieval_strategy", "dual_source")

                # Get results from Qdrant
                with tracer.start_as_current_span("qdrant_retrieval") as qdrant_span:
                    qdrant_span.set_attribute("langfuse.user.id", "user")
                    qdrant_span.set_attribute("source", "qdrant")
                    qdrant_span.set_attribute("query", user_query)

                    qdrant_results = await retrieve_from_qdrant(ctx, user_query)

                    qdrant_span.set_attribute("results_length", len(qdrant_results) if qdrant_results else 0)
                    qdrant_span.set_attribute("status", "success")

                # Get results from Pinecone
                with tracer.start_as_current_span("pinecone_retrieval") as pinecone_span:
                    pinecone_span.set_attribute("langfuse.user.id", "user")
                    pinecone_span.set_attribute("source", "pinecone")
                    pinecone_span.set_attribute("query", user_query)

                    pinecone_results = await retrieve_from_pinecone(ctx, user_query)

                    pinecone_span.set_attribute("results_length", len(pinecone_results) if pinecone_results else 0)
                    pinecone_span.set_attribute("status", "success")

                span.set_attribute("documents_retrieved", 2)
                span.set_attribute("qdrant_results_length", len(qdrant_results) if qdrant_results else 0)
                span.set_attribute("pinecone_results_length", len(pinecone_results) if pinecone_results else 0)
                span.set_attribute("status", "success")

                # Combine results with source indicators
                return f"""
## Results from Hip Resurfacing Group Messages
{qdrant_results}
## Results from Surface Hippy Forum
{pinecone_results}
"""
            else:
                print(f"DEBUG: Invalid source specified: '{source}'")
                span.set_attribute("status", "error")
                span.set_attribute("error.type", "invalid_source")
                span.set_attribute("error.message", f"Invalid source: {source}")

                return f"Invalid search source specified: '{source}'. Please use 'qdrant', 'pinecone', or 'both'."

        except Exception as e:
            span.set_attribute("status", "error")
            span.set_attribute("error.message", str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.record_exception(e)

            return f"Error retrieving documentation: {str(e)}"

@hip_rag_agent.tool
async def evaluate_retrieval(
    ctx: RunContext[CombinedDeps],
    user_query: str,
    llm_response: str,
    ground_truth: str = None
) -> str:
    """Evaluate the quality of RAG retrieval and response - using both OpenTelemetry and Langfuse."""

    # Create a Langfuse span if we have the client
    if ctx.deps.langfuse_client:
        with ctx.deps.langfuse_client.start_as_current_span(name="rag_evaluation") as langfuse_span:
            langfuse_span.update(
                input={"user_query": user_query, "llm_response": llm_response[:500]},
                metadata={"has_ground_truth": ground_truth is not None}
            )
    else:
        langfuse_span = None

    # Use OpenTelemetry (keeping your existing pattern)
    with tracer.start_as_current_span("rag_evaluation") as span:
        # Set the same attributes that work for your retrievals
        span.set_attribute("langfuse.user.id", "user")
        span.set_attribute("operation", "evaluation")
        span.set_attribute("query", user_query)
        span.set_attribute("has_ground_truth", ground_truth is not None)
        span.set_attribute("response_length", len(llm_response))

        try:
            # Your evaluation logic
            eval_prompt = f"""
            Evaluate the quality of this RAG retrieval and response:
            User Query: {user_query}
            LLM Response: {llm_response}
            {f'Ground Truth: {ground_truth}' if ground_truth else ''}
            Please evaluate on the following criteria:
            1. Relevance (1-10): How relevant are the retrieved documents to the query?
            2. Completeness (1-10): How completely does the answer address all aspects of the query?
            3. Factual Accuracy (1-10): How factually accurate is the response based on the retrieved content?
            4. Hallucination Assessment (1-10): Rate how well the response avoids making claims not supported by the retrieved content (10=no hallucinations).
            For each criterion, provide a score and a brief explanation.
            Format your response as valid JSON:
            {{
                "relevance": {{ "score": X, "explanation": "..." }},
                "completeness": {{ "score": X, "explanation": "..." }},
                "factual_accuracy": {{ "score": X, "explanation": "..." }},
                "hallucination_assessment": {{ "score": X, "explanation": "..." }}
            }}
            """

            # Create a nested span for the LLM call
            with tracer.start_as_current_span("llm_evaluation_call") as eval_span:
                eval_span.set_attribute("langfuse.user.id", "user")
                eval_span.set_attribute("model", "gpt-4o-mini")
                eval_span.set_attribute("operation", "llm_evaluation")
                eval_span.set_attribute("prompt_length", len(eval_prompt))

                # Also create a Langfuse generation if we have the span
                if langfuse_span:
                    with langfuse_span.start_as_current_span(name="evaluation_llm_call") as langfuse_generation:
                        langfuse_generation.update(
                            metadata={
                                "model": "gpt-4o-mini",
                                "input": eval_prompt[:1000]
                            }
                        )

                response = await ctx.deps.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are an expert RAG system evaluator. Analyze retrieval quality and response accuracy."},
                        {"role": "user", "content": eval_prompt}
                    ]
                )

                evaluation_result = json.loads(response.choices[0].message.content)

                # Update Langfuse generation with results
                if langfuse_generation:
                    langfuse_generation.update(
                        output=response.choices[0].message.content,
                        metadata={
                            "usage": {
                                "promptTokens": response.usage.prompt_tokens if response.usage else 0,
                                "completionTokens": response.usage.completion_tokens if response.usage else 0,
                                "totalTokens": response.usage.total_tokens if response.usage else 0
                            }
                        }
                    )
                    langfuse_generation.end()

                # Set evaluation results as attributes on the OTel span
                eval_span.set_attribute("evaluation_completed", True)
                if response.usage:
                    eval_span.set_attribute("tokens_used", response.usage.total_tokens)
                    eval_span.set_attribute("prompt_tokens", response.usage.prompt_tokens)
                    eval_span.set_attribute("completion_tokens", response.usage.completion_tokens)

            # Process and record scores
            total_score = 0
            scores_dict = {}
            for criterion, data in evaluation_result.items():
                score = data["score"]
                explanation = data["explanation"]

                # Record scores as attributes (OTel)
                span.set_attribute(f"score.{criterion}", score)
                span.set_attribute(f"explanation.{criterion}", explanation)
                scores_dict[criterion] = {"score": score, "explanation": explanation}
                total_score += score

            overall_score = total_score / 4
            span.set_attribute("overall_score", overall_score)
            span.set_attribute("evaluation_status", "success")

            # Update Langfuse span with final results
            if langfuse_span:
                langfuse_span.update(
                    metadata={
                        "score": {
                            "name": "overall_quality",
                            "value": overall_score
                        },
                        "output": {
                            "overall_score": overall_score,
                            "detailed_scores": scores_dict
                        }
                    }
                )
                langfuse_span.end()

            # Format results (same as before)
            formatted_evaluation = f"""
## RAG Evaluation Results
### Query
"{user_query}"
### Evaluation Scores
- Relevance: {evaluation_result["relevance"]["score"]}/10
  - {evaluation_result["relevance"]["explanation"]}
- Completeness: {evaluation_result["completeness"]["score"]}/10
  - {evaluation_result["completeness"]["explanation"]}
- Factual Accuracy: {evaluation_result["factual_accuracy"]["score"]}/10
  - {evaluation_result["factual_accuracy"]["explanation"]}
- Hallucination Assessment: {evaluation_result["hallucination_assessment"]["score"]}/10
  - {evaluation_result["hallucination_assessment"]["explanation"]}
### Overall Quality
{overall_score:.1f}/10
            """

            return formatted_evaluation

        except json.JSONDecodeError as e:
            span.set_attribute("evaluation_status", "error")
            span.set_attribute("error_type", "json_decode_error")
            span.set_attribute("error_message", str(e))
            span.record_exception(e)

            if langfuse_span:
                langfuse_span.update(
                    status="error",
                    metadata={"error": f"JSON parsing error: {str(e)}"}
                )
                langfuse_span.end()

            return f"JSON parsing error in evaluation: {str(e)}"

        except Exception as e:
            span.set_attribute("evaluation_status", "error")
            span.set_attribute("error_type", type(e).__name__)
            span.set_attribute("error_message", str(e))
            span.record_exception(e)

            if langfuse_span:
                langfuse_span.update(
                    status="error",
                    metadata={"error": str(e)}
                )
                langfuse_span.end()

            return f"Evaluation error: {str(e)}"




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
