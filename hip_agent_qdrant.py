from __future__ import annotations as _annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()
#llm_model = st.secrets.get("LLM_MODEL", "gpt-4o-mini")
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')
@dataclass
class PydanticAIDeps:
    qdrant_client: QdrantClient
    openai_client: AsyncOpenAI

class SearchResult(BaseModel):
    content: str
    similarity: float
    metadata: Dict
    title: str
    url: str

system_prompt = """You are an expert at retrieving and analyzing information from hip resurfacing group posts to answer questions about hip resurfacing surgery.

When answering questions:
1. Always ground your responses in the specific content retrieved from the posts
2. Quote relevant passages directly using markdown quotes to support your answers
3. Cite the source URL when referencing specific information
4. Express uncertainty when information is ambiguous or incomplete
5. Synthesize information from multiple posts when relevant
6. Maintain awareness of post dates to contextualize information
7. Focus only on hip resurfacing related content

Important: Don't make claims without supporting evidence from the posts. If you can't find relevant information, clearly state that limitation.
"""


hip_agent_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

@hip_agent_expert.tool
async def diagnose_qdrant_setup(ctx: RunContext[PydanticAIDeps]) -> str:
    """Diagnose Qdrant setup and collection status."""
    try:
        # Check if the collection exists
        try:
            collection_info = ctx.deps.qdrant_client.get_collection(collection_name="site_pages")
            collection_exists = True
            vector_size = collection_info.config.params.vectors.size
            points_count = collection_info.points_count
        except Exception as e:
            collection_exists = False
            vector_size = "N/A"
            points_count = "N/A"

        # Check for documents with source=hip_messages
        hip_docs_count = 0
        if collection_exists:
            try:
                count_result = ctx.deps.qdrant_client.count(
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
                hip_docs_count = count_result.count
            except Exception as e:
                hip_docs_count = f"Error counting: {str(e)}"

        # Get sample hip_messages document if available
        sample_doc = "None found"
        if collection_exists and hip_docs_count > 0:
            try:
                # Try to get one document with source=hip_messages
                scroll_result = ctx.deps.qdrant_client.scroll(
                    collection_name="site_pages",
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value="hip_messages")
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )

                points, _ = scroll_result

                if points:
                    sample_doc = f"Found document with ID: {points[0].id}\n"
                    sample_doc += "Fields: " + ", ".join(points[0].payload.keys())

                    # Check for specific fields
                    important_fields = ["title", "url", "content", "source"]
                    missing_fields = [field for field in important_fields
                                    if field not in points[0].payload]

                    if missing_fields:
                        sample_doc += f"\nWARNING: Missing important fields: {', '.join(missing_fields)}"
                    else:
                        sample_doc += "\nAll important fields present."
            except Exception as e:
                sample_doc = f"Error getting sample: {str(e)}"

        return f"""
## Qdrant Diagnostic Report

### Collection Status
- Collection 'site_pages' exists: {collection_exists}
- Vector size: {vector_size}
- Total points in collection: {points_count}

### Hip Messages Documents
- Documents with source=hip_messages: {hip_docs_count}

### Sample Document
{sample_doc}

### Recommendations
{("- Collection seems properly set up." if collection_exists and hip_docs_count > 0
  else "- Collection does not exist or has no hip_messages documents.")}
{("- Check that documents have been properly imported with the correct 'source' field."
  if hip_docs_count == 0 and collection_exists
  else "")}
"""

    except Exception as e:
        return f"Error diagnosing Qdrant setup: {str(e)}"
        
@hip_agent_expert.tool
async def check_qdrant_connection(ctx: RunContext[PydanticAIDeps]) -> str:
    """Check if Qdrant connection is working and count documents."""
    try:
        # Get collection info
        collection_info = ctx.deps.qdrant_client.get_collection(collection_name="site_pages")

        # Count total points
        count_result = ctx.deps.qdrant_client.count(
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

        # Get sample points
        sample_points = ctx.deps.qdrant_client.scroll(
            collection_name="site_pages",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="hip_messages")
                    )
                ]
            ),
            limit=3,
            with_payload=["title", "url"],
        )

        sample_titles = []
        for points, _ in sample_points:
            for point in points:
                if "title" in point.payload:
                    sample_titles.append(point.payload["title"])

        return f"""
Qdrant Connection Status:
- Collection exists: Yes
- Vector size: {collection_info.config.params.vectors.size}
- Points count with source=hip_messages: {count_result.count}
- Sample titles: {", ".join(sample_titles) if sample_titles else "None found"}
"""

    except Exception as e:
        return f"Error checking Qdrant connection: {str(e)}"

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


@hip_agent_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """Enhanced RAG retrieval with better ranking and formatting using Qdrant."""
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
            limit=5,  # Get top 5 results
            with_payload=True,
        )

        if not search_results:
            # If vector search returns no results, try text search
            try:
                # Create a text search filter for keywords from the user query
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
                    return "No relevant documentation found for your query."

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
                return "No relevant documentation found for your query."

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
        print(f"Error retrieving documentation: {e}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving documentation: {str(e)}"

@hip_agent_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Hip Resurfacing Email pages.

    Returns:
        List[str]: List of unique URLs for all Email pages
    """
    try:
        # Scroll returns batches of points and a next_page_offset
        scroll_result = ctx.deps.qdrant_client.scroll(
            collection_name="site_pages",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="hip_messages")
                    )
                ]
            ),
            limit=100,
            with_payload=["url"],
            with_vectors=False,
        )

        # Correct unpacking: scroll_result is a tuple (List[Point], Optional[PointId])
        points, next_page_offset = scroll_result

        # Extract unique URLs
        urls = set()
        for point in points:
            if "url" in point.payload:
                urls.add(point.payload["url"])

        # If there are more pages, continue scrolling
        while next_page_offset is not None:
            scroll_result = ctx.deps.qdrant_client.scroll(
                collection_name="site_pages",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value="hip_messages")
                        )
                    ]
                ),
                limit=100,
                with_payload=["url"],
                with_vectors=False,
                offset=next_page_offset
            )

            additional_points, next_page_offset = scroll_result

            for point in additional_points:
                if "url" in point.payload:
                    urls.add(point.payload["url"])

        return sorted(list(urls))

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        import traceback
        traceback.print_exc()
        return [f"Error: {str(e)}"]

@hip_agent_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Qdrant client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Use scroll for getting all chunks of a specific URL
        scroll_result = ctx.deps.qdrant_client.scroll(
            collection_name="site_pages",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url",
                        match=models.MatchValue(value=url)
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="hip_messages")
                    )
                ]
            ),
            limit=100,
            with_payload=["chunk_number", "title", "content"],
            with_vectors=False,
        )

        # Extract and organize the chunks
        chunks = []
        page_title = None

        # Process the first batch
        points, next_page_offset = scroll_result

        for point in points:
            if "chunk_number" in point.payload and "content" in point.payload:
                chunks.append((
                    int(point.payload["chunk_number"]),
                    point.payload["content"]
                ))
                # Get title from the first chunk
                if page_title is None and "title" in point.payload:
                    page_title = point.payload["title"].split(' - ')[0]  # Get the main title

        # If there are more chunks, retrieve them
        while next_page_offset is not None:
            scroll_result = ctx.deps.qdrant_client.scroll(
                collection_name="site_pages",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url)
                        ),
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value="hip_messages")
                        )
                    ]
                ),
                limit=100,
                with_payload=["chunk_number", "title", "content"],
                with_vectors=False,
                offset=next_page_offset
            )

            additional_points, next_page_offset = scroll_result

            for point in additional_points:
                if "chunk_number" in point.payload and "content" in point.payload:
                    chunks.append((
                        int(point.payload["chunk_number"]),
                        point.payload["content"]
                    ))
                    # Get title from the first chunk if not already set
                    if page_title is None and "title" in point.payload:
                        page_title = point.payload["title"].split(' - ')[0]  # Get the main title

        if not chunks:
            return f"No content found for URL: {url}"

        # Sort chunks by chunk_number
        chunks.sort(key=lambda x: x[0])

        # Format the page with its title and all chunks
        formatted_content = [f"# {page_title or 'Unknown Title'}\n"]

        # Add each chunk's content
        for _, content in chunks:
            formatted_content.append(content)

        # Join everything together
        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving page content: {str(e)}"
