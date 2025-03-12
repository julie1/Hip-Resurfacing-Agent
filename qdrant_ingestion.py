import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configurations
CHUNK_SIZE = 5000
MAX_CONCURRENT_CRAWLS = 3
MAX_CONCURRENT_CHUNKS = 5

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

class QdrantURLProcessor:
    def __init__(self):
        load_dotenv()
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = "site_pages"
        self.init_collection()
        self.processed_chunks_cache: Dict[str, Set[int]] = {}
        self.last_cache_update: Dict[str, float] = {}
        self.cache_ttl = 60

    def init_collection(self):
        """Initialize Qdrant collection with proper configuration."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # Create collection with vector config for embeddings
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embeddings dimension
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,  # Index immediately
                ),
            )

            # Create payload indexes for efficient filtering and text search
            # URL and chunk_number for unique identification
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="url",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_number",
                field_schema=models.PayloadSchemaType.INTEGER,
            )

            # Text fields for search
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="title",
                field_schema=models.PayloadSchemaType.TEXT,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="summary",
                field_schema=models.PayloadSchemaType.TEXT,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content",
                field_schema=models.PayloadSchemaType.TEXT,
            )

            # Common metadata fields for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="url_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print(f"Created collection '{self.collection_name}' with all necessary indexes")
        else:
            print(f"Collection '{self.collection_name}' already exists")

    async def get_processed_chunks(self, url: str) -> Set[int]:
        """Get processed chunk numbers for a URL with caching."""
        current_time = datetime.now().timestamp()

        if url in self.processed_chunks_cache and \
           current_time - self.last_cache_update.get(url, 0) < self.cache_ttl:
            return self.processed_chunks_cache[url]

        try:
            # Query Qdrant for all chunks with this URL
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url)
                        )
                    ]
                ),
                limit=1000,  # Adjust based on your expected max chunks per URL
                with_payload=["chunk_number"],
                with_vectors=False,
            )

            processed_chunks = {int(point.payload.get("chunk_number")) for point in results[0]}
            self.processed_chunks_cache[url] = processed_chunks
            self.last_cache_update[url] = current_time

            return processed_chunks
        except Exception as e:
            print(f"Error querying processed chunks for {url}: {e}")
            return set()

    async def insert_chunk(self, chunk: ProcessedChunk):
        """Insert a processed chunk into Qdrant."""
        try:
            # Generate a unique ID for the point
            # Using a hash of URL and chunk number ensures uniqueness
            import hashlib
            point_id = hashlib.md5(f"{chunk.url}_{chunk.chunk_number}".encode()).hexdigest()

            # Insert the point with payload
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=chunk.embedding,
                        payload={
                            "url": chunk.url,
                            "chunk_number": chunk.chunk_number,
                            "title": chunk.title,
                            "summary": chunk.summary,
                            "content": chunk.content,
                            # Expand metadata fields directly into payload
                            # This makes them directly filterable
                            **chunk.metadata
                        }
                    )
                ]
            )
            print(f"Successfully inserted chunk {chunk.chunk_number} for {chunk.url}")
            return True
        except Exception as e:
            print(f"Error inserting chunk: {e}")
            return False

async def setup_clients():
    """Initialize OpenAI client."""
    load_dotenv()
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return openai_client

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end)

    return chunks

async def generate_summary(text: str, openai_client: AsyncOpenAI) -> str:
    """Generate a summary using GPT-4o-mini."""
    summary_prompt = """Create a concise summary for this forum post that captures the key points of the discussion."""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Content:\n{text[:2000]}..."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error generating summary"

async def process_chunks(chunks: List[str], url: str, topic_data: Dict,
                        summary: str, openai_client: AsyncOpenAI,
                        url_processor: QdrantURLProcessor,
                        chunk_semaphore: asyncio.Semaphore):
    """Process all chunks for a URL with the same summary."""
    processed_chunks = await url_processor.get_processed_chunks(url)

    async def process_single_chunk(chunk: str, chunk_number: int):
        if chunk_number in processed_chunks:
            return

        async with chunk_semaphore:
            try:
                # Get embedding
                embedding_response = await openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                embedding = embedding_response.data[0].embedding

                processed = ProcessedChunk(
                    url=url,
                    chunk_number=chunk_number,
                    title=topic_data['subject'],
                    summary=summary,
                    content=chunk,
                    metadata={
                        "source": "hip_messages",
                        "chunk_size": len(chunk),
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "url_path": urlparse(url).path,
                        "started_date": topic_data['started_date'],
                        "most_recent_date": topic_data['most_recent_date'],
                        "total_chunks": len(chunks)
                    },
                    embedding=embedding
                )

                await url_processor.insert_chunk(processed)
                print(f"Processed chunk {chunk_number}/{len(chunks)} for {url}")
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error processing chunk {chunk_number} for {url}: {e}")

    tasks = []
    for i, chunk in enumerate(chunks):
        if i not in processed_chunks:
            tasks.append(process_single_chunk(chunk, i))

    if tasks:
        print(f"Processing {len(tasks)} remaining chunks for {url}")
        await asyncio.gather(*tasks)
    else:
        print(f"All chunks already processed for {url}")

async def crawl_parallel(topic_data: List[Dict], openai_client: AsyncOpenAI):
    """Crawl multiple URLs in parallel using pre-crawled metadata."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    chunk_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
    url_processor = QdrantURLProcessor()

    async def process_url(topic: Dict):
        """Process a single URL with its metadata."""
        url = topic['url']
        crawler = AsyncWebCrawler(config=browser_config)

        try:
            await crawler.start()
            result = await crawler.arun(url=url, config=crawl_config, session_id="session1")

            if result.success:
                content = result.markdown_v2.raw_markdown
                chunks = chunk_text(content)

                if chunks:
                    # Generate summary once for all chunks
                    summary = await generate_summary(content, openai_client)
                    print(f"\nProcessing {url}")
                    print(f"Generated summary: {summary[:100]}...")

                    await process_chunks(chunks, url, topic, summary,
                                       openai_client, url_processor,
                                       chunk_semaphore)
                else:
                    print(f"No content found for {url}")
            else:
                print(f"Failed to crawl: {url} - Error: {result.error_message}")

        except Exception as e:
            print(f"Error processing {url}: {e}")
        finally:
            await crawler.close()

    # Process URLs in parallel with rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRAWLS)
    async def process_with_semaphore(topic):
        async with semaphore:
            await process_url(topic)
            await asyncio.sleep(1)  # Rate limiting

    tasks = [process_with_semaphore(topic) for topic in topic_data]
    await asyncio.gather(*tasks)

async def main():
    print("Starting ingestion process with Qdrant...")
    openai_client = await setup_clients()

    # Load pre-crawled topic data
    try:
        with open('topic_data.json', 'r') as f:
            topic_data = json.load(f)
        print(f"Loaded {len(topic_data)} topics from topic_data.json")
    except Exception as e:
        print(f"Error loading topic_data.json: {e}")
        return

    await crawl_parallel(topic_data, openai_client)
    print("Ingestion process completed")

if __name__ == "__main__":
    asyncio.run(main())
