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
from pinecone import Pinecone, ServerlessSpec
from bs4 import BeautifulSoup

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

class PineconeURLProcessor:
    def __init__(self):
        load_dotenv()
        # Initialize Pinecone client using the new API pattern
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self.index_name = "forum-pages"
        self.namespace = "hip-forum"
        self.init_index()
        self.processed_chunks_cache: Dict[str, Set[int]] = {}
        self.last_cache_update: Dict[str, float] = {}
        self.cache_ttl = 60

        # Connect to the index
        self.index = self.pc.Index(self.index_name)

    def init_index(self):
        """Initialize Pinecone index with proper configuration."""
        # Check if index exists
        existing_indexes = self.pc.list_indexes().names()

        if self.index_name not in existing_indexes:
            # Create index with proper configuration
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created index '{self.index_name}'")
        else:
            print(f"Index '{self.index_name}' already exists")

    async def get_processed_chunks(self, url: str) -> Set[int]:
        """Get processed chunk numbers for a URL with caching."""
        current_time = datetime.now().timestamp()

        if url in self.processed_chunks_cache and \
           current_time - self.last_cache_update.get(url, 0) < self.cache_ttl:
            return self.processed_chunks_cache[url]

        try:
            # Query Pinecone for all chunks with this URL
            query_response = self.index.query(
                namespace=self.namespace,
                vector=[0.0] * 1536,  # Dummy vector for metadata-only query
                filter={"url": {"$eq": url}},
                include_metadata=True,
                top_k=1000  # Adjust based on expected max chunks per URL
            )

            processed_chunks = {int(match['metadata'].get('chunk_number', -1))
                              for match in query_response['matches']
                              if 'chunk_number' in match['metadata']}

            self.processed_chunks_cache[url] = processed_chunks
            self.last_cache_update[url] = current_time

            return processed_chunks
        except Exception as e:
            print(f"Error querying processed chunks for {url}: {e}")
            return set()

    async def insert_chunk(self, chunk: ProcessedChunk):
        """Insert a processed chunk into Pinecone."""
        try:
            # Generate a unique ID for the point
            import hashlib
            point_id = hashlib.md5(f"{chunk.url}_{chunk.chunk_number}".encode()).hexdigest()

            # Prepare metadata dictionary combining all fields
            metadata = {
                "url": chunk.url,
                "chunk_number": chunk.chunk_number,
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                # Expand metadata fields directly
                **chunk.metadata
            }

            # Insert the point with metadata
            self.index.upsert(
                vectors=[(point_id, chunk.embedding, metadata)],
                namespace=self.namespace
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

async def extract_started_date(html: str) -> str:
    """Extract the started_date from the first message in a forum topic."""
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # First look for the specific format in this forum
        # Format: <div class="smalltext">« <strong> on:</strong> March 15, 2025, 11:24:52 AM »</div>
        on_pattern = soup.select('div.smalltext strong:contains("on:")')
        if on_pattern:
            for element in on_pattern:
                parent = element.parent
                if parent:
                    date_text = parent.text.strip()
                    # Extract date after "on:"
                    if "on:" in date_text:
                        date_part = date_text.split("on:")[1].strip()
                        # Now extract just the date portion before the time
                        if "," in date_part:
                            date_parts = date_part.split(',')
                            if len(date_parts) >= 2:
                                # Combine month day and year
                                date_str = date_parts[0] + "," + date_parts[1]
                                try:
                                    date_obj = datetime.strptime(date_str.strip(), '%B %d, %Y')
                                    return date_obj.strftime('%Y-%m-%d')
                                except Exception as e:
                                    print(f"Error parsing specific date format: {date_str} - {e}")

        # Fallback: Look for any text that matches date pattern using regex
        import re
        full_date_pattern = re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}')

        all_text = soup.text
        date_matches = full_date_pattern.findall(all_text)

        if date_matches:
            # Search for the complete date string in the text
            complete_matches = re.findall(r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})', all_text)
            if complete_matches:
                try:
                    date_obj = datetime.strptime(complete_matches[0], '%B %d, %Y')
                    return date_obj.strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"Error parsing date match: {complete_matches[0]} - {e}")

        print(f"No valid date found in the page content")
        return None
    except Exception as e:
        print(f"Error extracting started date: {e}")
        return None
        
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
                        summary: str, started_date: str, openai_client: AsyncOpenAI,
                        url_processor: PineconeURLProcessor,
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
                        "source": "hip_forum",
                        "chunk_size": len(chunk),
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "url_path": urlparse(url).path,
                        "started_date": started_date,
                        "most_recent_date": topic_data.get('most_recent_date'),
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
    url_processor = PineconeURLProcessor()

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

                # Extract the started_date from the HTML
                started_date = await extract_started_date(result.html)
                if not started_date:
                    print(f"Warning: Could not extract started_date for {url}")
                    started_date = "unknown"  # Fallback value
                else:
                    print(f"Extracted started_date: {started_date} for {url}")

                if chunks:
                    # Generate summary once for all chunks
                    summary = await generate_summary(content, openai_client)
                    print(f"\nProcessing {url}")
                    print(f"Generated summary: {summary[:100]}...")

                    await process_chunks(chunks, url, topic, summary, started_date,
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
    print("Starting ingestion process with Pinecone...")
    openai_client = await setup_clients()

    # Load pre-crawled topic data
    try:
        with open('surfacehippy_topics.json', 'r') as f:
            topic_data = json.load(f)
        print(f"Loaded {len(topic_data)} topics from surfacehippy_topics.json")
    except Exception as e:
        print(f"Error loading surfacehippy_topics.json: {e}")
        return

    await crawl_parallel(topic_data, openai_client)
    print("Ingestion process completed")

if __name__ == "__main__":
    asyncio.run(main())
