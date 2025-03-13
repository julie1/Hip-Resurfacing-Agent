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
from supabase import create_client, Client

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

class URLProcessor:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.processed_chunks_cache: Dict[str, Set[int]] = {}
        self.last_cache_update: Dict[str, float] = {}
        self.cache_ttl = 60

    async def get_processed_chunks(self, url: str) -> Set[int]:
        """Get processed chunk numbers for a URL with caching."""
        current_time = datetime.now().timestamp()

        if url in self.processed_chunks_cache and \
           current_time - self.last_cache_update.get(url, 0) < self.cache_ttl:
            return self.processed_chunks_cache[url]

        try:
            result = self.supabase.table("site_pages")\
                .select("chunk_number")\
                .eq("url", url)\
                .execute()

            processed_chunks = {r['chunk_number'] for r in result.data}
            self.processed_chunks_cache[url] = processed_chunks
            self.last_cache_update[url] = current_time

            return processed_chunks
        except Exception as e:
            print(f"Error querying processed chunks for {url}: {e}")
            return set()

async def setup_clients():
    """Initialize OpenAI and Supabase clients."""
    load_dotenv()
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
    return openai_client, supabase

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

async def insert_chunk(chunk: ProcessedChunk, supabase: Client):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }

        result = supabase.table("site_pages").insert(data).execute()
        print(f"Successfully inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_chunks(chunks: List[str], url: str, topic_data: Dict,
                        summary: str, openai_client: AsyncOpenAI,
                        url_processor: URLProcessor, supabase: Client,
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

                await insert_chunk(processed, supabase)
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

async def crawl_parallel(topic_data: List[Dict], openai_client: AsyncOpenAI,
                        supabase: Client):
    """Crawl multiple URLs in parallel using pre-crawled metadata."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    chunk_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
    url_processor = URLProcessor(supabase)

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
                                      supabase, chunk_semaphore)
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
    print("Starting ingestion process...")
    openai_client, supabase = await setup_clients()
    
    # Load pre-crawled topic data
    try:
        with open('topic_data.json', 'r') as f:
            topic_data = json.load(f)
        print(f"Loaded {len(topic_data)} topics from topic_data.json")
    except Exception as e:
        print(f"Error loading topic_data.json: {e}")
        return

    await crawl_parallel(topic_data, openai_client, supabase)
    print("Ingestion process completed")

if __name__ == "__main__":
    asyncio.run(main())
