import asyncio
import json
import os
from datetime import datetime
from dateutil import parser
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re
import time

# Load environment variables
load_dotenv()

# Qdrant client configuration
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
COLLECTION_NAME = "site_pages"

# Forum configuration
BASE_GROUP_URL = "https://groups.io/g/Hipresurfacingsite"
BASE_URL = f"{BASE_GROUP_URL}/topics"

async def get_latest_date_from_qdrant():
    """Retrieve the latest date from Qdrant database."""
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total points: {collection_info.points_count}")
        
        # Get all records with payload containing most_recent_date
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,  # Adjust based on your collection size
            with_payload=["most_recent_date", "url"],
            with_vectors=False,
        )
        
        if results[0]:
            # Find the point with the latest date manually
            latest_date = None
            latest_url = None
            
            for point in results[0]:
                date_str = point.payload.get("most_recent_date")
                if not date_str:
                    continue
                
                try:
                    date = parser.parse(date_str)
                    if latest_date is None or date > latest_date:
                        latest_date = date
                        latest_url = point.payload.get("url")
                except Exception as e:
                    print(f"Error parsing date {date_str}: {e}")
            
            if latest_date:
                latest_date_str = latest_date.strftime('%Y-%m-%d')
                print(f"Latest date in Qdrant: {latest_date_str}")
                print(f"URL with latest date: {latest_url}")
                return latest_date_str
            else:
                print("No valid dates found in Qdrant.")
                return None
        else:
            print("No records found in Qdrant.")
            return None
    except Exception as e:
        print(f"Error getting latest date from Qdrant: {e}")
        return None

async def extract_topic_info(html: str, base_group_url: str):
    """Extract topic URLs, subjects, and dates from a page."""
    topics = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        rows = soup.find_all('tr')
        
        for row in rows:
            topic_cell = row.find('td')
            if not topic_cell:
                continue
                
            # Get subject and URL
            subject_link = topic_cell.find('a', class_='subject')
            if not subject_link:
                continue
                
            subject = subject_link.text.strip()
            topic_url = subject_link['href']
            if not topic_url.startswith('http'):
                topic_url = base_group_url.rstrip('/') + topic_url
            
            # Get thread attribution
            thread_attr = topic_cell.find('span', class_='thread-attribution')
            if thread_attr:
                # Find all date spans
                date_spans = thread_attr.find_all('span', title=True)
                
                started_date = None
                most_recent_date = None
                
                if date_spans:
                    # First span is typically the start date
                    if len(date_spans) > 0:
                        title = date_spans[0].get('title', '')
                        if title:
                            try:
                                date = datetime.strptime(title, '%b %d, %Y %I:%M%p')
                                started_date = date.strftime('%Y-%m-%d')
                            except Exception as e:
                                print(f"Error parsing start date: {e}")
                    
                    # Last span is the most recent date
                    if len(date_spans) > 1:
                        title = date_spans[-1].get('title', '')
                        if title:
                            try:
                                date = datetime.strptime(title, '%b %d, %Y %I:%M%p')
                                most_recent_date = date.strftime('%Y-%m-%d')
                            except Exception as e:
                                print(f"Error parsing recent date: {e}")
                
                topics.append({
                    'url': topic_url,
                    'subject': subject,
                    'started_date': started_date,
                    'most_recent_date': most_recent_date
                })
                
    except Exception as e:
        print(f"Error extracting topic information: {e}")
        import traceback
        traceback.print_exc()
    return topics

async def extract_next_page_info(html: str):
    """Extract the 'after' token for the next page."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        pagination = soup.find('ul', class_='pagination')
        if pagination:
            next_link = pagination.find('i', class_='fa-chevron-right')
            if next_link:
                next_a_tag = next_link.find_parent('a')
                if next_a_tag and 'href' in next_a_tag.attrs:
                    href = next_a_tag['href']
                    match = re.search(r'after=([^&]+)', href)
                    if match:
                        return match.group(1)
    except Exception as e:
        print(f"Error parsing HTML for next page: {e}")
    return ""

async def crawl_new_topics(latest_date_in_qdrant):
    """Crawl forum topics that are newer than the latest date in Qdrant."""
    latest_date_obj = parser.parse(latest_date_in_qdrant) if latest_date_in_qdrant else None
    new_topics = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        current_url = BASE_URL
        page_num = 1
        seen_tokens = set()
        continue_crawling = True

        while continue_crawling:
            print(f"\nFetching page {page_num}: {current_url}")
            
            try:
                # Navigate to the page and wait for the table to load
                await page.goto(current_url)
                await page.wait_for_selector('table.table-condensed')
                
                # Get the fully rendered HTML
                html = await page.content()
                
                page_topics = await extract_topic_info(html, BASE_GROUP_URL)
                
                # Check if we've reached topics older than our latest date
                page_has_new_topics = False
                for topic in page_topics:
                    topic_started_date = parser.parse(topic['started_date']) if topic['started_date'] else None
                    topic_most_recent_date = parser.parse(topic['most_recent_date']) if topic['most_recent_date'] else None
                    
                    # If the topic has a most_recent_date that's newer than our latest date, or if it's
                    # a new topic (started_date is newer than our latest date), add it to new_topics
                    if ((topic_most_recent_date and latest_date_obj and topic_most_recent_date > latest_date_obj) or
                        (topic_started_date and latest_date_obj and topic_started_date > latest_date_obj)):
                        new_topics.append(topic)
                        page_has_new_topics = True
                
                # If this page had no new topics, we can stop crawling
                if not page_has_new_topics and latest_date_obj:
                    print(f"No new topics found on page {page_num}. Stopping crawl.")
                    break
                
                print(f"Found {len(page_topics)} topics on page {page_num}, " + 
                     (f"including {len(new_topics)} new topics." if latest_date_obj else "processing all topics."))
                
                # Get the next page token
                next_token = await extract_next_page_info(html)
                
                if not next_token or next_token in seen_tokens:
                    print("No more pages to fetch")
                    break
                
                seen_tokens.add(next_token)
                current_url = f"{BASE_URL}?page={page_num + 1}&after={next_token}"
                page_num += 1
                
                await asyncio.sleep(1)  # Be nice to the server
                
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                break
                
        await browser.close()
        
    return new_topics

async def main():
    # Get the latest date from Qdrant
    latest_date = await get_latest_date_from_qdrant()
    
    if not latest_date:
        print("No latest date found in Qdrant. Will run a full crawl instead.")
    
    # Crawl new topics
    start_time = time.time()
    new_topics = await crawl_new_topics(latest_date)
    end_time = time.time()
    
    print(f"\nFound {len(new_topics)} new topics in {end_time - start_time:.2f} seconds")
    
    if new_topics:
        # Save new topics to a file
        output_file = 'new_topic_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_topics, f, indent=2, ensure_ascii=False)
        
        print(f"\nNew data saved to {output_file}")
        
        # Print the first new topic for verification
        print("\nFirst new topic data:")
        print(json.dumps(new_topics[0], indent=2))
        
        # Now you can run your ingestion script with only the new topics
        print("\nRunning ingestion script with new topics...")
        
        # Import the ingestion code and run the ingestion with only the new topics
        try:
            from qdrant_ingestion import setup_clients, crawl_parallel
            
            openai_client = await setup_clients()
            await crawl_parallel(new_topics, openai_client)
            
            print("\nIngestion of new topics completed.")
        except ImportError:
            print("\nPlease run ingestion manually using the new_topic_data.json file:")
            print("python qdrant_ingestion.py --input new_topic_data.json")
    else:
        print("No new topics found. Database is up to date.")

if __name__ == "__main__":
    asyncio.run(main())
