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
from groupsio_crawl import crawl_new_topics
import re
import time
import sys

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

STATE_FILE = 'qdrant_latest_date.json'

def save_latest_date(date_str):
    """Save latest date to state file."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'latest_date': date_str,
                'updated_at': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\n✓ Saved latest date to state file: {date_str}")
    except Exception as e:
        print(f"\n✗ Error saving state file: {e}")

def load_latest_date():
    """Load latest date from state file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                latest_date = data.get('latest_date')
                if latest_date:
                    print(f"✓ Using cached latest date: {latest_date}")
                    return latest_date
        except Exception as e:
            print(f"✗ Error loading state file: {e}")
    return None

def extract_and_save_latest_date(topics):
    """
    Extract the newest date from topics list and save to state file.
    This is called AFTER successful ingestion.

    Args:
        topics: List of topic dicts from new_topics
    """
    if not topics:
        print("No topics to extract date from")
        return

    newest_date = None
    newest_url = None

    for topic in topics:
        # Use most_recent_date if available, otherwise started_date
        date_str = topic.get('most_recent_date') or topic.get('started_date')

        if not date_str:
            continue

        try:
            date = parser.parse(date_str)
            if newest_date is None or date > newest_date:
                newest_date = date
                newest_url = topic.get('url')
        except Exception as e:
            print(f"Warning: Error parsing date '{date_str}': {e}")

    if newest_date:
        newest_str = newest_date.strftime('%Y-%m-%d')
        print(f"\n{'='*70}")
        print(f"📅 Updating state file with newest date from this batch:")
        print(f"   Date: {newest_str}")
        print(f"   URL:  {newest_url}")
        print(f"{'='*70}")
        save_latest_date(newest_str)
    else:
        print("⚠️  No valid dates found in topics")



async def get_latest_date_from_qdrant():
    """Retrieve the latest date from Qdrant database."""
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total points: {collection_info.points_count}")

        # Get all records with payload containing most_recent_date
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=30000,  # Adjust based on your collection size
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


async def main():
    # Try to load from state file first (much faster than scanning database)
    latest_date = load_latest_date()

# If no state file exists, scan the database once
    if not latest_date:
        print("No state file found, scanning database for latest date...")
        latest_date = await get_latest_date_from_qdrant()
        if latest_date:
            save_latest_date(latest_date)  # Cache it
    override = os.environ.get("SINCE_DATE_OVERRIDE", "").strip()
    if override:
        latest_date = override
        print(f"⚠️ Backfill mode: using override date {override}")

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
            failed = await crawl_parallel(new_topics, openai_client)
            if isinstance(failed, (list, tuple, set)) and failed:
                raise RuntimeError(f"{len(failed)} topics failed to ingest; state not advanced")
            # ⭐ THIS IS THE KEY LINE - Extract newest date from new_topics and save it
            extract_and_save_latest_date(new_topics)
        except ImportError:
            print("\nPlease run ingestion manually using the new_topic_data.json file:")
            print("python qdrant_ingestion.py --input new_topic_data.json")
    else:
        print("No new topics found.")
        if latest_date and not override:
            save_latest_date(latest_date)   # refreshes the cache; never writes today's date
    # else:
    #     print("No new topics found. Database is up to date.")

if __name__ == "__main__":
    if __name__ == "__main__":
        try:
            # Your existing main code here
            asyncio.run(main())
            print("\n✅ Qdrant ingestion completed successfully")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Qdrant ingestion FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
