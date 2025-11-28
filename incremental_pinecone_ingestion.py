import asyncio
import json
import os
import time
import re
import sys
from datetime import datetime, timedelta, timezone
from dateutil import parser
from dotenv import load_dotenv
from pinecone import Pinecone
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Pinecone client configuration
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "forum-pages"
NAMESPACE = "hip-forum"
# Forum configuration
BASE_URL = "https://surfacehippy.info/hiptalk"

STATE_FILE = 'pinecone_latest_date.json'

def save_latest_date(date_str):
    """Save latest date to state file."""
    with open(STATE_FILE, 'w') as f:
        json.dump({'latest_date': date_str, 'updated_at': datetime.now().isoformat()}, f, indent=2)
    print(f"Saved latest date to state file: {date_str}")

def load_latest_date():
    """Load latest date from state file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                latest_date = data.get('latest_date')
                if latest_date:
                    print(f"Loaded latest date from state file: {latest_date}")
                    return latest_date
        except Exception as e:
            print(f"Error loading state file: {e}")
    return None

async def get_latest_date_from_pinecone():
    """Retrieve the latest date from Pinecone database using list() for serverless indexes."""
    try:
        index = pc.Index(INDEX_NAME)
        print(f"Connected to index: {INDEX_NAME}")

        latest_date = None
        latest_url = None
        vector_count = 0

        # Use list() which automatically paginates through all vectors
        list_params = {}
        if NAMESPACE and NAMESPACE.strip():
            list_params['namespace'] = NAMESPACE

        print("Fetching all vector IDs...")
        all_ids = []

        # Collect all IDs first
        for id_batch in index.list(**list_params):
            all_ids.extend(id_batch)
            if len(all_ids) % 5000 == 0:  # Progress update every 5000
                print(f"Collected {len(all_ids)} IDs so far...")

        print(f"Total vectors found: {len(all_ids)}")

        # Use smaller batch size to avoid URI too large error
        # Reduce from 1000 to 100 to stay well under URL length limits
        batch_size = 100
        total_batches = (len(all_ids) + batch_size - 1) // batch_size

        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i+batch_size]
            batch_num = i // batch_size + 1

            fetch_params = {'ids': batch_ids}
            if NAMESPACE and NAMESPACE.strip():
                fetch_params['namespace'] = NAMESPACE

            if batch_num % 10 == 0 or batch_num == total_batches:  # Progress every 10 batches
                print(f"Fetching batch {batch_num}/{total_batches} (vectors {i+1} to {min(i+batch_size, len(all_ids))})...")

            try:
                fetch_result = index.fetch(**fetch_params)

                if fetch_result and hasattr(fetch_result, 'vectors'):
                    vectors_dict = fetch_result.vectors

                    for vector_id, vector_data in vectors_dict.items():
                        vector_count += 1
                        metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else vector_data.get('metadata', {})

                        most_recent_date = metadata.get('most_recent_date')
                        started_date = metadata.get('started_date')

                        date_str = most_recent_date or started_date
                        if not date_str:
                            continue

                        try:
                            date = parser.parse(date_str)
                            if latest_date is None or date > latest_date:
                                latest_date = date
                                latest_url = metadata.get('url')
                                print(f"  New latest date found: {latest_date.strftime('%Y-%m-%d')} from {latest_url}")
                        except Exception as e:
                            print(f"Error parsing date {date_str}: {e}")

            except Exception as fetch_error:
                print(f"Error fetching batch {batch_num}: {fetch_error}")
                # Continue with next batch even if one fails
                continue

        print(f"\nProcessed {vector_count} vectors total")

        if latest_date:
            latest_date_str = latest_date.strftime('%Y-%m-%d')
            print(f"Latest date in Pinecone: {latest_date_str}")
            print(f"URL with latest date: {latest_url}")
            return latest_date_str
        else:
            print("No valid dates found in Pinecone.")
            return None

    except Exception as e:
        print(f"Error getting latest date from Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return None


# async def get_latest_date_from_pinecone():
#     """Retrieve the latest date from Pinecone database."""
#     try:
#         index = pc.Index(INDEX_NAME)
#         print(f"Connected to index: {INDEX_NAME}")
#
#         results = index.query(
#             namespace=NAMESPACE,
#             vector=[0.0] * 1536,
#             include_metadata=True,
#             top_k=1000
#         )
#
#         if results and 'matches' in results and results['matches']:
#             latest_date = None
#             latest_url = None
#
#             for point in results['matches']:
#                 metadata = point.get('metadata', {})
#                 most_recent_date = metadata.get('most_recent_date')
#                 started_date = metadata.get('started_date')
#                 date_str = most_recent_date or started_date
#
#                 if not date_str:
#                     continue
#
#                 try:
#                     date = parser.parse(date_str)
#                     if latest_date is None or date > latest_date:
#                         latest_date = date
#                         latest_url = metadata.get('url')
#                 except Exception as e:
#                     print(f"Error parsing date {date_str}: {e}")
#
#             if latest_date:
#                 latest_date_str = latest_date.strftime('%Y-%m-%d')
#                 print(f"Latest date in Pinecone: {latest_date_str}")
#                 print(f"URL with latest date: {latest_url}")
#                 return latest_date_str
#             else:
#                 print("No valid dates found in Pinecone.")
#                 return None
#         else:
#             print("No records found in Pinecone.")
#             return None
#     except Exception as e:
#         print(f"Error getting latest date from Pinecone: {e}")
#         return None

def parse_date_string(date_str: str) -> str:
    """Parse different date formats and return YYYY-MM-DD."""
    try:
        date_str = date_str.strip()

        # Handle relative dates FIRST (before removing time)
        # Check for "Yesterday at HH:MM:SS PM" or just "Yesterday"
        if date_str.lower().startswith("yesterday"):
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            return yesterday.strftime('%Y-%m-%d')

        if date_str.lower().startswith("today"):
            today = datetime.now(timezone.utc)
            return today.strftime('%Y-%m-%d')

        # Handle "X days ago at...", "X hours ago at...", etc.
        time_ago_match = re.match(r'(\d+)\s+(day|hour|week)s?\s+ago', date_str.lower())
        if time_ago_match:
            amount = int(time_ago_match.group(1))
            unit = time_ago_match.group(2)

            if unit == 'day':
                date_obj = datetime.now(timezone.utc) - timedelta(days=amount)
            elif unit == 'hour':
                date_obj = datetime.now(timezone.utc) - timedelta(hours=amount)
            elif unit == 'week':
                date_obj = datetime.now(timezone.utc) - timedelta(weeks=amount)

            return date_obj.strftime('%Y-%m-%d')

        # Remove time portion: ", 08:15:55 PM" or " at 08:15:55 PM"
        date_str = re.sub(r'(\s+at\s+|,\s*)\d{1,2}:\d{2}(:\d{2})?\s*[AP]M', '', date_str, flags=re.IGNORECASE)

        date_patterns = [
            (r'([A-Za-z]+ \d+, \d{4})', '%B %d, %Y'),
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
            (r'(\d{1,2}/\d{1,2}/\d{4})', '%m/%d/%Y'),
        ]

        for pattern, date_format in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                date_part = match.group(1)
                try:
                    date_obj = datetime.strptime(date_part, date_format)
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue

        return None

    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

async def extract_board_info(html: str, base_url: str, latest_date_obj=None):
    """Extract board URLs and their last post dates."""
    boards = []
    recent_boards = []

    try:
        soup = BeautifulSoup(html, 'html.parser')

        board_links = soup.find_all('a', class_='subject mobile_subject')
        print(f"Found {len(board_links)} board links")

        lastpost_divs = soup.find_all('div', class_='lastpost')
        print(f"Found {len(lastpost_divs)} lastpost divs")

        for i, board_link in enumerate(board_links):
            board_name = board_link.get_text(strip=True)
            board_url = board_link.get('href', '')

            if not board_name or not board_url:
                continue

            if not board_url.startswith('http'):
                board_url = base_url + '/' + board_url.lstrip('/')

            last_post_date = None

            if i < len(lastpost_divs):
                lastpost_div = lastpost_divs[i]
                strong_tag = lastpost_div.find('strong', string=re.compile(r'Last post:', re.IGNORECASE))
                if strong_tag:
                    p_tag = strong_tag.find_parent('p')
                    if p_tag:
                        full_text = p_tag.get_text(strip=True)
                        date_text = re.sub(r'Last post:\s*', '', full_text, flags=re.IGNORECASE)
                        date_text = re.split(r'\s+by\s+', date_text, flags=re.IGNORECASE)[0]
                        last_post_date = parse_date_string(date_text)

            board_info = {
                'url': board_url,
                'name': board_name,
                'last_post_date': last_post_date
            }

            boards.append(board_info)

            if not latest_date_obj:
                recent_boards.append(board_info)
                continue

            if not last_post_date:
                recent_boards.append(board_info)
                continue

            try:
                board_date = parser.parse(last_post_date)
                if board_date > latest_date_obj:
                    print(f"  Found recent board: {board_name} with date {last_post_date}")
                    recent_boards.append(board_info)
            except Exception as e:
                print(f"  Error comparing dates for {board_name}: {e}")
                recent_boards.append(board_info)

        print(f"Found {len(boards)} total boards")
        print(f"Found {len(recent_boards)} boards with recent activity\n")

    except Exception as e:
        print(f"Error extracting board information: {e}")
        import traceback
        traceback.print_exc()

    return recent_boards

async def extract_topic_info(html: str, base_url: str):
    """Extract topic URLs, subjects, and dates from a board page."""
    topics = []

    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Find all links with topic,XXXX in the URL
        all_links = soup.find_all('a', href=re.compile(r'topic,\d+'))

        # Group by topic ID to match subject with date
        topics_dict = {}

        for link in all_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)

            if not href or not text:
                continue

            # Extract topic ID
            topic_match = re.search(r'topic,(\d+)', href)
            if not topic_match:
                continue

            topic_id = topic_match.group(1)

            # Skip pagination links
            if text.lower() in ['last', 'first', 'next', 'prev', '«', '»'] or text.isdigit():
                continue

            # Make URL absolute
            if not href.startswith('http'):
                href = base_url + '/' + href.lstrip('/')

            # Initialize topic if new
            if topic_id not in topics_dict:
                topics_dict[topic_id] = {
                    'url': f"{base_url}/index.php/topic,{topic_id}.0.html",
                    'subject': None,
                    'most_recent_date': None,
                    'started_date': None
                }

            # Check if this is a date link (has #msg or .msg in URL)
            if '#msg' in href or '.msg' in href:
                # This is a date link
                parsed_date = parse_date_string(text)
                if parsed_date:
                    topics_dict[topic_id]['most_recent_date'] = parsed_date
            else:
                # This is a subject link (if long enough and not a keyword)
                if len(text) > 5 and 'pages' not in text.lower():
                    topics_dict[topic_id]['subject'] = text

        # Convert to list (only topics with subjects)
        for topic_id, topic_data in topics_dict.items():
            if topic_data['subject']:
                topics.append(topic_data)

        print(f"  Extracted {len(topics)} topics")

    except Exception as e:
        print(f"Error extracting topic information: {e}")
        import traceback
        traceback.print_exc()

    return topics

async def get_pagination_info(page):
    """Extract pagination information from the current page."""
    try:
        nav_pages = await page.query_selector_all('a.navPages')

        if not nav_pages or len(nav_pages) == 0:
            return None, 1

        current_page_element = await page.query_selector('span.current_page')
        max_pages = 1

        if current_page_element:
            text = await current_page_element.text_content()
            match = re.search(r'Page (\d+) of (\d+)', text)
            if match:
                current_page = int(match.group(1))
                max_pages = int(match.group(2))
                return current_page, max_pages

        for nav_page in nav_pages:
            href = await nav_page.get_attribute('href')
            if href:
                match = re.search(r'board=\d+\.(\d+)', href)
                if match:
                    page_offset = int(match.group(1))
                    potential_page = (page_offset // 20) + 1
                    max_pages = max(max_pages, potential_page)

        return None, max_pages

    except Exception as e:
        print(f"  Error getting pagination info: {e}")
        return None, 1

async def navigate_to_specific_page(page, board_url, page_num):
    """Navigate to a specific page of a board."""
    try:
        match = re.search(r'board=(\d+)', board_url)
        if not match:
            # Try board,X format
            match = re.search(r'board,(\d+)', board_url)

        if not match:
            return False

        board_num = match.group(1)
        offset = (page_num - 1) * 20

        # Handle both URL formats
        if 'index.php/' in board_url:
            page_url = f"{BASE_URL}/index.php/board,{board_num}.{offset}.html"
        else:
            page_url = f"{BASE_URL}/index.php?board={board_num}.{offset}"

        await page.goto(page_url, timeout=30000)
        await page.wait_for_load_state('networkidle', timeout=30000)

        return True
    except Exception as e:
        print(f"  Error navigating to page {page_num}: {e}")
        return False

async def process_board(page, board, latest_date_obj, new_topics):
    """Process a single board to find new topics."""
    try:
        print(f"Processing board: {board['name']}")

        await page.goto(board['url'], timeout=30000)
        await page.wait_for_load_state('networkidle', timeout=30000)

        _, max_pages = await get_pagination_info(page)
        print(f"  Found {max_pages} pages")

        for page_num in range(1, max_pages + 1):
            print(f"  Processing page {page_num}/{max_pages}")

            if page_num > 1:
                success = await navigate_to_specific_page(page, board['url'], page_num)
                if not success:
                    print(f"  Failed to navigate to page {page_num}")
                    break

            html = await page.content()
            page_topics = await extract_topic_info(html, BASE_URL)

            page_has_new_topics = False
            for topic in page_topics:
                if not topic['most_recent_date']:
                    # No date found, include it to be safe
                    new_topics.append(topic)
                    page_has_new_topics = True
                    print(f"    New: '{topic['subject'][:60]}' (no date)")
                    continue

                topic_most_recent_date = parser.parse(topic['most_recent_date'])

                if not latest_date_obj or topic_most_recent_date > latest_date_obj:
                    new_topics.append(topic)
                    page_has_new_topics = True
                    print(f"    New: '{topic['subject'][:60]}' ({topic['most_recent_date']})")

            if not page_has_new_topics and latest_date_obj:
                print(f"  No new topics on page {page_num}, stopping board scan")
                break

            await asyncio.sleep(1)

    except Exception as e:
        print(f"Error processing board {board['name']}: {e}")
        import traceback
        traceback.print_exc()

async def crawl_new_content(latest_date_in_pinecone):
    """Crawl forum boards and topics newer than the latest date in Pinecone."""
    latest_date_obj = parser.parse(latest_date_in_pinecone) if latest_date_in_pinecone else None
    new_topics = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("Fetching main forum page...")
        await page.goto(BASE_URL, timeout=30000)
        await page.wait_for_load_state('networkidle', timeout=30000)
        html = await page.content()

        recent_boards = await extract_board_info(html, BASE_URL, latest_date_obj)

        print(f"Processing {len(recent_boards)} boards with recent activity\n")

        for board in recent_boards:
            await process_board(page, board, latest_date_obj, new_topics)
            print()

        await browser.close()

    return new_topics

async def main():
     # First, try to load from state file
    latest_date = load_latest_date()

    # If no state file exists, scan Pinecone (this is slow)
    if not latest_date:
        print("No state file found. Scanning Pinecone for latest date...")
        latest_date = await get_latest_date_from_pinecone()

        # Save it for next time
        if latest_date:
            save_latest_date(latest_date)

    # Now do your incremental crawl from latest_date
    if latest_date:
        print(f"Running incremental update from {latest_date}")
        # new_data = await crawl_new_content(since_date=latest_date)
    # latest_date = await get_latest_date_from_pinecone() #"2025-07-14" #
    #
    # if not latest_date:
    #     print("No latest date found in Pinecone. Running full crawl.\n")

    start_time = time.time()
    new_topics = await crawl_new_content(latest_date)
    end_time = time.time()

    print(f"\n{'='*60}")
    print(f"Found {len(new_topics)} new topics in {end_time - start_time:.2f} seconds")
    print(f"{'='*60}\n")

    if new_topics:
        output_file = 'new_topic_surface_hippy.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_topics, f, indent=2, ensure_ascii=False)

        print(f"New data saved to {output_file}\n")

        print("First new topic:")
        print(json.dumps(new_topics[0], indent=2))

        from collections import defaultdict
        topics_by_date = defaultdict(int)
        for topic in new_topics:
            if topic['most_recent_date']:
                topics_by_date[topic['most_recent_date']] += 1

        print("\nNew topics by date:")
        for date, count in sorted(topics_by_date.items()):
            print(f"  {date}: {count} topics")

        print("\nRunning ingestion script...")
        try:
            from pinecone_ingestion import setup_clients, crawl_parallel

            openai_client = await setup_clients()
            await crawl_parallel(new_topics, openai_client)

            print("\nIngestion completed!")
        except ImportError:
            print(f"\nRun ingestion manually:")
            print(f"python updated_pinecone_ingestion.py --input {output_file}")
    else:
        print("No new topics found. Database is up to date.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n✅ Pinecone ingestion completed successfully")
        sys.exit(0)  # Exit with success code
    except Exception as e:
        print(f"\n❌ Pinecone ingestion FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Exit with error code - this tells GitHub Actions it failed
