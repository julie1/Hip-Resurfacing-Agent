import asyncio
import json
import os
import time
import re
from datetime import datetime
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

async def get_latest_date_from_pinecone():
    """Retrieve the latest date from Pinecone database."""
    try:
        # Connect to the index
        index = pc.Index(INDEX_NAME)
        print(f"Connected to index: {INDEX_NAME}")

        # Query for all records, sorting by most_recent_date if possible
        # Since Pinecone doesn't support direct sorting, we'll fetch records and sort manually
        results = index.query(
            namespace=NAMESPACE,
            vector=[0.0] * 1536,  # Dummy vector for metadata-only query
            include_metadata=True,
            top_k=1000  # Adjust based on expected collection size
        )

        if results and 'matches' in results and results['matches']:
            # Find the point with the latest date manually
            latest_date = None
            latest_url = None

            for point in results['matches']:
                metadata = point.get('metadata', {})
                most_recent_date = metadata.get('most_recent_date')
                started_date = metadata.get('started_date')

                # Use the most_recent_date if available, otherwise try started_date
                date_str = most_recent_date or started_date

                if not date_str:
                    continue

                try:
                    date = parser.parse(date_str)
                    if latest_date is None or date > latest_date:
                        latest_date = date
                        latest_url = metadata.get('url')
                except Exception as e:
                    print(f"Error parsing date {date_str}: {e}")

            if latest_date:
                latest_date_str = latest_date.strftime('%Y-%m-%d')
                print(f"Latest date in Pinecone: {latest_date_str}")
                print(f"URL with latest date: {latest_url}")
                return latest_date_str
            else:
                print("No valid dates found in Pinecone.")
                return None
        else:
            print("No records found in Pinecone.")
            return None
    except Exception as e:
        print(f"Error getting latest date from Pinecone: {e}")
        return None

def parse_date_string(date_str: str) -> str:
    """Parse different date formats and return YYYY-MM-DD."""
    try:
        # Handle format: June 24, 2025, 11:18:12 AM
        # Extract just the date part before the comma and time
        date_patterns = [
            r'([A-Za-z]+ \d+, \d{4})',  # "June 24, 2025"
            r'(\d{4}-\d{2}-\d{2})',     # "2025-06-24"
            r'(\d{1,2}/\d{1,2}/\d{4})', # "6/24/2025"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                date_part = match.group(1)
                try:
                    if '/' in date_part:
                        date_obj = datetime.strptime(date_part, '%m/%d/%Y')
                    else:
                        date_obj = datetime.strptime(date_part, '%B %d, %Y')
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        
        return None
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

async def extract_board_info(html: str, base_url: str, latest_date_obj=None):
    """Extract board URLs and their last post dates using the updated structure."""
    boards = []
    recent_boards = []
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for board links with the new class structure
        board_links = soup.find_all('a', class_='subject mobile_subject')
        print(f"Found {len(board_links)} board links with 'subject mobile_subject' class")
        
        # Also look for lastpost divs
        lastpost_divs = soup.find_all('div', class_='lastpost')
        print(f"Found {len(lastpost_divs)} lastpost divs")
        
        # Try to match board links with their corresponding lastpost info
        # We need to find them in the same parent structure
        for i, board_link in enumerate(board_links):
            board_name = board_link.get_text(strip=True)
            board_url = board_link.get('href', '')
            
            if not board_name or not board_url:
                continue
            
            # Make sure URL is absolute
            if not board_url.startswith('http'):
                board_url = base_url + '/' + board_url.lstrip('/')
            
            # Find the corresponding lastpost div
            # Since they're in parallel structures, try to match by index first
            last_post_date = None
            
            if i < len(lastpost_divs):
                lastpost_div = lastpost_divs[i]
                lastpost_text = lastpost_div.get_text(strip=True)
                
                # Look for "Last post: DATE" pattern
                date_match = re.search(r'Last post:\s*([^by]+?)(?:\s+by\s+|\s*$)', lastpost_text)
                if date_match:
                    date_part = date_match.group(1).strip()
                    # Remove time portion if present
                    date_part = re.sub(r',\s*\d{1,2}:\d{2}:\d{2}\s*[AP]M', '', date_part)
                    last_post_date = parse_date_string(date_part)
            
            board_info = {
                'url': board_url,
                'name': board_name,
                'last_post_date': last_post_date
            }
            
            boards.append(board_info)
            
            # Check if this board has recent activity
            if not latest_date_obj:
                recent_boards.append(board_info)
                continue
            
            if not last_post_date:
                # Include boards where we can't determine date (to be safe)
                recent_boards.append(board_info)
                continue
            
            try:
                board_date = parser.parse(last_post_date)
                if board_date > latest_date_obj:
                    print(f"Found recent board: {board_name} with date {last_post_date}")
                    recent_boards.append(board_info)
            except Exception as e:
                print(f"Error comparing dates for {board_name}: {e}")
                recent_boards.append(board_info)
        
        print(f"Found {len(boards)} total boards")
        print(f"Found {len(recent_boards)} boards with recent activity")
        
    except Exception as e:
        print(f"Error extracting board information: {e}")
        import traceback
        traceback.print_exc()
    
    return recent_boards

async def extract_topic_info(html: str, base_url: str):
    """Extract topic URLs, subjects, and dates from a board page using updated structure."""
    topics = []
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for topic links directly - topics have URLs with 'topic=' parameter
        topic_links = soup.find_all('a', href=lambda href: href and 'topic=' in href)
        
        # Group links by their parent rows to avoid duplicates and get associated data
        processed_urls = set()
        
        for link in topic_links:
            topic_url = link.get('href', '')
            subject = link.get_text(strip=True)
            
            if not subject or not topic_url or topic_url in processed_urls:
                continue
            
            # Skip navigation links and other non-topic links
            if any(skip in subject.lower() for skip in ['last', 'first', 'next', 'prev', '«', '»']):
                continue
            
            # Make URL absolute
            if not topic_url.startswith('http'):
                topic_url = base_url + '/' + topic_url.lstrip('/')
            
            processed_urls.add(topic_url)
            
            # Look for date information in the same row or nearby elements
            started_date = None
            most_recent_date = None
            
            # Find parent row or container
            parent_row = link.find_parent(['tr', 'div', 'article', 'section'])
            if parent_row:
                # Look for lastpost information in the same row
                lastpost_element = parent_row.find(['td', 'div'], class_=lambda c: c and 'lastpost' in str(c).lower())
                if lastpost_element:
                    date_text = lastpost_element.get_text(strip=True)
                    most_recent_date = parse_date_string(date_text)
                
                # If no specific lastpost element, look for any date in the row
                if not most_recent_date:
                    row_text = parent_row.get_text()
                    # Look for date patterns
                    date_patterns = [
                        r'([A-Za-z]+ \d+, \d{4})',
                        r'(\d{4}-\d{2}-\d{2})',
                        r'(\d{1,2}/\d{1,2}/\d{4})'
                    ]
                    
                    for pattern in date_patterns:
                        matches = re.findall(pattern, row_text)
                        if matches:
                            # Use the last date found (likely the most recent)
                            most_recent_date = parse_date_string(matches[-1])
                            break
            
            topic_info = {
                'url': topic_url,
                'subject': subject,
                'started_date': started_date,
                'most_recent_date': most_recent_date
            }
            
            topics.append(topic_info)
        
        print(f"Extracted {len(topics)} unique topics")
        
    except Exception as e:
        print(f"Error extracting topic information: {e}")
        import traceback
        traceback.print_exc()
    
    return topics

async def get_pagination_info(page):
    """Extract pagination information from the current page."""
    try:
        # Look for pagination elements
        nav_pages = await page.query_selector_all('a.navPages')

        # Check if we have pagination at all
        if not nav_pages or len(nav_pages) == 0:
            return None, 1

        # Try to find the current page and max pages
        # First, try to find elements showing the current page
        current_page_element = await page.query_selector('span.current_page')
        max_pages = 1

        if current_page_element:
            # Format is often "Page X of Y"
            text = await current_page_element.text_content()
            match = re.search(r'Page (\d+) of (\d+)', text)
            if match:
                current_page = int(match.group(1))
                max_pages = int(match.group(2))
                return current_page, max_pages

        # Alternative: Look at URLs of pagination links to determine max pages
        for nav_page in nav_pages:
            href = await nav_page.get_attribute('href')
            if href:
                # URLs often have format board=X.Y where Y is related to page number
                match = re.search(r'board=\d+\.(\d+)', href)
                if match:
                    page_offset = int(match.group(1))
                    # Usually the page offset is (page_number - 1) * items_per_page
                    # Estimate max page based on highest offset seen
                    potential_page = (page_offset // 20) + 1  # Assuming 20 items per page
                    max_pages = max(max_pages, potential_page)

        return None, max_pages

    except Exception as e:
        print(f"Error getting pagination info: {e}")
        return None, 1

async def navigate_to_specific_page(page, board_url, page_num):
    """Navigate to a specific page of a board."""
    try:
        # Extract the board number from the URL
        match = re.search(r'board=(\d+)', board_url)
        if not match:
            return False

        board_num = match.group(1)

        # Calculate the offset based on page number (20 items per page is common)
        offset = (page_num - 1) * 20

        # Construct the URL for the specific page
        page_url = f"{board_url.split('?')[0]}?board={board_num}.{offset}"

        # Navigate to the page
        await page.goto(page_url)
        await page.wait_for_load_state('networkidle')

        return True
    except Exception as e:
        print(f"Error navigating to page {page_num}: {e}")
        return False

async def process_board(page, board, latest_date_obj, new_topics):
    """Process a single board to find new topics."""
    try:
        # Go to the first page of the board
        await page.goto(board['url'])
        await page.wait_for_load_state('networkidle')

        # Determine how many pages there are
        _, max_pages = await get_pagination_info(page)
        print(f"Found {max_pages} pages for board: {board['name']}")

        # Process each page
        for page_num in range(1, max_pages + 1):
            print(f"Processing page {page_num} of {max_pages} for board: {board['name']}")

            # Navigate to the specific page (for page 1, we're already there)
            if page_num > 1:
                success = await navigate_to_specific_page(page, board['url'], page_num)
                if not success:
                    print(f"Failed to navigate to page {page_num}, stopping processing for this board")
                    break

            # Get the fully rendered HTML
            html = await page.content()

            # Extract topics from current page
            page_topics = await extract_topic_info(html, BASE_URL)

            # Check for new topics
            page_has_new_topics = False
            for topic in page_topics:
                topic_most_recent_date = parser.parse(topic['most_recent_date']) if topic['most_recent_date'] else None

                # Add topic if it's newer than our latest date
                if not latest_date_obj or (topic_most_recent_date and topic_most_recent_date > latest_date_obj):
                    new_topics.append(topic)
                    page_has_new_topics = True
                    print(f"  Found new topic: '{topic['subject']}' - {topic['most_recent_date']}")

            print(f"Found {len(page_topics)} topics on page {page_num}, " +
                f"including {len([t for t in page_topics if t in new_topics])} new topics")

            # If this page had no new topics, we can stop processing this board
            if not page_has_new_topics and latest_date_obj:
                print(f"No new topics found on page {page_num}. Moving to next board.")
                break

            # Be nice to the server
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Error processing board {board['name']}: {e}")
        import traceback
        traceback.print_exc()

async def crawl_new_content(latest_date_in_pinecone):
    """Crawl forum boards and topics that are newer than the latest date in Pinecone."""
    latest_date_obj = parser.parse(latest_date_in_pinecone) if latest_date_in_pinecone else None
    new_topics = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # First get the main forum page
        print("Fetching main forum page...")
        await page.goto(BASE_URL)
        await page.wait_for_load_state('networkidle')
        html = await page.content()

        # Extract only boards with recent activity
        recent_boards = await extract_board_info(html, BASE_URL, latest_date_obj)

        print(f"Processing {len(recent_boards)} boards with recent activity")

        # Now process each recent board
        for board in recent_boards:
            print(f"\nProcessing board: {board['name']} - Last post date: {board['last_post_date']}")
            await process_board(page, board, latest_date_obj, new_topics)

        await browser.close()

    return new_topics

async def main():
    # Get the latest date from Pinecone
    latest_date = await get_latest_date_from_pinecone()

    if not latest_date:
        print("No latest date found in Pinecone. Will run a full crawl instead.")

    # Crawl new topics with the improved approach
    start_time = time.time()
    new_topics = await crawl_new_content(latest_date)
    end_time = time.time()

    print(f"\nFound {len(new_topics)} new topics in {end_time - start_time:.2f} seconds")

    if new_topics:
        # Save new topics to a file
        output_file = 'new_topic_surface_hippy.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_topics, f, indent=2, ensure_ascii=False)

        print(f"\nNew data saved to {output_file}")

        # Print the first new topic for verification
        print("\nFirst new topic data:")
        print(json.dumps(new_topics[0], indent=2))

        # Print summary of new topics by date
        from collections import defaultdict
        topics_by_date = defaultdict(int)
        for topic in new_topics:
            if topic['most_recent_date']:
                topics_by_date[topic['most_recent_date']] += 1
        
        print("\nNew topics by date:")
        for date, count in sorted(topics_by_date.items()):
            print(f"  {date}: {count} topics")

        # Now you can run your ingestion script with only the new topics
        print("\nRunning ingestion script with new topics...")

        # Import the ingestion code and run the ingestion with only the new topics
        try:
            from pinecone_ingestion import setup_clients, crawl_parallel

            openai_client = await setup_clients()
            await crawl_parallel(new_topics, openai_client)

            print("\nIngestion of new topics completed.")
        except ImportError:
            print("\nPlease run ingestion manually using the new_topic_surface_hippy.json file:")
            print("python updated_pinecone_ingestion.py --input new_topic_surface_hippy.json")
    else:
        print("No new topics found. Database is up to date.")

if __name__ == "__main__":
    asyncio.run(main())
