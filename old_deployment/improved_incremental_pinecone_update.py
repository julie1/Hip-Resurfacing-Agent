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
        # Handle format: November 17, 2020, 07:24:51 PM
        match = re.search(r'([A-Za-z]+ \d+, \d{4})', date_str)
        if match:
            date_part = match.group(1)
            date_obj = datetime.strptime(date_part, '%B %d, %Y')
            return date_obj.strftime('%Y-%m-%d')
        return None
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None


async def extract_board_info(html: str, base_url: str, latest_date_obj=None):
    """Extract board URLs and their last post dates, filtering by latest date if provided."""
    boards = []
    recent_boards = []
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Find all td elements with class 'lastpost' as these contain date info
        lastpost_cells = soup.find_all('td', class_='lastpost')

        for lastpost_cell in lastpost_cells:
            # Find the parent row
            row = lastpost_cell.find_parent('tr')
            if not row:
                continue

            # Find the info cell in the same row
            info_cell = row.find('td', class_='info')
            if not info_cell:
                continue

            # Find the board link in the info cell
            board_link = info_cell.find('a')
            if not board_link:
                continue

            board_name = board_link.text.strip()
            board_url = board_link['href']
            if not board_url.startswith('http'):
                board_url = base_url + '/' + board_url.lstrip('/')

            # Extract date information from the lastpost cell
            last_post_text = lastpost_cell.get_text(strip=True)
            last_post_date = None

            # Look for "on Month Day, Year" pattern
            match = re.search(r'on ([A-Za-z]+ \d+, \d{4})', last_post_text)
            if match:
                date_part = match.group(1)
                try:
                    date_obj = datetime.strptime(date_part, '%B %d, %Y')
                    last_post_date = date_obj.strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"Error parsing date for {board_name}: {e}")

            board = {
                'url': board_url,
                'name': board_name,
                'last_post_date': last_post_date
            }

            boards.append(board)

            # Add to recent boards if no date filter or if date is newer
            if not latest_date_obj:
                recent_boards.append(board)
                continue

            if not last_post_date:
                # If we can't determine date, include it to be safe
                recent_boards.append(board)
                continue

            try:
                board_date = parser.parse(last_post_date)
                if board_date > latest_date_obj:
                    print(f"Found recent board: {board_name} with date {last_post_date}")
                    recent_boards.append(board)
            except Exception as e:
                print(f"Error comparing dates for {board_name}: {e}")
                # Include board if date parsing failed (to be safe)
                recent_boards.append(board)

        print(f"\nFound {len(boards)} total boards")
        print(f"Found {len(recent_boards)} boards with recent activity")

    except Exception as e:
        print(f"Error extracting board information: {e}")
        import traceback
        traceback.print_exc()

    return recent_boards

async def extract_topic_info(html: str, base_url: str):
    """Extract topic URLs, subjects, and dates from a page."""
    topics = []
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Find all topic rows
        topic_rows = soup.find_all('tr')

        for row in topic_rows:
            # Find the subject cell
            subject_cell = row.find('td', class_=lambda c: c and ('subject' in c if c else False))
            if not subject_cell:
                continue

            # Find the link within a span with id starting with "msg_"
            msg_span = subject_cell.find('span', id=lambda i: i and i.startswith('msg_') if i else False)
            if not msg_span:
                continue

            subject_link = msg_span.find('a')
            if not subject_link:
                continue

            subject = subject_link.text.strip()
            topic_url = subject_link['href']
            if not topic_url.startswith('http'):
                topic_url = base_url + '/' + topic_url.lstrip('/')

            # Get started date (might be in a different location)
            started_date = None

            # Get most recent date from lastpost cell in the same row
            lastpost_cell = row.find('td', class_='lastpost')
            most_recent_date = None

            if lastpost_cell:
                # Extract date text
                date_text = lastpost_cell.text.strip()
                most_recent_date = parse_date_string(date_text)

            # Only add if we have valid data
            if subject and topic_url:
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

            print(f"Found {len(page_topics)} topics on page {page_num}, " +
                (f"including {len([t for t in page_topics if t in new_topics])} new topics" if latest_date_obj else "processing all topics"))

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
