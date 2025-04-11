import asyncio
from playwright.async_api import async_playwright
import re
import json
from typing import List, Dict
from bs4 import BeautifulSoup
import time
from datetime import datetime

def parse_date_string(date_str: str) -> str:
    """Parse different date formats and return YYYY-MM-DD."""
    try:
        # Handle format: May 11, 2007, 04:17:30 PM
        if ',' in date_str and ':' in date_str:
            # Extract the date part before the time
            date_parts = date_str.split(',')
            if len(date_parts) >= 2:
                date_part = date_parts[0] + ',' + date_parts[1]
                date_obj = datetime.strptime(date_part.strip(), '%B %d, %Y')
                return date_obj.strftime('%Y-%m-%d')
        return None
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

async def extract_topic_info(html: str, base_url: str) -> List[Dict]:
    """Extract topic URLs, subjects, and dates from a page."""
    topics = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all topic rows - they might be in table rows with windowbg or stickybg classes
        topic_rows = soup.find_all('tr')
        
        for row in topic_rows:
            # Find the subject cell
            subject_cell = row.find('td', class_=lambda c: c and ('subject' in c))
            if not subject_cell:
                continue
                
            # Find the link within a span with id starting with "msg_"
            msg_span = subject_cell.find('span', id=lambda i: i and i.startswith('msg_'))
            if not msg_span:
                continue
                
            subject_link = msg_span.find('a')
            if not subject_link:
                continue
                
            subject = subject_link.text.strip()
            topic_url = subject_link['href']
            if not topic_url.startswith('http'):
                topic_url = base_url + topic_url.lstrip('/')
            
            # Get most recent date from lastpost cell in the same row
            lastpost_cell = row.find('td', class_='lastpost')
            most_recent_date = None
            
            if lastpost_cell:
                # The date is typically between the image and the "by" text
                for element in lastpost_cell.contents:
                    if isinstance(element, str) and element.strip() and 'by' not in element:
                        date_text = element.strip()
                        most_recent_date = parse_date_string(date_text)
                        break
            
            # Only add if we have valid data
            if subject and topic_url:
                topics.append({
                    'url': topic_url,
                    'subject': subject,
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

async def process_board(playwright, base_url: str, board_url: str) -> List[Dict]:
    """Process a single board to extract all topics."""
    browser = await playwright.chromium.launch()
    page = await browser.new_page()
    
    all_topics = []
    
    try:
        # Go to the first page of the board
        await page.goto(board_url)
        await page.wait_for_load_state('networkidle')
        
        # Determine how many pages there are
        _, max_pages = await get_pagination_info(page)
        print(f"Found {max_pages} pages for board: {board_url}")
        
        # Process each page
        for page_num in range(1, max_pages + 1):
            print(f"Processing page {page_num} of {max_pages} for board: {board_url}")
            
            # Navigate to the specific page (for page 1, we're already there)
            if page_num > 1:
                success = await navigate_to_specific_page(page, board_url, page_num)
                if not success:
                    print(f"Failed to navigate to page {page_num}, stopping processing for this board")
                    break
            
            # Get the fully rendered HTML
            html = await page.content()
            
            # Extract topics from current page
            page_topics = await extract_topic_info(html, base_url)
            all_topics.extend(page_topics)
            print(f"Found {len(page_topics)} topics on page {page_num}")
            
            # Be nice to the server
            await asyncio.sleep(1)
    
    except Exception as e:
        print(f"Error processing board {board_url}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await browser.close()
    
    return all_topics

async def get_all_boards(html: str, base_url: str) -> List[Dict]:
    """Extract all board URLs from the main page."""
    boards = []
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all board links with class "subject"
        board_links = soup.find_all('a', class_='subject')
        
        for link in board_links:
            if 'href' in link.attrs:
                board_url = link['href']
                if not board_url.startswith('http'):
                    board_url = base_url + board_url.lstrip('/')
                
                boards.append({
                    'url': board_url,
                    'name': link.text.strip()
                })
    
    except Exception as e:
        print(f"Error extracting board links: {e}")
    
    return boards

async def main():
    print("Starting forum crawler...")
    start_time = time.time()
    
    base_url = "https://surfacehippy.info/hiptalk/"
    main_url = base_url
    
    all_topics = []
    
    async with async_playwright() as p:
        # First, get all board links
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        try:
            await page.goto(main_url)
            await page.wait_for_load_state('networkidle')
            
            html = await page.content()
            boards = await get_all_boards(html, base_url)
            
            print(f"Found {len(boards)} boards")
            
        finally:
            await browser.close()
        
        # Now process each board
        for board in boards:
            print(f"\nProcessing board: {board['name']}")
            board_topics = await process_board(p, base_url, board['url'])
            all_topics.extend(board_topics)
    
    end_time = time.time()
    print(f"\nFound {len(all_topics)} topics in total in {end_time - start_time:.2f} seconds")
    
    output_file = 'surfacehippy_topics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_topics, f, indent=2, ensure_ascii=False)
    
    print(f"\nData saved to {output_file}")
    if all_topics:
        print("\nFirst topic data:")
        print(json.dumps(all_topics[0], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
