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
        # Handle format: Jan 22 (current year assumed)
        if len(date_str) <= 6:  # like "Jan 22"
            current_year = "2025"  # hardcoded current year
            date_str = f"{date_str}, {current_year}"
            date_obj = datetime.strptime(date_str, '%b %d, %Y')
        # Handle format: 10/05/24
        elif '/' in date_str:
            date_obj = datetime.strptime(date_str, '%m/%d/%y')
        else:
            return None
        
        return date_obj.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

async def extract_topic_info(html: str, base_group_url: str) -> List[Dict]:
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
            
            # Get thread attribution using the successful approach from our test
            thread_attr = topic_cell.find('span', class_='thread-attribution')
            if thread_attr:
                print("\nDEBUG: Raw thread attribution HTML:", thread_attr)  # Debug print
                
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

async def extract_next_page_info(html: str) -> str:
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

async def get_topic_info() -> List[Dict]:
    """Fetch all topic information using pagination using Playwright."""
    base_group_url = "https://groups.io/g/Hipresurfacingsite"
    base_url = f"{base_group_url}/topics"
    all_topics = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        current_url = base_url
        page_num = 1
        seen_tokens = set()

        while True:
            print(f"\nFetching page {page_num}: {current_url}")
            
            try:
                # Navigate to the page and wait for the table to load
                await page.goto(current_url)
                await page.wait_for_selector('table.table-condensed')
                
                # Get the fully rendered HTML
                html = await page.content()
                
                page_topics = await extract_topic_info(html, base_group_url)
                all_topics.extend(page_topics)
                print(f"Found {len(page_topics)} topics on page {page_num}")

                next_token = await extract_next_page_info(html)

                if not next_token or next_token in seen_tokens:
                    print("No more pages to fetch")
                    break

                seen_tokens.add(next_token)
                current_url = f"{base_url}?page={page_num + 1}&after={next_token}"
                page_num += 1

                await asyncio.sleep(1)  # Be nice to the server
                
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                break

        await browser.close()

    return all_topics

async def main():
    print("Starting crawler...")
    start_time = time.time()
    topics = await get_topic_info()
    end_time = time.time()

    print(f"\nFound {len(topics)} topics in {end_time - start_time:.2f} seconds")
    
    output_file = 'topic_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    
    print(f"\nData saved to {output_file}")
    if topics:
        print("\nFirst topic data:")
        print(json.dumps(topics[0], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
