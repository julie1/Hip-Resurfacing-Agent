import asyncio
import random
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
# Forum configuration  (moved to .net after hack)
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



def extract_and_save_latest_date(topics):
    """
    Extract the newest date from topics list and save to state file.
    Call this AFTER successful ingestion.

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



def parse_date_string(date_str: str) -> str:
    """Parse different date formats and return YYYY-MM-DD."""
    try:
        date_str = date_str.strip()

        # Handle relative dates FIRST
        if date_str.lower().startswith("yesterday"):
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            return yesterday.strftime('%Y-%m-%d')

        if date_str.lower().startswith("today"):
            today = datetime.now(timezone.utc)
            return today.strftime('%Y-%m-%d')

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

        # FIX: Strip out the time parts without altering critical separating commas
        # Converts "June 13, 2026, 05:40:26 AM" cleanly into "June 13, 2026"
        date_str = re.sub(r',\s*\d{1,2}:\d{2}(:\d{2})?\s*[AP]M', '', date_str, flags=re.IGNORECASE)
        date_str = re.sub(r'\s+at\s+\d{1,2}:\d{2}(:\d{2})?\s*[AP]M', '', date_str, flags=re.IGNORECASE)
        date_str = date_str.strip()

        date_patterns = [
            (r'([A-Za-z]+ \d+, \d{4})', '%B %d, %Y'), # Handles "June 13, 2026"
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
    """Fast, index-matched board extractor that maintains your 3-minute runtime."""
    boards = []
    recent_boards = []

    try:
        soup = BeautifulSoup(html, 'html.parser')

        # 1. Gather all board links using your verified selector
        board_links = soup.find_all('a', class_='subject mobile_subject')
        if not board_links:
            board_links = soup.find_all('a', href=re.compile(r'board,\d+', re.I))
            board_links = [a for a in board_links if len(a.get_text(strip=True)) > 4]

        # 2. Inline helper to extract date cleanly from target sibling text node
        def extract_date_from_lastpost_p(p_tag):
            strong = p_tag.find('strong', string=re.compile(r'Last\s+post', re.I))
            if not strong:
                # Handle cases where the text node might not be perfectly bound by <strong>
                strong = p_tag.find(lambda tag: tag.name == 'strong' and 'last post' in tag.get_text().lower())
            
            if not strong:
                return None

            date_parts = []
            for sibling in strong.next_siblings:
                if hasattr(sibling, 'name') and sibling.name == 'span':
                    break
                text = str(sibling).strip()
                if text:
                    date_parts.append(text)

            raw = ' '.join(date_parts).strip()
            if not raw:
                full = p_tag.get_text(strip=True)
                raw = re.sub(r'(?i)Last\s+post:\s*', '', full)
                raw = re.split(r'\s+by\s+', raw, flags=re.IGNORECASE)[0].strip()

            return parse_date_string(raw) if raw else None

        # 3. Collect paragraph and division timestamp indicators in exact document order
        lastpost_p_tags = []
        for strong in soup.find_all('strong', string=re.compile(r'Last\s+post', re.I)):
            p = strong.find_parent('p')
            if p and p not in lastpost_p_tags:
                lastpost_p_tags.append(p)

        lastpost_divs = soup.find_all('div', class_=re.compile(r'lastpost', re.I))
        
        print(f"Found {len(board_links)} board links, "
              f"{len(lastpost_p_tags)} lastpost <p> tags, "
              f"{len(lastpost_divs)} lastpost <div> tags")

        # 4. Map timestamps back to links using parallel indexing
        for i, board_link in enumerate(board_links):
            board_name = board_link.get_text(strip=True)
            board_url = board_link.get('href', '')

            if not board_name or not board_url:
                continue

            if not board_url.startswith('http'):
                board_url = f"{base_url.rstrip('/')}/{board_url.lstrip('/')}"
            
            last_post_date = None

            # Attempt extraction via sequential array matching
            if i < len(lastpost_p_tags):
                last_post_date = extract_date_from_lastpost_p(lastpost_p_tags[i])

            if not last_post_date and i < len(lastpost_divs):
                p_tag = lastpost_divs[i].find('p')
                if p_tag:
                    last_post_date = extract_date_from_lastpost_p(p_tag)
                
                if not last_post_date:
                    full_text = lastpost_divs[i].get_text(strip=True)
                    date_text = re.sub(r'Last post:\s*', '', full_text, flags=re.IGNORECASE)
                    date_text = re.split(r'\s+by\s+', date_text, flags=re.IGNORECASE)[0]
                    last_post_date = parse_date_string(date_text)

            board_info = {
                'url': board_url,
                'name': board_name,
                'last_post_date': last_post_date
            }
            boards.append(board_info)

            # Strict comparison: do NOT fall back to crawling if the date is missing/old
            if last_post_date and latest_date_obj:
                try:
                    # Support both datetime and date object states smoothly
                    board_date = parser.parse(last_post_date)
                    target_date = latest_date_obj if isinstance(latest_date_obj, datetime) else datetime.combine(latest_date_obj, datetime.min.time())
                    
                    if board_date.date() >= target_date.date():
                        print(f"  Found recent active board: '{board_name}' ({last_post_date})")
                        recent_boards.append(board_info)
                except Exception as e:
                    print(f"  [DEBUG] Skipping conversion exception for {board_name}: {e}")
            elif not latest_date_obj:
                recent_boards.append(board_info)

        print(f"Total boards evaluated: {len(boards)}")
        print(f"Filtered down to {len(recent_boards)} active boards to crawl\n")

    except Exception as e:
        print(f"Error extracting board information: {e}")
        import traceback
        traceback.print_exc()

    return recent_boards




    def extract_date_from_lastpost_p(p_tag):
        """Pull the date text node that immediately follows <strong>Last post:</strong>."""
        strong = p_tag.find('strong', string=re.compile(r'Last\s+post', re.I))
        if not strong:
            return None

        # Walk siblings of <strong> inside <p>
        date_parts = []
        for sibling in strong.next_siblings:
            if hasattr(sibling, 'name'):
                # Hit a real tag (e.g. <span class="postby">) — stop
                break
            text = str(sibling).strip()
            if text:
                date_parts.append(text)

        raw = ' '.join(date_parts).strip()
        if not raw:
            # Fallback: grab full p text and strip the "Last post:" prefix + "by …" suffix
            full = p_tag.get_text(strip=True)
            raw = re.sub(r'(?i)Last\s+post:\s*', '', full)
            raw = re.split(r'\s+by\s+', raw, flags=re.IGNORECASE)[0].strip()

        return parse_date_string(raw) if raw else None

    # Build a flat list of all last-post <p> tags in document order
    lastpost_p_tags = []
    for strong in soup.find_all('strong', string=re.compile(r'Last\s+post', re.I)):
        p = strong.find_parent('p')
        if p and p not in lastpost_p_tags:
            lastpost_p_tags.append(p)

    # Also try the old <div class="lastpost"> structure as a secondary source
    lastpost_divs = soup.find_all('div', class_=re.compile(r'lastpost', re.I))
    print(f"Found {len(board_links)} board links, "
          f"{len(lastpost_p_tags)} lastpost <p> tags, "
          f"{len(lastpost_divs)} lastpost <div> tags")

    for i, board_link in enumerate(board_links):
        board_name = board_link.get_text(strip=True)
        board_url = board_link.get('href', '')

        if not board_name or not board_url:
            continue

        if not board_url.startswith('http'):
            board_url = base_url + '/' + board_url.lstrip('/')
        
        last_post_date = None

        # Try new-style <p> tags first
        if i < len(lastpost_p_tags):
            last_post_date = extract_date_from_lastpost_p(lastpost_p_tags[i])

        # Fall back to old <div class="lastpost"> approach
        if not last_post_date and i < len(lastpost_divs):
            p_tag = lastpost_divs[i].find('p')
            if p_tag:
                last_post_date = extract_date_from_lastpost_p(p_tag)
                if not last_post_date:
                    # Original text-scrape approach
                    full_text = lastpost_divs[i].get_text(strip=True)
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
            # Can't determine date — include to be safe
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

        


async def extract_topic_info(html: str, base_url: str, debug: bool = False):
    """Robust topic extractor that scales to broad table/grid layouts."""
    topics = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # Matches raw, parameterized, or SEO clean topic paths
        topic_link_pattern = re.compile(r'(topic[,=]\d+|\/topic\/\d+)', re.IGNORECASE)
        
        all_links = soup.find_all('a')
        matched_links = [a for a in all_links if a.get('href') and topic_link_pattern.search(a['href'])]
        
        if len(matched_links) == 0:
            print("\n🚨 [DIAGNOSTIC] No topics found. Checking for severe layout blocks...")
            page_title = soup.title.string.strip() if soup.title else "No Title Tag"
            print(f"  -> Server Document Title: '{page_title}'")
            print(f"  -> HTML Length: {len(html)} characters")
            body_text = ' '.join(soup.get_text().split())[:300]
            print(f"  -> Page Context Text: {body_text}\n")
            return topics

        topic_containers = {}
        for a in matched_links:
            href = a.get('href', '')
            id_match = re.search(r'topic[,=](\d+)', href)
            topic_id = id_match.group(1) if id_match else href.split('/')[-1]
            
            # Skip noise or utility tracking IDs
            if not topic_id:
                continue

            # FIX: Walk up the tree to find the macro row container (tr, li, or main row layout)
            # We ignore generic styling span/div blocks that wrap individual links closely.
            container = a
            current = a.parent
            while current is not None:
                # Check for table rows or modern responsive forum topic blocks
                if current.name in ('tr', 'li') or (current.name == 'div' and any(cls in ''.join(current.get('class', [])).lower() for cls in ['row', 'topic', 'windowbg', 'boardindex'])):
                    container = current
                    break
                current = current.parent
                
            # If no structured row macro element was discovered, default to its broader parent element
            if container == a and a.parent:
                container = a.parent.parent or a.parent

            topic_containers[topic_id] = container

        for topic_id, container in topic_containers.items():
            subject = None
            topic_url = None
            
            # Scrape links strictly residing within the clustered row container block
            for a in container.find_all('a'):
                href = a.get('href', '')
                if not href or any(x in href.lower() for x in ['profile', 'action=', 'last', 'new', 'gopost', 'msg', 'topicseen']):
                    continue
                
                text = a.get_text(strip=True)
                if text.lower() in ('last', 'first', 'next', 'prev', '«', '»', '', 'new') or text.isdigit():
                    continue
                
                if subject is None or len(text) > len(subject):
                    subject = text
                    topic_url = href if href.startswith('http') else f"{base_url.rstrip('/')}/{href.lstrip('/')}"
            
            if not subject: 
                continue
            
            # Universal fall-back date parsing logic
            last_post_date = None
            for text_node in container.strings:
                text = text_node.strip()
                if len(text) >= 6 and re.search(r'\d', text):
                    # Uses your existing global parse_date_string implementation
                    last_post_date = parse_date_string(text)
                    if last_post_date: 
                        break
                    
            # Protect against duplicates from the same page block processing cycle
            if topic_url not in [t['url'] for t in topics]:
                topics.append({
                    'url': topic_url,
                    'subject': subject,
                    'most_recent_date': last_post_date,
                    'started_date': None
                })
                
        print(f"  Successfully extracted {len(topics)} topics")
    except Exception as e:
        print(f"Extraction Error: {e}")
        
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

        await page.goto(page_url, timeout=60000)
        await page.wait_for_load_state('networkidle', timeout=60000)
        # await page.goto(page_url, timeout=30000)
        # await page.wait_for_load_state('networkidle', timeout=30000)

        return True
    except Exception as e:
        print(f"  Error navigating to page {page_num}: {e}")
        return False

async def process_board(page, board, latest_date_obj, new_topics, debug=False):
    try:
        print(f"Processing board: {board['name']}")
        print(f"[DEBUG] Board URL: {board['url']}")

        # Simulate human behavior
        await page.mouse.move(200, 300)
        await page.mouse.move(500, 200)
        await asyncio.sleep(random.uniform(2.0, 4.0))

        # First visit to navigate past initial challenges
        await page.goto(board['url'], timeout=60000, referer=BASE_URL)
        await asyncio.sleep(8.0)  # Full window for post-verification redirect
        await page.wait_for_load_state('networkidle')

        # Re-verify page content delivery
        await page.goto(board['url'], timeout=60000, referer=BASE_URL)
        await asyncio.sleep(random.uniform(3.5, 6.0))
        await page.wait_for_load_state('domcontentloaded')

        # FIX: Instead of waiting for specific volatile CSS classes like 'a.subject',
        # we dynamically wait for any standard content layout marker or anchor tags to appear.
        try:
            await page.wait_for_selector('a', timeout=15000)
        except Exception:
            print("  [WARNING] Universal link elements took too long to load. Proceeding...")
            await asyncio.sleep(5)

        html = await page.content()
        if debug:
            with open('debug_board.html', 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"[DEBUG] Board HTML saved ({len(html):,} chars)")

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
            
            # --- ROBUST INLINE FALLBACK TOPIC EXTRACTION ---
            # If your underlying extract_topic_info() relies on 'a.subject', it will return 0.
            # We patch it directly by running a regex-based URL lookup over the BeautifulSoup DOM.
            page_topics = await extract_topic_info(html, BASE_URL, debug=debug)

            if not page_topics:
                if debug:
                    print("  [DEBUG] extract_topic_info returned 0 topics. Applying URL-regex extraction fallback...")
                
                soup = BeautifulSoup(html, 'html.parser')
                # Match links matching the standard SMF topic pattern: topic,XXXX.X.html
                topic_elements = soup.find_all('a', href=re.compile(r'topic,\d+\.\d+\.html'))
                
                extracted_fallback = []
                for link in topic_elements:
                    href = link.get('href', '')
                    
                    # Avoid noise elements like "Go to latest message" shortcuts
                    if ';topicseen' in href or '#msg' in href or 'msg' in href:
                        continue
                    
                    topic_url = href if href.startswith('http') else f"https://surfacehippy.info{href}"
                    topic_title = link.get_text(strip=True)
                    
                    if topic_title and topic_url not in [t['url'] for t in extracted_fallback]:
                        # Mock object format returned by extract_topic_info
                        # If your RAG requires actual dates, we default to None so it pulls it as a new post
                        extracted_fallback.append({
                            'url': topic_url,
                            'subject': topic_title,
                            'most_recent_date': None  
                        })
                page_topics = extracted_fallback

            print(f"  Extracted {len(page_topics)} total topics from page content")

             # --- UPDATED FAIL-SAFE FILTERING LOOP IN process_board ---
            page_has_new_topics = False
            for topic in page_topics:
                # 1. FORCE INGESTION: If there's no date, or if the extracted date string is unparseable
                if not topic.get('most_recent_date'):
                    new_topics.append(topic)
                    page_has_new_topics = True
                    print(f"    New (Forced): '{topic['subject'][:60]}' (Missing date context)")
                    continue

                try:
                    # 2. STANDARD CHECKPOINT COMPARISON
                    topic_most_recent_date = parser.parse(topic['most_recent_date'])
                    
                    # Normalize both to timezone-naive datetimes or dates for safe evaluation
                    if isinstance(latest_date_obj, datetime):
                        target_compare = latest_date_obj
                    else:
                        target_compare = datetime.combine(latest_date_obj, datetime.min.time())
                        
                    if topic_most_recent_date.tzinfo is not None:
                        topic_most_recent_date = topic_most_recent_date.replace(tzinfo=None)
                    if target_compare.tzinfo is not None:
                        target_compare = target_compare.replace(tzinfo=None)

                    if topic_most_recent_date >= target_compare:
                        new_topics.append(topic)
                        page_has_new_topics = True
                        print(f"    New: '{topic['subject'][:60]}' ({topic['most_recent_date']})")
                
                except Exception as e:
                    # 3. SAFETY NET: If date parsing blows up, do NOT skip the post.
                    # Send it to Pinecone anyway, where your vector hashes will safely deduplicate it.
                    new_topics.append(topic)
                    page_has_new_topics = True
                    print(f"    New (Safety Fallback): '{topic['subject'][:60]}' (Parsing error: {e})")
            # CHANGE HERE: Since we want to ensure we inspect everything on Page 1 
            # regardless of whether the first few topics look old, we do NOT break on Page 1.
            if not page_has_new_topics and latest_date_obj and page_num > 1:
                print(f"  No new topics on page {page_num}, stopping board scan")
                break
           
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Error processing board {board['name']}: {e}")
        import traceback
        traceback.print_exc()

# async def process_board(page, board, latest_date_obj, new_topics, debug=False):
#     try:
#         print(f"Processing board: {board['name']}")
#         print(f"[DEBUG] Board URL: {board['url']}")

#         # Simulate human behavior
#         await page.mouse.move(200, 300)
#         await page.mouse.move(500, 200)
#         await asyncio.sleep(3)

#         await page.mouse.move(800, 400)
#         await page.mouse.move(600, 300)
#         await asyncio.sleep(5)
#         # In process_board, update the page.goto call:
                
#         await page.goto(board['url'], timeout=60000, referer=BASE_URL)
        
#         # Give Cloudflare a full window to complete the post-verification redirect
#         await asyncio.sleep(8.0)
#         await page.wait_for_load_state('networkidle')

#         await page.goto(board['url'], timeout=60000, referer=BASE_URL)
#         await asyncio.sleep(random.uniform(3.5, 6.0))
#         await page.wait_for_load_state('domcontentloaded')

#         try:
#             await page.wait_for_selector('a.subject', timeout=15000)
#         except:
#             await asyncio.sleep(5)
#         html = await page.content()
#         if debug:
#             with open('debug_board.html', 'w', encoding='utf-8') as f:
#                 f.write(html)
#             print(f"[DEBUG] Board HTML saved ({len(html):,} chars)")

#         _, max_pages = await get_pagination_info(page)
#         print(f"  Found {max_pages} pages")

#         for page_num in range(1, max_pages + 1):
#             print(f"  Processing page {page_num}/{max_pages}")

#             if page_num > 1:
#                 success = await navigate_to_specific_page(page, board['url'], page_num)
#                 if not success:
#                     print(f"  Failed to navigate to page {page_num}")
#                     break

#             html = await page.content()
#             page_topics = await extract_topic_info(html, BASE_URL, debug=debug)

#             page_has_new_topics = False
#             for topic in page_topics:
#                 if not topic['most_recent_date']:
#                     new_topics.append(topic)
#                     page_has_new_topics = True
#                     print(f"    New: '{topic['subject'][:60]}' (no date)")
#                     continue

#                 topic_most_recent_date = parser.parse(topic['most_recent_date'])

#                 if not latest_date_obj or topic_most_recent_date > latest_date_obj:
#                     new_topics.append(topic)
#                     page_has_new_topics = True
#                     print(f"    New: '{topic['subject'][:60]}' ({topic['most_recent_date']})")

#             if not page_has_new_topics and latest_date_obj:
#                 print(f"  No new topics on page {page_num}, stopping board scan")
#                 break

#             await asyncio.sleep(1)

#     except Exception as e:
#         print(f"Error processing board {board['name']}: {e}")
#         import traceback
#         traceback.print_exc()


async def crawl_new_content(latest_date_in_pinecone, debug=False):
    """Crawl forum boards and topics newer than the latest date in Pinecone."""
    latest_date_obj = parser.parse(latest_date_in_pinecone) if latest_date_in_pinecone else None
    new_topics = []

    async with async_playwright() as p:
        browser = await p.firefox.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox"
            ]
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Dest": "document"
            }
        )
        page = await context.new_page()
        # Override browser test flags natively inside the DOM without crashing JS
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        """)

        
              
       

        # Simulate human behavior before navigating
        await page.mouse.move(100, 200)
        await page.mouse.move(300, 400)
        await asyncio.sleep(2)

        print(f"Fetching main forum page: {BASE_URL}")
        await page.goto(BASE_URL, timeout=60000)
        
        # Save debug HTML before waiting for networkidle
        html = await page.content()
        if debug:
            debug_file = 'debug_forum_index.html'
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"\n[DEBUG] Raw HTML saved to: {debug_file}  ({len(html):,} chars)")
            _soup = BeautifulSoup(html, 'html.parser')
            a_classes = sorted(set(
                ' '.join(a['class']) for a in _soup.find_all('a') if a.get('class')
            ))
            print(f"[DEBUG] Unique <a> class values ({len(a_classes)} total):")
            for c in a_classes[:40]:
                print(f"         {c}")
            lp_tags = [tag for tag in _soup.find_all(True)
                       if 'last post' in tag.get_text().lower() and tag.name in ('p','div','td','li','span')]
            print(f"\n[DEBUG] Tags containing 'Last post' text ({len(lp_tags)} found):")
            for tag in lp_tags[:10]:
                snippet = ' '.join(tag.get_text().split())[:120]
                print(f"         <{tag.name} class={tag.get('class')}> : {snippet}")
            print()

        await page.wait_for_load_state('networkidle', timeout=60000)
        html = await page.content()

        recent_boards = await extract_board_info(html, BASE_URL, latest_date_obj)

        print(f"Processing {len(recent_boards)} boards with recent activity\n")

        for board in recent_boards:
            await process_board(page, board, latest_date_obj, new_topics, debug=debug)
            print()

        await browser.close()

    return new_topics


async def main():
    # Support --debug flag to dump raw HTML for selector diagnosis
    debug = '--debug' in sys.argv

    # First, try to load from state file
    latest_date = load_latest_date()

    # If no state file exists, scan Pinecone (this is slow)
    if not latest_date:
        print("No state file found. Scanning Pinecone for latest date...")
        latest_date = await get_latest_date_from_pinecone()

        # Save it for next time
        if latest_date:
            save_latest_date(latest_date)

    if latest_date:
        print(f"Running incremental update from {latest_date}")

    start_time = time.time()
    new_topics = await crawl_new_content(latest_date, debug=debug)
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
            extract_and_save_latest_date(new_topics)
        except ImportError:
            print(f"\nRun ingestion manually:")
            print(f"python updated_pinecone_ingestion.py --input {output_file}")
    else:
        print("No new topics found. Database is up to date.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n✅ Pinecone ingestion completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pinecone ingestion FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
