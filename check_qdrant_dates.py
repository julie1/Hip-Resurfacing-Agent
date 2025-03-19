import os
import sys
import json
from typing import Dict, Any, Set
from datetime import datetime
from dateutil import parser
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

def analyze_date_metadata():
    # Load environment variables from your .env file
    load_dotenv()

    # Connect to Qdrant using the same credentials as your ingestion script
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    collection_name = "site_pages"

    # Get collection info
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection: {collection_name}")
        print(f"Total points: {collection_info.points_count}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return

    # Get date distribution
    date_stats = {
        "urls": set(),
        "earliest_date": None,
        "latest_date": None,
        "date_counts": {},
        "invalid_dates": 0
    }

    # Process in batches to avoid memory issues
    batch_size = 1000
    offset = None
    total_processed = 0

    print("Analyzing date metadata in batches...")
    while True:
        try:
            results = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=["url", "started_date", "most_recent_date"],
                with_vectors=False,
            )

            points = results[0]
            if not points:
                break

            offset = results[1]  # Get offset for next batch
            total_processed += len(points)

            for point in points:
                url = point.payload.get("url")
                date_stats["urls"].add(url)

                try:
                    recent_date_str = point.payload.get("most_recent_date")
                    if recent_date_str:
                        recent_date = parser.parse(recent_date_str)

                        # Track earliest and latest dates
                        if date_stats["earliest_date"] is None or recent_date < date_stats["earliest_date"]:
                            date_stats["earliest_date"] = recent_date

                        if date_stats["latest_date"] is None or recent_date > date_stats["latest_date"]:
                            date_stats["latest_date"] = recent_date
                            date_stats["latest_url"] = url

                        # Count by month
                        month_key = recent_date.strftime("%Y-%m")
                        date_stats["date_counts"][month_key] = date_stats["date_counts"].get(month_key, 0) + 1
                except Exception as e:
                    date_stats["invalid_dates"] += 1

            print(f"Processed {total_processed}/{collection_info.points_count} records...")

            if not offset:
                break

        except Exception as e:
            print(f"Error processing batch: {e}")
            break

    # Print results
    print("\n=== Date Analysis Results ===")
    print(f"Total unique URLs: {len(date_stats['urls'])}")
    print(f"Earliest date found: {date_stats['earliest_date']}")
    print(f"Latest date found: {date_stats['latest_date']}")
    print(f"URL with latest date: {date_stats.get('latest_url', 'N/A')}")
    print(f"Invalid dates encountered: {date_stats['invalid_dates']}")

    # Distribution by month
    print("\nDistribution by month:")
    sorted_months = sorted(date_stats["date_counts"].keys())
    for month in sorted_months:
        print(f"{month}: {date_stats['date_counts'][month]} records")

    # Check for recent gaps
    if date_stats["latest_date"]:
        now = datetime.now()
        days_since_latest = (now - date_stats["latest_date"]).days
        print(f"\nDays since most recent record: {days_since_latest}")
        if days_since_latest > 30:
            print(f"WARNING: No records found for the past {days_since_latest} days")

    # Find records with recent dates
    print("\nSample of most recent records:")
    if date_stats["latest_date"]:
        try:
            latest_month = date_stats["latest_date"].strftime("%Y-%m")
            recent_results = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="most_recent_date",
                            match=models.MatchText(text=latest_month)
                        )
                    ]
                ),
                limit=5,
                with_payload=["url", "most_recent_date", "title"],
            )

            for point in recent_results[0]:
                print(f"URL: {point.payload.get('url')}")
                print(f"Date: {point.payload.get('most_recent_date')}")
                print(f"Title: {point.payload.get('title')}")
                print("---")
        except Exception as e:
            print(f"Error retrieving recent records: {e}")

def check_specific_url(url):
    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    collection_name = "site_pages"

    print(f"\n=== Checking specific URL: {url} ===")

    try:
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url",
                        match=models.MatchValue(value=url)
                    )
                ]
            ),
            limit=100,  # Get all chunks for this URL
            with_payload=["chunk_number", "started_date", "most_recent_date", "title"],
            with_vectors=False,
        )

        points = results[0]
        print(f"Found {len(points)} chunks for this URL")

        if points:
            print("\nMetadata consistency check:")
            dates = {}
            for point in points:
                start_date = point.payload.get("started_date")
                recent_date = point.payload.get("most_recent_date")
                chunk = point.payload.get("chunk_number")
                dates[chunk] = (start_date, recent_date)

            # Check if all chunks have the same dates
            first_start, first_recent = next(iter(dates.values()))
            consistent = all(start == first_start and recent == first_recent for start, recent in dates.values())
            print(f"Date consistency across chunks: {'Consistent' if consistent else 'INCONSISTENT'}")

            if not consistent:
                print("\nInconsistencies found:")
                for chunk, (start, recent) in dates.items():
                    if start != first_start or recent != first_recent:
                        print(f"Chunk {chunk}: start={start}, recent={recent}")
        else:
            print("No records found for this URL")

    except Exception as e:
        print(f"Error checking URL: {e}")

def compare_with_source(source_file='topic_data.json'):
    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    collection_name = "site_pages"

    print(f"\n=== Comparing with source data: {source_file} ===")

    try:
        with open(source_file, 'r') as f:
            source_data = json.load(f)

        # Create lookup dict
        source_dates = {item['url']: item['most_recent_date'] for item in source_data}

        # Check dates for each URL
        mismatches = 0
        checked = 0

        for url, source_date in source_dates.items():
            checked += 1
            if checked % 100 == 0:
                print(f"Checked {checked}/{len(source_dates)} URLs...")

            results = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url)
                        )
                    ]
                ),
                limit=1,  # Just need one chunk to check dates
                with_payload=["most_recent_date"],
                with_vectors=False,
            )

            points = results[0]
            if not points:
                print(f"URL in source but not in database: {url}")
                continue

            db_date = points[0].payload.get('most_recent_date')

            if db_date != source_date:
                mismatches += 1
                print(f"Mismatch for {url}:")
                print(f"  In Qdrant: {db_date}")
                print(f"  In source: {source_date}")

                if mismatches >= 10:
                    print("Showing only first 10 mismatches...")
                    break

        print(f"\nComparison complete. {mismatches} mismatches found out of {checked} URLs checked.")

    except FileNotFoundError:
        print(f"Source file not found: {source_file}")
    except Exception as e:
        print(f"Error comparing with source: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "check_url" and len(sys.argv) > 2:
            check_specific_url(sys.argv[2])
        elif command == "compare" and len(sys.argv) > 2:
            compare_with_source(sys.argv[2])
        else:
            print("Unknown command or missing arguments")
            print("Usage:")
            print("  python check_qdrant_dates.py analyze")
            print("  python check_qdrant_dates.py check_url <url>")
            print("  python check_qdrant_dates.py compare <source_file>")
    else:
        analyze_date_metadata()  # Default action
