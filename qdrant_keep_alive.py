"""
Qdrant Keep-Alive Script
Runs a simple query to prevent Qdrant free tier from pausing the database
"""
import os
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from openai import OpenAI

print("=" * 70)
print("QDRANT KEEP-ALIVE")
print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
print("=" * 70)

try:
    # Initialize clients
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    collection_name = "hip_messages"  # Adjust if your collection has a different name
    
    # Check collection status
    print("\n✓ Connected to Qdrant")
    collection_info = qdrant_client.get_collection(collection_name)
    print(f"✓ Collection: {collection_name}")
    print(f"✓ Points count: {collection_info.points_count}")
    print(f"✓ Status: {collection_info.status}")
    
    # Generate embedding for a simple query
    query_text = "hip resurfacing recovery"
    print(f"\n→ Generating embedding for: '{query_text}'")
    
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )
    query_embedding = response.data[0].embedding
    print("✓ Embedding generated")
    
    # Perform keep-alive search using query_points (new API)
    print("→ Performing keep-alive query...")
    from qdrant_client.models import QueryRequest, VectorInput
    
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=1
    ).points
    
    if search_result:
        print(f"✓ Query successful - found {len(search_result)} result(s)")
        print(f"✓ Top result score: {search_result[0].score:.4f}")
    else:
        print("⚠ Query returned no results (database may be empty)")
    
    print("\n" + "=" * 70)
    print("KEEP-ALIVE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    print("=" * 70)
    print("KEEP-ALIVE FAILED")
    print("=" * 70)
    raise
