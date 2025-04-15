# Hip-Resurfacing-Agent

This repository is a rag (retrieval augmented generation) streamlit application for posts of the groups https://surfacehippy.info/hiptalk/ and https://groups.io/g/Hipresurfacingsite.  To use the app please go to https://hipresurfacingagent.streamlit.app/ and ask a question. Note that the code here can easily be adapted to create a rag search engine for any on-line forum.

## Overview

Since 2005 prospective hip surgery patients and former patients have posted questions and 
related experiences on a wide range of topics related to hip dysfunction and potential
surgeries mostly hip resurfacing to improve mobility and relieve pain. This app uses a
Large Language Model (LLM) approach to facilitate the search for answers from the wealth 
of knowledge in the posts of surface hippy and hip resurfacing group members.

## Data

The dataset consists of posts that are chunked and embedded into the Pincone and Qdrant databases.
The information stored includes the following for each chunk in a message:

- **message url**
- **chunk number**
- **message title** 
- **message summary** 
- **chunk content** 
- **metadata** 
- **embedding** 

The metadata consists of the following:

- **source:** hip_messages
- **chunk_size:** len(chunk)
- **crawled_at:** datetime.now(timezone.utc).isoformat()
- **url_path:** urlparse(url).path
- **started_date:** topic_data['started_date']
- **most_recent_date:** topic_data['most_recent_date']
- **total_chunks:** len(chunks)

The Pinecone record count currently contains 34,729 records.
The Qdrant database table currently contains 23,622 points_count.

## Technologies

- Python 3.12
- Qdrant is a vector database
- OpenAI as an LLM
- conda
- Streamlit for browser interactions with the applicatio
- Supabase, a postgres database, was previously used
                
## LLM   

The app is currently powered by openai's gpt-4o-mini which gave the best results within the current budget. 

## Code

The code associated with the Supabase database is in the Supabase folder.

- [`forum_crawler.py`] - obtains a json file containing url's, titles, and dates of messages from https://surfacehippy.info/hiptalk/ 
- [`paginationcrawlerv2.py`] - obtains a json file containing url's, titles, and dates of messages from https://groups.io/g/Hipresurfacingsite
- [`pinecone_ingestion.py`] - obtains summary of message and chunks content, inserts it into Pinecone with a vector embedding
- [`improved_incremental_pinecone_update.py`] - implements the crawler and ingestion pipeline for just the latest messages from https://surfacehippy.info/hiptalk/ 
- [`qdrant_ingestion.py`] - obtains summary of message and chunks content, inserts it into Qdrant with a vector embedding
- [`check_qdrant_dates.py`] - obtains the latest started_date in Qdrant which is used in the following incremental update
- [`incremental_update_qdrant.py`] - implements the crawler and ingestion pipeline for just the latest messages from https://groups.io/g/Hipresurfacingsite
- [`combine_agents.py`] - uses pydantic-ai agent methods and the LLM to retrieve data to answer user messages
- [`streamlit_ui.py`] - creates the Streamlit user interface for this app 
- [`Supabase/corrected_ingestion.py`] - obtains summary of message and chunks content, inserts it into Supabase with a vector embedding
- [`Supabase/updated_schema.sql`] - creates the Supabase table containing the message data
- [`Supabase/hip_agent.py`] - uses pydantic-ai agent methods and the LLM to retrieve data to answer user messages
- [`Supabase/streamlit__ui.py`] - creates the Streamlit user interface for this app

###  How to use

Please see https://github.com/julie1/Find-a-Doctor for more detailed information.

1. Create a Python environment
2. clone the repository
3. adjust the code to your particular use case
4. obtain the openai key, the qdrant url and service key, then fill out the .env file with these and put this into your local
environment
5. ```bash
   cd Hip-Resurfacing-Agent
   pip install -r requirements.txt
6. run the crawler, then ingestion, then streamlit scripts
   

## Acknowledgments

This app is a continuation of my Find-a-Doctor project ( https://github.com/julie1/Find-a-Doctor) that was part of Alexey Grigorev's LLMops zoomcamp.  Recently, I was inspired by Cole Medin's instructive youtube videos: https://www.youtube.com/@ColeMedin               
