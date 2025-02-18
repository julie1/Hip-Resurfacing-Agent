# Hip-Resurfacing-Agent

This repository is a rag (retrieval augmented generation) streamlit application for posts of the group Hipresurfacingsite@groups.io.  To use the app please go to https://hipresurfacingagent.streamlit.app/ and ask a question.

## Overview

Since 2010 prospective hip surgery patients and former patients have posted questions and 
related experiences on a wide range of topics related to hip dysfunction and potential
surgeries mostly hip resurfacing to improve mobility and relieve pain. This app uses a
Large Language Model (LLM) approach to facilitate the search for answers from the wealth 
of knowledge in the posts of hip resurfacing group members.

## Data

The dataset consists of posts that are chunked and embedded into the Supabase database.
The information stored includes the post urls, the chunk number, the post title, a post summary, the chunk content, the metadata, and the embedding.  The metadata is the following:
metadata={
                        "source": "hip_messages",
                        "chunk_size": len(chunk),
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "url_path": urlparse(url).path,
                        "started_date": topic_data['started_date'],
                        "most_recent_date": topic_data['most_recent_date'],
                        "total_chunks": len(chunks)
                    }.
                    
## LLM   

The app is currently powered by openai's gpt-4o-mini which gave the best results within the current budget. 

## Acknowledgments

This app is a continuation of my Find-a-Doctor project ( https://github.com/julie1/Find-a-Doctor) that was part of Alexey Grigorev's LLMops zoomcamp.  Recently, I was inspired by Cole Medin's instructive youtube videos: https://www.youtube.com/@ColeMedin               
