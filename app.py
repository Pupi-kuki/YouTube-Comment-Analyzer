# pyright: reportPrivateImportUsage=false

import streamlit as st
import pandas as pd #data manipulation and handling of tabula
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import random
from datetime import datetime
import io                                                                       # without download
import base64 #extracting features like phone numbers, emails,
from transformers import pipeline
import warnings
from typing import List, Dict, Any, Optional
warnings.filterwarnings('ignore')
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="YouTube Comment Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff3333;
        color: white;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .sentiment-positive {
        color: #00ff88;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffaa00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model with caching for performance."""
    try:
        
        return pipeline("sentiment-analysis")  
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_service(api_key: str):
    """Create YouTube API service."""
    try:
        return build('youtube', 'v3', developerKey=api_key)
    except Exception as e:
        st.error(f"Error creating YouTube service: {e}")
        return None

def search_videos(query: str, api_key: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search for videos using YouTube API."""
    youtube = get_youtube_service(api_key)
    if not youtube:
        return []
    
    try:
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results,
            type='video'
        ).execute()
        
        videos = []
        for item in search_response.get('items', []):
            video_data = {
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                'channel': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt']
            }
            videos.append(video_data)
        
        return videos
    except HttpError as e:
        if e.resp.status == 403:
            st.error("YouTube API quota exceeded. Please try again later or check your API key.")
        else:
            st.error(f"YouTube API error: {e}")
        return []
    except Exception as e:
        st.error(f"Error searching videos: {e}")
        return []

def get_video_comments(video_id: str, api_key: str, max_results: int = 100) -> List[Dict[str, Any]]:
    """Fetch comments for a specific video."""
    youtube = get_youtube_service(api_key)
    if not youtube:
        return []
    
    try:
        comments_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            order='relevance'
        ).execute()
        
        comments = []
        for item in comments_response.get('items', []):
            comment_data = {
                'comment': item['snippet']['topLevelComment']['snippet']['textDisplay'],
                'author': item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                'likes': item['snippet']['topLevelComment']['snippet']['likeCount'],
                'published_at': item['snippet']['topLevelComment']['snippet']['publishedAt']
            }
            comments.append(comment_data)
        
        return comments
    except HttpError as e:
        if e.resp.status == 403:
            st.error("YouTube API quota exceeded. Please try again later or check your API key.")
        elif e.resp.status == 404:
            st.error("Video not found or comments disabled.")
        else:
            st.error(f"YouTube API error: {e}")
        return []
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []

def analyze_sentiment(comments: List[Dict[str, Any]], sentiment_analyzer) -> List[Dict[str, Any]]:
    """Analyze sentiment for a list of comments."""
    if not sentiment_analyzer:
        return []
    
    results = []
    for comment in comments:
        try:
           
            clean_text = re.sub(r'<[^>]+>', '', comment['comment'])
            clean_text = re.sub(r'http\S+|www\S+|https\S+', '', clean_text, flags=re.MULTILINE)
            clean_text = clean_text.strip()
            
            if len(clean_text) < 3:  
                continue
            
            # Analyzesentiment
            analysis = sentiment_analyzer(clean_text[:512])[0]  # Limit text length
            
            # Classify sentiment
            label = analysis['label'].upper()  
            score = analysis['score']
            
            # Map different labels
            if label in ['POSITIVE', 'POS'] and score > 0.6:
                sentiment = 'Positive'
            elif label in ['NEGATIVE', 'NEG'] and score > 0.6:
                sentiment = 'Negative'
            elif label in ['NEUTRAL', 'NEU']:
                sentiment = 'Neutral'
            else:
                #Default classification based on score
                if score > 0.6:
                    sentiment = 'Positive' if label in ['POSITIVE', 'POS'] else 'Negative'
                else:
                    sentiment = 'Neutral'
            
            results.append({
                'comment': clean_text,
                'author': comment['author'],
                'likes': comment['likes'],
                'sentiment': sentiment,
                'confidence': score,
                'original_label': label
            })
            
        except Exception as e:
            st.warning(f"Error analyzing comment: {e}")
            continue
    
    return results

def create_sentiment_chart(sentiment_data: List[Dict[str, Any]]):
    """Create a bar chart for sentiment distribution."""
    if not sentiment_data:
        return None
    
    df = pd.DataFrame(sentiment_data)
    sentiment_counts = df['sentiment'].value_counts()
    
    
    colors = {'Positive': '#00ff88', 'Negative': '#ff4444', 'Neutral': '#ffaa00'}
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        color=sentiment_counts.index,
        color_discrete_map=colors,
        title="Sentiment Distribution",
        labels={'x': 'Sentiment', 'y': 'Count'},
        template='plotly_dark'
    )
    
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_video_info(video_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Get video information."""
    youtube = get_youtube_service(api_key)
    if not youtube:
        return None
    
    try:
        video_response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
        
        if video_response['items']:
            video = video_response['items'][0]
            return {
                'title': video['snippet']['title'],
                'channel': video['snippet']['channelTitle'],
                'view_count': video['statistics'].get('viewCount', 0),
                'like_count': video['statistics'].get('likeCount', 0),
                'comment_count': video['statistics'].get('commentCount', 0)
            }
    except Exception as e:
        st.warning(f"Could not fetch video info: {e}")
    
    return None

def download_csv(data: List[Dict[str, Any]], filename: str) -> str:
    """Create a downloadable CSV file."""
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def get_trending_videos(api_key: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Fetch trending videos using YouTube API."""
    youtube = get_youtube_service(api_key)
    if not youtube:
        return []
    
    try:
        #Trending Videos
        trending_response = youtube.videos().list(
            part='id,snippet,statistics',
            chart='mostPopular',
            maxResults=max_results,
            regionCode='US'
        ).execute()
        
        videos = []
        for item in trending_response.get('items', []):
            video_data = {
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                'channel': item['snippet']['channelTitle'],
                'view_count': item['statistics'].get('viewCount', 0),
                'like_count': item['statistics'].get('likeCount', 0),
                'published_at': item['snippet']['publishedAt']
            }
            videos.append(video_data)
        
        return videos
    except HttpError as e:
        if e.resp.status == 403:
            st.error("YouTube API quota exceeded. Please try again later or check your API key.")
        else:
            st.error(f"YouTube API error: {e}")
        return []
    except Exception as e:
        st.error(f"Error fetching trending videos: {e}")
        return []
    
def plot_wordcloud(comments):
    all_text = ' '.join(comment['comment'] for comment in comments if 'comment' in comment and isinstance(comment['comment'], str))
    if not all_text.strip():
        st.info("No text available for word cloud.")
        return
    wc = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(all_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.close()

def perform_analysis(video_id: str, video_url: str, api_key: str):
    """Perform complete sentiment analysis workflow for a video."""
    # Get video info
    video_info = get_video_info(video_id, api_key)
    
    # Fetch comments
    with st.spinner("Fetching comments..."):
        comments = get_video_comments(video_id, api_key)
    
    if not comments:
        st.error("No comments found for this video or comments are disabled.")
        return False
    
    # Load sentiment model
    with st.spinner("Loading sentiment analysis model..."):
        sentiment_analyzer = load_sentiment_model()
    
    if not sentiment_analyzer:
        st.error("Failed to load sentiment analysis model.")
        return False
    
    # Analyze sentiment
    with st.spinner("Analyzing comment sentiment..."):
        sentiment_results = analyze_sentiment(comments, sentiment_analyzer)
    if not sentiment_results:
        st.error("No comments could be analyzed.")
        return False
    
    # Display results
    st.markdown("## 📊 Analysis Results")
    # Show video info
    if video_info:
        st.markdown(f"""
        ### 🎥 {video_info['title']}
        **Channel:** {video_info['channel']} | 
        **Views:** {int(video_info['view_count']):,} | 
        **Likes:** {int(video_info['like_count']):,} | 
        **Comments:** {int(video_info['comment_count']):,}
        """)
    
    # Create and display chart
    chart = create_sentiment_chart(sentiment_results)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    
    # Create word cloud
    if sentiment_results:
        st.markdown("## ☁️ Word Cloud of Comments")
        plot_wordcloud(sentiment_results)
    
    # Display metrics
    df = pd.DataFrame(sentiment_results)
    sentiment_counts = df['sentiment'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comments", len(sentiment_results))
    with col2:
        st.metric("Positive", sentiment_counts.get('Positive', 0))
    with col3:
        st.metric("Negative", sentiment_counts.get('Negative', 0))
    with col4:
        st.metric("Neutral", sentiment_counts.get('Neutral', 0))
    
    # Display top comments
    st.markdown("## 🏆 Top Comments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 😊 Most Positive")
        positive_comments = df[df['sentiment'] == 'Positive'].sort_values(by='confidence', ascending=False)  # type: ignore
        if not positive_comments.empty:
            top_positive = positive_comments.iloc[0]
            st.markdown(f"""
            **Comment:** {top_positive['comment'][:100]}...
            **Confidence:** {top_positive['confidence']:.2f}
            **Author:** {top_positive['author']}
            """)
    
    with col2:
        st.markdown("### 😞 Most Negative")
        negative_comments = df[df['sentiment'] == 'Negative'].sort_values(by='confidence', ascending=False)  # type: ignore
        if not negative_comments.empty:
            top_negative = negative_comments.iloc[0]
            st.markdown(f"""
            **Comment:** {top_negative['comment'][:100]}...
            **Confidence:** {top_negative['confidence']:.2f}
            **Author:** {top_negative['author']}
            """)
    
    # Display sample comments
    st.markdown("## 💬 Sample Comments")
    sample_comments = df.sample(min(10, len(df))).sort_values('sentiment')
    
    for _, comment in sample_comments.iterrows():
        sentiment_value = str(comment['sentiment'])
        sentiment_color = {
            'Positive': 'sentiment-positive',
            'Negative': 'sentiment-negative',
            'Neutral': 'sentiment-neutral'
        }.get(sentiment_value, 'sentiment-neutral')
        
        st.markdown(f"""
        <div class="metric-card">
            <span class="{sentiment_color}">[{sentiment_value}]</span> 
            <strong>Confidence: {comment['confidence']:.2f}</strong><br>
            <em>{comment['comment']}</em><br>
            <small>— {comment['author']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Export data
    st.markdown("## 📥 Export Data")
    csv_data = [{
        'Comment': row['comment'],
        'Author': row['author'],
        'Sentiment': row['sentiment'],
        'Confidence': row['confidence'],
        'Likes': row['likes']
    } for _, row in df.iterrows()]
    
    st.markdown(
        download_csv(csv_data, f"youtube_sentiment_analysis_{video_id}.csv"),
        unsafe_allow_html=True
    )
    
    # Add to history
    history_entry = {
        'title': video_info['title'] if video_info else f"Video {video_id}",
        'url': video_url,
        'comment_count': len(sentiment_results),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sentiment_counts': {
            'Positive': sentiment_counts.get('Positive', 0),
            'Negative': sentiment_counts.get('Negative', 0),
            'Neutral': sentiment_counts.get('Neutral', 0)
        },
        'video_id': video_id
    }
    
    st.session_state.analysis_history.append(history_entry)
    return True

def main():
    st.markdown("""
    # 🎬 YouTube Comment Sentiment Analyzer
    ### Analyze the sentiment of YouTube video comments using AI
    ---
    """)
    
    with st.sidebar:
        st.markdown("## 🔑 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "YouTube API Key",
            type="password",
            help="Get your API key from Google Cloud Console"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your YouTube API key to continue")
            st.markdown("""
            **How to get an API key:**
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select existing
            3. Enable YouTube Data API v3
            4. Create credentials (API Key)
            5. Copy and paste the key above
            """)
            return
        
        st.markdown("---")
        st.markdown("## 🔍 Search Videos")
        
        # Search functionality
        search_query = st.text_input(
            "Search for videos:",
            key="search_input",
            placeholder="Enter search terms...",
            help="Type keywords to search for YouTube videos"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            search_button = st.button("🔍 Search", use_container_width=True)
        with col2:
            trending_button = st.button("🔥 Trending", use_container_width=True)
        
        # Search when button is clicked
        if search_button and search_query and search_query.strip():
            with st.spinner("Searching for videos..."):
                videos = search_videos(search_query, api_key)
                st.session_state.search_results = videos
                st.session_state.last_search_query = search_query
                
                if not videos:
                    st.info("No videos found for your search query.")
        
        # Get trending videos when button is clicked
        if trending_button:
            with st.spinner("Fetching trending videos..."):
                trending_videos = get_trending_videos(api_key)
                st.session_state.search_results = trending_videos
                st.session_state.last_search_query = "Trending Videos"
                
                if not trending_videos:
                    st.info("Could not fetch trending videos. Please try again later.")
        
        # Display search results from session state
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            if hasattr(st.session_state, 'last_search_query') and st.session_state.last_search_query == "Trending Videos":
                st.markdown("### 🔥 Trending Videos")
            else:
                st.markdown("### 📺 Search Results")
                
            for i, video in enumerate(st.session_state.search_results):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(video['thumbnail'], width=80)
                    
                
                with col2:
                    st.markdown(f"**{video['title'][:50]}...**")
                    st.markdown(f"*{video['channel']}*")
                    
                    if 'view_count' in video:
                        st.markdown(f"👁️ {int(video['view_count']):,} views")
                    
                    if st.button(f"Analyze", key=f"analyze_{i}"):
                        st.session_state.selected_video_id = video['video_id']
                        st.session_state.selected_video_title = video['title']
                        st.session_state.auto_analyze = True
                        st.rerun()
            
            
            if st.button("Clear Results"):
                if hasattr(st.session_state, 'search_results'):
                    del st.session_state.search_results
                if hasattr(st.session_state, 'last_search_query'):
                    del st.session_state.last_search_query
                st.rerun()
    
 
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📝 Input")
        
       
        default_url = ""
        if hasattr(st.session_state, 'selected_video_id') and st.session_state.selected_video_id:
            default_url = f"https://www.youtube.com/watch?v={st.session_state.selected_video_id}"
        
        video_url = st.text_input(
            "YouTube Video URL",
            value=default_url,
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube video URL to analyze its comments"
        )
        
       
        if hasattr(st.session_state, 'selected_video_id') and st.session_state.selected_video_id:
            st.success(f"Selected: {st.session_state.selected_video_title}")
    
    with col2:
        st.markdown("## 📊 Last Video Stats")
        if 'analysis_history' in st.session_state and st.session_state.analysis_history:
            latest = st.session_state.analysis_history[-1]
            st.metric("Total Comments", latest['comment_count'])
            
          
            sentiment_counts = latest.get('sentiment_counts', {})
            st.metric("Positive", sentiment_counts.get('Positive', 0))
            st.metric("Negative", sentiment_counts.get('Negative', 0))
            st.metric("Neutral", sentiment_counts.get('Neutral', 0))
    
    
    if hasattr(st.session_state, 'auto_analyze') and st.session_state.auto_analyze:
        if hasattr(st.session_state, 'selected_video_id') and st.session_state.selected_video_id:
            video_url = f"https://www.youtube.com/watch?v={st.session_state.selected_video_id}"
            video_id = extract_video_id(video_url)
            
            if video_id:
                
                perform_analysis(video_id, video_url, api_key)
                
                st.session_state.auto_analyze = False
                if hasattr(st.session_state, 'selected_video_id'):
                    del st.session_state.selected_video_id
                if hasattr(st.session_state, 'selected_video_title'):
                    del st.session_state.selected_video_title
    
   
    if st.button("🚀 Analyze Comments", type="primary", use_container_width=True):
        
        if hasattr(st.session_state, 'selected_video_id') and st.session_state.selected_video_id:
            video_url = f"https://www.youtube.com/watch?v={st.session_state.selected_video_id}"
        
        if not video_url:
            st.error("Please enter a YouTube video URL or select a video from search results")
            return
        
        
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL format.")
            return
        
       
        perform_analysis(video_id, video_url, api_key)
    
   
    if st.session_state.analysis_history:
        st.markdown("---")
        with st.expander("📜 Session History", expanded=False):
            history_df = pd.DataFrame(st.session_state.analysis_history)
            
            
            display_data = []
            for _, row in history_df.iterrows():
                sentiment_counts_dict = row['sentiment_counts']
                if isinstance(sentiment_counts_dict, dict):
                    sentiment_summary = f"P:{sentiment_counts_dict.get('Positive', 0)} N:{sentiment_counts_dict.get('Negative', 0)} U:{sentiment_counts_dict.get('Neutral', 0)}"
                else:
                    sentiment_summary = "N/A"
                    
                display_data.append({
                    'Title': row['title'][:50] + "..." if len(row['title']) > 50 else row['title'],
                    'Comments': row['comment_count'],
                    'Sentiment Summary': sentiment_summary,
                    'Date/Time': row['timestamp'],
                    'Video ID': row['video_id']
                })
            
            st.dataframe(
                pd.DataFrame(display_data),
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("Clear History"):
                st.session_state.analysis_history = []
                st.rerun()

if __name__ == "__main__":
    main()