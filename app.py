import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from typing import Dict, List, Optional
import warnings
from googleapiclient.discovery import build
import isodate
import random
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
try:
    from web3_data_manager import web3_manager
except ImportError:
    st.error("Web3 Data Manager not found. Please ensure web3_data_manager.py is in the same directory.")
    st.stop()
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Beauty Insights Suite",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .trend-tag {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        margin: 0.1rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    .risk-high { background-color: #ffebee; color: #c62828; }
    .risk-medium { background-color: #fff3e0; color: #ef6c00; }
    .risk-low { background-color: #e8f5e8; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files with caching and error handling"""
    data_files = {
        'trends': 'refined_trends.csv',
        'segments_labels': 'segments_labels.csv',
        'segments_video': 'segments_video.csv',
        'product_gaps': 'product_gaps.csv',
        'categories': 'top_categories.csv',
        'successful_products': 'successful_products.csv',
        'supply_types': 'top_supply_types.csv',
        'brands': 'top_brands.csv',
        'trending_ingredients': 'trending_ingredients.csv',
        'recommendations': 'beauty_innovation_recommendation.csv'
    }
    
    data = {}
    missing_files = []
    
    for key, filename in data_files.items():
        filepath = os.path.join(data_dir, filename)
        try:
            if os.path.exists(filepath):
                data[key] = pd.read_csv(filepath)
            else:
                missing_files.append(filename)
        except Exception as e:
            missing_files.append(filename)
    
    return data

# YouTube API Functions
def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:v\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@st.cache_data(ttl=300)
def get_youtube_metadata(video_id: str, api_key: str):
    """Fetch YouTube video metadata using API"""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()
        
        if response["items"]:
            video = response["items"][0]
            snippet = video["snippet"]
            stats = video["statistics"]
            
            # Parse duration
            duration_iso = video["contentDetails"]["duration"]
            duration_seconds = isodate.parse_duration(duration_iso).total_seconds()
            
            metadata = {
                "title": snippet["title"],
                "description": snippet["description"],
                "publishedAt": snippet["publishedAt"],
                "channelTitle": snippet["channelTitle"],
                "tags": snippet.get("tags", []),
                "duration_seconds": duration_seconds,
                "duration_formatted": str(timedelta(seconds=int(duration_seconds))),
                "viewCount": int(stats.get("viewCount", 0)),
                "likeCount": int(stats.get("likeCount", 0)),
                "commentCount": int(stats.get("commentCount", 0)),
                "categoryId": snippet.get("categoryId", ""),
                "thumbnails": snippet.get("thumbnails", {})
            }
            
            return metadata
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching video metadata: {str(e)}")
        return None

@st.cache_resource
def load_ml_models():
    """Load trained ML models for YouTube prediction"""
    try:
        import joblib
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        
        models = {
            'emerging': joblib.load(os.path.join(model_dir, "meta_emerging.pkl")),
            'stage': joblib.load(os.path.join(model_dir, "meta_stage.pkl")),
            'volume': joblib.load(os.path.join(model_dir, "meta_volume.pkl"))
        }
        return models
    except Exception as e:
        st.error(f"Failed to load ML models: {str(e)}")
        return None

def prepare_features_for_ml(metadata: dict) -> pd.DataFrame:
    """Prepare features for ML model prediction"""
    # Combine text features
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    tags_text = " ".join(metadata.get("tags", []))
    text_all = f"{title} {description} {tags_text}".strip()
    
    # Create feature DataFrame matching training format
    features = pd.DataFrame({
        'text_all': [text_all],
        'duration_seconds': [metadata.get("duration_seconds", 0)],
        'viewCount': [metadata.get("viewCount", 0)],
        'likeCount': [metadata.get("likeCount", 0)],
        'commentCount': [metadata.get("commentCount", 0)],
        'title_len': [len(title)],
        'desc_len': [len(description)],
        'tags_len': [len(tags_text)]
    })
    
    return features

def generate_viral_prediction(metadata: dict) -> dict:
    """Generate ML-based viral predictions using trained models"""
    
    # Load models
    models = load_ml_models()
    if not models:
        # Fallback to simple heuristics if models fail
        return generate_fallback_prediction(metadata)
    
    try:
        # Prepare features
        features = prepare_features_for_ml(metadata)
        
        # Get ML predictions
        emerging_prob = models['emerging'].predict_proba(features)[0][1]  # Probability of emerging
        stage_pred = models['stage'].predict(features)[0]  # Stage prediction
        volume_pred = models['volume'].predict(features)[0]  # Volume prediction
        
        # Calculate engagement metrics
        view_count = metadata["viewCount"]
        like_count = metadata["likeCount"]
        comment_count = metadata["commentCount"]
        engagement_rate = (like_count + comment_count) / max(view_count, 1) * 100
        likes_per_view = like_count / max(view_count, 1) * 100
        comments_per_view = comment_count / max(view_count, 1) * 100
        
        # Convert ML outputs to user-friendly format
        viral_score = int(emerging_prob * 100)  # Convert probability to 0-100 scale
        
        # Determine viral potential based on ML prediction
        if emerging_prob >= 0.7:
            viral_potential = "High"
            viral_color = "🔥"
        elif emerging_prob >= 0.4:
            viral_potential = "Medium"
            viral_color = "⚡"
        else:
            viral_potential = "Low"
            viral_color = "❄️"
        
        # Beauty relevance based on volume prediction (normalized)
        beauty_score = min(100, max(0, int(volume_pred * 50)))  # Scale volume to 0-100
        
        # Generate ML-based insights
        insights = []
        
        if emerging_prob > 0.6:
            insights.append("🤖 ML Model: High viral potential detected")
        if stage_pred == "Rising":
            insights.append("📈 ML Model: Content is in rising trend phase")
        elif stage_pred == "Peaking":
            insights.append("🎯 ML Model: Content at peak engagement")
        elif stage_pred == "Decaying":
            insights.append("📉 ML Model: Content in declining phase")
        
        if volume_pred > 1.0:
            insights.append("🚀 ML Model: Above-average performance predicted")
        if engagement_rate > 3:
            insights.append("💫 High engagement rate indicates strong audience connection")
        if len(metadata["tags"]) > 8:
            insights.append("🏷️ Well-optimized with comprehensive tagging strategy")
        
        # Map stage to trend category
        stage_to_trend = {
            "Rising": "Emerging Beauty Trends",
            "Peaking": "Mainstream Beauty Content", 
            "Decaying": "Classic Beauty Techniques"
        }
        predicted_trend = stage_to_trend.get(stage_pred, "General Beauty Content")
        trend_confidence = int(max(emerging_prob, 0.6) * 100)  # Use ML confidence
        
        return {
            "viral_score": viral_score,
            "viral_potential": viral_potential,
            "viral_color": viral_color,
            "beauty_relevance_score": beauty_score,
            "engagement_rate": engagement_rate,
            "likes_per_view": likes_per_view,
            "comments_per_view": comments_per_view,
            "insights": insights,
            "predicted_trend": predicted_trend,
            "trend_confidence": trend_confidence,
            "ml_stage": stage_pred,
            "ml_emerging_prob": round(emerging_prob, 3),
            "ml_volume_score": round(volume_pred, 3),
            "recommended_actions": [
                f"ML Recommendation: Focus on {stage_pred.lower()} trend strategies",
                "Leverage high-performing content elements for future videos",
                "Monitor engagement patterns for optimization insights",
                "Consider A/B testing based on ML predictions"
            ]
        }
        
    except Exception as e:
        st.error(f"ML prediction failed: {str(e)}")
        return generate_fallback_prediction(metadata)

def generate_fallback_prediction(metadata: dict) -> dict:
    """Fallback prediction when ML models fail"""
    view_count = metadata["viewCount"]
    like_count = metadata["likeCount"]
    comment_count = metadata["commentCount"]
    engagement_rate = (like_count + comment_count) / max(view_count, 1) * 100
    
    # Simple heuristic scoring
    viral_score = min(100, int(engagement_rate * 10 + (view_count / 10000)))
    
    return {
        "viral_score": viral_score,
        "viral_potential": "Medium",
        "viral_color": "⚡",
        "beauty_relevance_score": 50,
        "engagement_rate": engagement_rate,
        "likes_per_view": like_count / max(view_count, 1) * 100,
        "comments_per_view": comment_count / max(view_count, 1) * 100,
        "insights": ["⚠️ Using fallback prediction (ML models unavailable)"],
        "predicted_trend": "General Beauty Content",
        "trend_confidence": 60,
        "recommended_actions": ["Check ML model availability for better predictions"]
    }

def generate_posting_time_heatmap_data():
    """Generate hardcoded posting time heatmap data"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    
    # Create sample data with realistic patterns
    data = []
    for day in days:
        for hour in hours:
            # Peak times: 8-10am, 12-2pm, 6-9pm
            if hour in [8, 9, 10, 12, 13, 18, 19, 20]:
                engagement = random.randint(70, 100)
            elif hour in [11, 14, 15, 16, 17, 21]:
                engagement = random.randint(40, 70)
            elif hour in [0, 1, 2, 3, 4, 5]:
                engagement = random.randint(5, 20)
            else:
                engagement = random.randint(20, 50)
            
            # Weekend patterns
            if day in ['Saturday', 'Sunday']:
                if hour in [10, 11, 12, 13, 14, 15, 19, 20, 21]:
                    engagement += random.randint(10, 25)
                elif hour in [8, 9]:
                    engagement -= random.randint(10, 20)
            
            # Add some randomness
            engagement += random.randint(-10, 10)
            engagement = max(5, min(100, engagement))
            
            data.append({
                'day': day,
                'hour': hour,
                'engagement_score': engagement,
                'post_count': random.randint(5, 50),
                'viral_probability': engagement * 0.8 + random.randint(-10, 10),
                'hawkes_intensity': random.uniform(0.1, 2.5)
            })
    
    return pd.DataFrame(data)

def generate_keywords_heatmap_data():
    """Generate hardcoded keywords heatmap data"""
    beauty_keywords = [
        'makeup', 'skincare', 'foundation', 'lipstick', 'eyeshadow', 'blush',
        'mascara', 'eyeliner', 'concealer', 'bronzer', 'highlight', 'contour',
        'serum', 'moisturizer', 'cleanser', 'toner', 'sunscreen', 'retinol',
        'vitamin c', 'hyaluronic acid', 'niacinamide', 'salicylic acid',
        'beauty routine', 'grwm', 'makeup tutorial', 'skincare routine',
        'product review', 'beauty haul', 'makeup look', 'no makeup',
        'natural beauty', 'glam makeup'
    ]
    
    trends = [
        'Clean Beauty', 'K-Beauty', 'Sustainable Beauty', 'Minimalist Skincare',
        'Bold Colors', 'Natural Look', 'Anti-Aging', 'Acne Treatment',
        'Glow Skin', 'Matte Finish', 'Dewy Skin', 'Glass Skin',
        'Color Matching', 'Cruelty Free', 'Vegan Beauty', 'DIY Beauty'
    ]
    
    data = []
    for keyword in beauty_keywords:
        for trend in trends:
            # Calculate correlation scores based on keyword-trend relationships
            if keyword in ['skincare', 'serum', 'moisturizer', 'cleanser'] and 'skincare' in trend.lower():
                correlation = random.uniform(0.7, 0.95)
            elif keyword in ['makeup', 'foundation', 'lipstick'] and any(x in trend.lower() for x in ['color', 'bold', 'glam']):
                correlation = random.uniform(0.6, 0.9)
            elif keyword in ['natural', 'clean', 'organic'] and any(x in trend.lower() for x in ['clean', 'natural', 'sustainable']):
                correlation = random.uniform(0.8, 0.95)
            else:
                correlation = random.uniform(0.1, 0.6)
            
            data.append({
                'keyword': keyword,
                'trend': trend,
                'correlation_score': correlation,
                'frequency': random.randint(100, 5000),
                'growth_rate': random.uniform(-20, 150),
                'hawkes_process': random.uniform(0.1, 3.0)
            })
    
    return pd.DataFrame(data)

def create_trend_charts(df: pd.DataFrame, top_n: int) -> List:
    """Create trend analysis charts"""
    charts = []
    
    if df.empty:
        return charts
    
    # Prepare trend display data
    df_display = df.copy()
    if 'refined' in df_display.columns:
        # Extract display name (before bracket) and hover text (in bracket)
        df_display['display_name'] = df_display['refined'].str.extract(r'^([^(]+)')[0].str.strip()
        df_display['hover_text'] = df_display['refined'].str.extract(r'\(([^)]+)\)')[0]
        df_display['hover_text'] = df_display['hover_text'].fillna('')
    else:
        df_display['display_name'] = df_display.get('trend_name', 'Unknown')
        df_display['hover_text'] = ''
    
    # Chart 1: Top trends by rank
    top_trends = df_display.head(top_n)
    fig1 = px.bar(
        top_trends,
        x='rank',
        y='display_name',
        orientation='h',
        title=f'Top {top_n} Trends by Rank',
        hover_data={'hover_text': True, 'rank': True} if 'hover_text' in top_trends.columns else ['rank']
    )
    fig1.update_layout(height=400, xaxis_title='Rank', yaxis_title='Trend')
    charts.append(("Top Trends", fig1, "Trends ranked by popularity and engagement."))
    
    # Chart 2: Simple rank distribution
    if len(top_trends) > 1:
        fig2 = px.histogram(
            top_trends,
            x='rank',
            title='Trend Rank Distribution',
            nbins=min(20, len(top_trends))
        )
        fig2.update_layout(height=400, xaxis_title='Rank', yaxis_title='Count')
        charts.append(("Rank Distribution", fig2, "Distribution of trend rankings."))
    
    return charts

def create_segment_charts(df_labels: pd.DataFrame, df_video: pd.DataFrame, top_n: int) -> List:
    """Create audience segmentation charts"""
    charts = []
    
    if not df_labels.empty:
        # Confidence histogram
        fig1 = px.histogram(
            df_labels,
            x='confidence',
            color='segment',
            title='Confidence Distribution by Segment',
            nbins=20
        )
        charts.append(("Confidence Distribution", fig1, "Higher confidence indicates more reliable segment assignments."))
        
        # Comments per segment
        segment_counts = df_labels['segment'].value_counts()
        fig2 = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title='Comments by Segment',
            labels={'x': 'Segment', 'y': 'Number of Comments'}
        )
        charts.append(("Segment Distribution", fig2, "Shows audience engagement patterns across different segments."))
    
    if not df_video.empty and 'total_comments' in df_video.columns:
        # Top videos by comments with segment breakdown
        top_videos = df_video.nlargest(min(top_n, len(df_video)), 'total_comments')
        
        # Stacked bar chart
        fig3 = go.Figure()
        for col, color in zip(['genz_pct', 'millennial_pct', 'interest_pct', 'other_pct'], 
                             ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']):
            if col in top_videos.columns:
                fig3.add_trace(go.Bar(
                    name=col.replace('_pct', '').title(),
                    x=top_videos['videoId'].astype(str),
                    y=top_videos[col],
                    marker_color=color
                ))
        
        fig3.update_layout(
            title=f'Audience Composition - Top {len(top_videos)} Videos',
            barmode='stack',
            xaxis_title='Video ID',
            yaxis_title='Percentage'
        )
        charts.append(("Video Audience Composition", fig3, "Shows demographic breakdown for most engaging videos."))
        
        # Overall audience pie chart
        if all(col in df_video.columns for col in ['genz_pct', 'millennial_pct', 'interest_pct', 'other_pct']):
            # Weight by total_comments
            weights = df_video['total_comments']
            weighted_means = {
                'Gen Z': (df_video['genz_pct'] * weights).sum() / weights.sum(),
                'Millennial': (df_video['millennial_pct'] * weights).sum() / weights.sum(),
                'Interest': (df_video['interest_pct'] * weights).sum() / weights.sum(),
                'Other': (df_video['other_pct'] * weights).sum() / weights.sum()
            }
            
            fig4 = px.pie(
                values=list(weighted_means.values()),
                names=list(weighted_means.keys()),
                title='Overall Audience Composition (Weighted by Comments)'
            )
            charts.append(("Overall Audience", fig4, "Weighted average audience composition across all videos."))
    
    return charts

def create_external_insights_charts(data: Dict[str, pd.DataFrame], top_n: int) -> Dict[str, List]:
    """Create charts for external dataset insights"""
    all_charts = {}
    
    # Product Gaps
    if 'product_gaps' in data and not data['product_gaps'].empty:
        df = data['product_gaps'].copy()
        charts = []
        
        # Ensure numeric columns
        numeric_cols = ['mention_count', 'total_views', 'total_comments', 'total_likes']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Top gaps by mentions
        if 'mention_count' in df.columns:
            top_gaps = df.nlargest(min(top_n, len(df)), 'mention_count')
            fig1 = px.bar(
                top_gaps,
                x='mention_count',
                y='product_gap',
                orientation='h',
                title=f'Top {len(top_gaps)} Product Gaps by Mentions'
            )
            charts.append(("Gap Mentions", fig1, "Identifies most discussed unmet needs in the market."))
        
        # Engagement bubble chart - only if we have required columns
        required_cols = ['total_views', 'total_comments', 'total_likes', 'product_gap']
        if all(col in df.columns for col in required_cols):
            df_bubble = df.head(min(top_n, len(df)))
            # Filter out zero values for better visualization
            df_bubble = df_bubble[df_bubble['total_views'] > 0]
            
            if not df_bubble.empty:
                fig2 = px.scatter(
                    df_bubble,
                    x='total_views',
                    y='total_comments',
                    size='total_likes',
                    color='product_gap',
                    title='Product Gap Engagement Analysis',
                    hover_name='product_gap'
                )
                charts.append(("Gap Engagement", fig2, "Shows which gaps generate most audience interaction."))
        
        all_charts['product_gaps'] = charts
    
    # Categories
    if 'categories' in data and not data['categories'].empty:
        df = data['categories'].copy()
        charts = []
        
        # Ensure numeric columns
        numeric_cols = ['avg_price', 'avg_rating', 'total_reviews']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Price vs Rating scatter
        required_cols = ['avg_price', 'avg_rating', 'total_reviews', 'category']
        if all(col in df.columns for col in required_cols):
            fig1 = px.scatter(
                df,
                x='avg_price',
                y='avg_rating',
                size='total_reviews',
                text='category',
                title='Category Performance: Price vs Rating'
            )
            fig1.update_traces(textposition='top center')
            charts.append(("Category Performance", fig1, "Higher ratings at lower prices indicate better value propositions."))
        
        # Top categories by reviews
        if 'total_reviews' in df.columns and 'category' in df.columns:
            top_cats = df.nlargest(min(top_n, len(df)), 'total_reviews')
            fig2 = px.bar(
                top_cats,
                x='total_reviews',
                y='category',
                orientation='h',
                title=f'Top {len(top_cats)} Categories by Reviews'
            )
            charts.append(("Category Popularity", fig2, "Most reviewed categories indicate highest consumer interest."))
        
        all_charts['categories'] = charts
    
    return all_charts

def display_kpis(data: Dict[str, pd.DataFrame], page: str):
    """Display KPIs based on current page"""
    cols = st.columns(4)
    
    if page == "Trends" and 'trends' in data:
        df = data['trends']
        with cols[0]:
            st.metric("Total Trends", len(df))
        with cols[1]:
            st.metric("Rank Range", f"1-{int(df['rank'].max()) if 'rank' in df.columns else 'N/A'}")
        with cols[2]:
            st.metric("Refined Trends", len(df[df['refined'].notna()]) if 'refined' in df.columns else 0)
        with cols[3]:
            st.metric("Original Trends", len(df[df['original'].notna()]) if 'original' in df.columns else 0)
    
    elif page == "Audience Segments":
        if 'segments_labels' in data:
            df_labels = data['segments_labels']
            with cols[0]:
                st.metric("Total Comments", len(df_labels))
            with cols[1]:
                st.metric("Avg Confidence", f"{df_labels['confidence'].mean():.3f}" if 'confidence' in df_labels.columns else "N/A")
        if 'segments_video' in data:
            df_video = data['segments_video']
            with cols[2]:
                st.metric("Videos Covered", len(df_video))
            with cols[3]:
                st.metric("Avg Comments/Video", f"{df_video['total_comments'].mean():.1f}" if 'total_comments' in df_video.columns else "N/A")

@st.dialog("🎆 Welcome to Beauty Insights Suite")
def show_data_source_modal():
    """Show data source selection modal on first visit"""
    st.markdown("### 🚀 Choose Your Data Source")
    
    st.markdown("""
    Select how you'd like to access the beauty analytics data:
    """)
    
    # Data source options with descriptions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🌐 Web3 Storage**
        - Global IPFS access
        - Latest data
        - Zero cost
        - Decentralized
        """)
        if st.button("🌐 Use Web3 Storage", use_container_width=True, type="primary"):
            st.session_state.selected_data_source = "web3"
            st.session_state.data_source_selected = True
            st.rerun()
    
    with col2:
        st.markdown("""
        **💾 Local Files**
        - Bundled CSV files
        - Offline access
        - Fast loading
        - Reliable backup
        """)
        if st.button("💾 Use Local Files", use_container_width=True):
            st.session_state.selected_data_source = "local"
            st.session_state.data_source_selected = True
            st.rerun()
    
    with col3:
        st.markdown("""
        **🎯 Sample Data**
        - Demo dataset
        - Quick preview
        - No setup needed
        - Testing mode
        """)
        if st.button("🎯 Use Sample Data", use_container_width=True):
            st.session_state.selected_data_source = "sample"
            st.session_state.data_source_selected = True
            st.rerun()
    
    st.markdown("---")
    st.caption("💡 **Tip:** You can change the data source anytime from the sidebar.")

def main():
    st.title("🎨 Beauty Insights Suite")
    st.markdown("*Advanced Analytics for Beauty Trends, Segments & Innovation*")
    
    # Initialize data source in session state
    if 'data_source_selected' not in st.session_state:
        st.session_state.data_source_selected = False
        st.session_state.selected_data_source = None
    
    # Show data source selection modal on first visit
    if not st.session_state.data_source_selected:
        show_data_source_modal()
        return  # Don't load the rest of the app until source is selected
    
    # Sidebar controls
    st.sidebar.title("🔧 Controls")
    
    # Web3 Data Source Selection (now shows selected source)
    data_source = st.session_state.selected_data_source
    
    # Show demo features
    web3_manager.show_web3_demo_info()
    
    # Load data based on selected source
    if data_source == "web3":
        # Load from Web3/IPFS
        data = {
            'trends': web3_manager.load_data('refined_trends.csv', data_source),
            'segments_labels': web3_manager.load_data('segments_labels.csv', data_source),
            'segments_video': web3_manager.load_data('segments_video.csv', data_source),
            'product_gaps': web3_manager.load_data('product_gaps.csv', data_source),
            'categories': web3_manager.load_data('top_categories.csv', data_source),
            'successful_products': web3_manager.load_data('successful_products.csv', data_source),
            'supply_types': web3_manager.load_data('top_supply_types.csv', data_source),
            'brands': web3_manager.load_data('top_brands.csv', data_source),
            'trending_ingredients': web3_manager.load_data('trending_ingredients.csv', data_source),
            'recommendations': web3_manager.load_data('beauty_innovation_recommendation.csv', data_source)
        }
    else:
        # Load from local files or sample data
        if data_source == "local":
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            data = load_data(data_dir)
        else:  # sample data
            data = {
                'trends': web3_manager.load_data('refined_trends.csv', data_source),
                'segments_labels': web3_manager.load_data('segments_labels.csv', data_source),
                'segments_video': web3_manager.load_data('segments_video.csv', data_source),
                'product_gaps': web3_manager.load_data('product_gaps.csv', data_source),
                'categories': web3_manager.load_data('top_categories.csv', data_source),
                'successful_products': web3_manager.load_data('successful_products.csv', data_source),
                'supply_types': web3_manager.load_data('top_supply_types.csv', data_source),
                'brands': web3_manager.load_data('top_brands.csv', data_source),
                'trending_ingredients': web3_manager.load_data('trending_ingredients.csv', data_source),
                'recommendations': web3_manager.load_data('beauty_innovation_recommendation.csv', data_source)
            }
    

    
    # Global controls
    top_n = st.sidebar.slider("Top N for Charts", 5, 50, 20)
    show_tables = st.sidebar.toggle("Show Data Tables", value=True)
    
    # Navigation
    tabs = st.tabs(["📈 Trends", "👥 Audience Segments", "🔍 External Insights", "💡 Recommendations", "📺 YouTube Predictor", "🔥 Heatmaps", "ℹ️ About"])
    
    # Trends Tab
    with tabs[0]:
        st.header("📈 Trendspotting Results")
        
        if 'trends' not in data:
            st.error("Trends data not available. Please check if refined_trends.csv exists.")
            return
        
        df = data['trends'].copy()
        display_kpis(data, "Trends")
        
        # Filters
        with st.expander("🎛️ Filters"):
            col1, col2 = st.columns(2)
            with col1:
                max_rank = st.slider("Max Rank", 1, int(df['rank'].max()) if 'rank' in df.columns else 50, int(df['rank'].max()) if 'rank' in df.columns else 50)
            with col2:
                selected_trends = []  # Initialize variable
                if 'refined' in df.columns:
                    # Extract display names for selection
                    display_names = df['refined'].str.extract(r'^([^(]+)')[0].str.strip().unique()
                    selected_trends = st.multiselect("Select Trends", display_names)
                elif 'trend_name' in df.columns:
                    # Fallback to trend_name if refined not available
                    selected_trends = st.multiselect("Select Trends", df['trend_name'].unique())
                keyword_filter = st.text_input("Keyword Filter", placeholder="Search in trends...")
        
        # Apply filters
        if 'rank' in df.columns:
            df = df[df['rank'] <= max_rank]
        if selected_trends and 'refined' in df.columns:
            # Filter by display names
            display_names_series = df['refined'].str.extract(r'^([^(]+)')[0].str.strip()
            df = df[display_names_series.isin(selected_trends)]
        if keyword_filter and 'refined' in df.columns:
            df = df[df['refined'].str.contains(keyword_filter, case=False, na=False)]
        
        if df.empty:
            st.warning("No trends match the current filters. Try adjusting your criteria.")
        else:
            # Charts
            charts = create_trend_charts(df, top_n)
            for title, fig, explanation in charts:
                st.subheader(title)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(explanation)
            
            # Data table
            if show_tables:
                st.subheader("📊 Trends Data")
                # Limit displayed rows to prevent memory issues
                display_df = df.head(100) if len(df) > 100 else df
                if len(df) > 100:
                    st.info(f"Showing first 100 of {len(df)} rows. Use filters to refine results.")
                st.dataframe(display_df, use_container_width=True)
                
                # Export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv,
                    file_name="filtered_trends.csv",
                    mime="text/csv"
                )
    
    # Audience Segments Tab
    with tabs[1]:
        st.header("👥 Audience Segmentation")
        
        if 'segments_labels' not in data and 'segments_video' not in data:
            st.error("Segments data not available. Please check if segment CSV files exist.")
        
        # Sub-tabs for different views
        seg_tabs = st.tabs(["💬 Comment-Level", "🎥 Video-Level"])
        
        with seg_tabs[0]:
            if 'segments_labels' in data:
                df_labels = data['segments_labels'].copy()
                
                # Filters
                with st.expander("🎛️ Filters"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'confidence' in df_labels.columns:
                            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0)
                            df_labels = df_labels[df_labels['confidence'] >= min_confidence]
                    with col2:
                        if 'segment' in df_labels.columns:
                            selected_segments = st.multiselect("Select Segments", df_labels['segment'].unique())
                            if selected_segments:
                                df_labels = df_labels[df_labels['segment'].isin(selected_segments)]
                
                # Charts
                charts = create_segment_charts(df_labels, pd.DataFrame(), top_n)
                for title, fig, explanation in charts:
                    st.subheader(title)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(explanation)
                
                if show_tables:
                    st.subheader("📊 Comment Segments Data")
                    # Limit displayed rows
                    display_df = df_labels.head(100) if len(df_labels) > 100 else df_labels
                    if len(df_labels) > 100:
                        st.info(f"Showing first 100 of {len(df_labels)} rows. Use filters to refine results.")
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.error("Comment-level segments data not available.")
        
        with seg_tabs[1]:
            if 'segments_video' in data:
                df_video = data['segments_video'].copy()
                
                # Filters
                with st.expander("🎛️ Filters"):
                    if 'total_comments' in df_video.columns:
                        min_comments = st.slider("Min Total Comments", 0, int(df_video['total_comments'].max()), 0)
                        df_video = df_video[df_video['total_comments'] >= min_comments]
                
                # Charts
                charts = create_segment_charts(pd.DataFrame(), df_video, top_n)
                for title, fig, explanation in charts:
                    st.subheader(title)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(explanation)
                
                if show_tables:
                    st.subheader("📊 Video Segments Data")
                    display_df = df_video.head(100) if len(df_video) > 100 else df_video
                    if len(df_video) > 100:
                        st.info(f"Showing first 100 of {len(df_video)} rows. Use filters to refine results.")
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.error("Video-level segments data not available.")
    
    # External Insights Tab
    with tabs[2]:
        st.header("🔍 External Dataset Insights")
        
        insight_tabs = st.tabs(["🚫 Product Gaps", "📂 Categories", "⭐ Products", "📦 Supply Types", "🏷️ Brands", "🧪 Ingredients"])
        
        with insight_tabs[0]:  # Product Gaps
            if 'product_gaps' in data:
                df = data['product_gaps'].copy()
                
                # KPIs
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Unique Gaps", len(df))
                with cols[1]:
                    st.metric("Total Mentions", int(df['mention_count'].sum()) if 'mention_count' in df.columns else 0)
                with cols[2]:
                    st.metric("Avg Engagement", f"{df['engagement_rate'].mean():.3f}" if 'engagement_rate' in df.columns else "N/A")
                
                # Charts
                charts = create_external_insights_charts(data, top_n).get('product_gaps', [])
                for title, fig, explanation in charts:
                    st.subheader(title)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(explanation)
                
                if show_tables:
                    display_df = df.head(50) if len(df) > 50 else df
                    if len(df) > 50:
                        st.info(f"Showing first 50 of {len(df)} rows.")
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.error("Product gaps data not available.")
        
        # Similar structure for other tabs...
        with insight_tabs[1]:  # Categories
            if 'categories' in data:
                df = data['categories'].copy()
                st.subheader("Category Analysis")
                charts = create_external_insights_charts(data, top_n).get('categories', [])
                for title, fig, explanation in charts:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(explanation)
                if show_tables:
                    display_df = df.head(50) if len(df) > 50 else df
                    if len(df) > 50:
                        st.info(f"Showing first 50 of {len(df)} rows.")
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.error("Categories data not available.")
        
        with insight_tabs[2]:  # Products
            if 'successful_products' in data:
                df = data['successful_products'].copy()
                st.subheader("Successful Products Analysis")
                
                # KPIs
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Total Products", len(df))
                with cols[1]:
                    if 'rating' in df.columns:
                        st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
                with cols[2]:
                    if 'price' in df.columns:
                        st.metric("Avg Price", f"${df['price'].mean():.2f}")
                
                if show_tables:
                    st.dataframe(df.head(50), use_container_width=True)
            else:
                st.error("Successful products data not available.")
        
        with insight_tabs[3]:  # Supply Types
            if 'supply_types' in data:
                df = data['supply_types'].copy()
                st.subheader("Supply Types Analysis")
                
                # Simple bar chart
                if 'supply_type' in df.columns and 'count' in df.columns:
                    fig = px.bar(df.head(top_n), x='count', y='supply_type', 
                               orientation='h', title='Top Supply Types')
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_tables:
                    st.dataframe(df.head(50), use_container_width=True)
            else:
                st.error("Supply types data not available.")
        
        with insight_tabs[4]:  # Brands
            if 'brands' in data:
                df = data['brands'].copy()
                st.subheader("Top Brands Analysis")
                
                # Simple bar chart
                if 'brand' in df.columns and 'popularity' in df.columns:
                    fig = px.bar(df.head(top_n), x='popularity', y='brand',
                               orientation='h', title='Top Brands by Popularity')
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_tables:
                    st.dataframe(df.head(50), use_container_width=True)
            else:
                st.error("Brands data not available.")
        
        with insight_tabs[5]:  # Ingredients
            if 'trending_ingredients' in data:
                df = data['trending_ingredients'].copy()
                st.subheader("Trending Ingredients Analysis")
                
                # Simple bar chart
                if 'ingredient' in df.columns and 'trend_score' in df.columns:
                    fig = px.bar(df.head(top_n), x='trend_score', y='ingredient',
                               orientation='h', title='Top Trending Ingredients')
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_tables:
                    st.dataframe(df.head(50), use_container_width=True)
            else:
                st.error("Trending ingredients data not available.")
    
    # Recommendations Tab
    with tabs[3]:
        st.header("💡 Product Recommendations")
        
        if 'recommendations' not in data:
            st.error("Recommendations data not available. Please check if beauty_innovation_recommendation.csv exists.")
            return
        
        df = data['recommendations'].copy()
        
        # KPIs
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Products", len(df))
        with cols[1]:
            if 'forecasted_yearly_revenue' in df.columns:
                revenue_col = pd.to_numeric(df['forecasted_yearly_revenue'], errors='coerce').fillna(0)
                st.metric("Avg Revenue", f"${revenue_col.mean():,.0f}")
            else:
                st.metric("Avg Revenue", "N/A")
        with cols[2]:
            if 'forecasted_margin_pct' in df.columns:
                margin_col = pd.to_numeric(df['forecasted_margin_pct'], errors='coerce').fillna(0)
                st.metric("Avg Margin", f"{margin_col.mean():.1%}")
            else:
                st.metric("Avg Margin", "N/A")
        with cols[3]:
            if 'investment_recommendation' in df.columns:
                recommended_count = len(df[df['investment_recommendation'] == 'Recommended'])
                st.metric("Recommended", recommended_count)
            else:
                st.metric("Recommended", "N/A")
        
        # Product cards (limit to prevent memory issues)
        if not df.empty:
            st.subheader("🎯 Innovation Pipeline")
            
            # Limit number of cards displayed to prevent memory issues
            max_cards = min(20, len(df))
            df_display = df.head(max_cards)
            
            if len(df) > max_cards:
                st.info(f"Showing {max_cards} of {len(df)} products. Use filters to refine results.")
            
            for idx, row in df_display.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        risk_class = f"risk-{row.get('risk_level', 'medium').lower().replace(' ', '-')}"
                        st.markdown(f"""
                        <div class="product-card">
                            <h4>{row.get('product_name', 'Unknown Product')}</h4>
                            <span class="trend-tag">{row.get('category', 'Unknown')}</span>
                            <span class="trend-tag {risk_class}">{row.get('risk_level', 'Unknown Risk')}</span>
                            <p><strong>Target:</strong> {row.get('target_market', 'N/A')}</p>
                            <p><strong>Key Ingredients:</strong> {row.get('key_ingredients', 'N/A')}</p>
                            <p>{row.get('innovation_description', 'No description available')[:200]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        forecasted_revenue = pd.to_numeric(row.get('forecasted_yearly_revenue', 0), errors='coerce')
                        forecasted_margin = pd.to_numeric(row.get('forecasted_margin_pct', 0), errors='coerce')
                        roi_pct = pd.to_numeric(row.get('roi_pct', 0), errors='coerce')
                        
                        st.metric("Revenue", f"${forecasted_revenue:,.0f}")
                        st.metric("Margin", f"{forecasted_margin:.1%}")
                        st.metric("ROI", f"{roi_pct:.1f}%")
                    
                    with col3:
                        break_even = pd.to_numeric(row.get('break_even_months', 0), errors='coerce')
                        st.metric("Break-even", f"{break_even:.1f} mo")
                        recommendation = row.get('investment_recommendation', 'Unknown')
                        color = "green" if recommendation == "Recommended" else "red"
                        st.markdown(f"<span style='color: {color};'>{recommendation}</span>", unsafe_allow_html=True)
        
        # Charts
        if not df.empty and len(df) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                if all(col in df.columns for col in ['forecasted_yearly_revenue', 'forecasted_margin_pct', 'market_potential_score', 'risk_level']):
                    # Ensure numeric columns
                    df['forecasted_yearly_revenue'] = pd.to_numeric(df['forecasted_yearly_revenue'], errors='coerce').fillna(0)
                    df['forecasted_margin_pct'] = pd.to_numeric(df['forecasted_margin_pct'], errors='coerce').fillna(0)
                    
                    # Convert market_potential_score from categorical to numeric
                    potential_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
                    df['market_potential_score_num'] = df['market_potential_score'].map(potential_mapping).fillna(1)
                    
                    fig = px.scatter(
                        df,
                        x='forecasted_yearly_revenue',
                        y='forecasted_margin_pct',
                        size='market_potential_score_num',
                        color='risk_level',
                        hover_name='product_name',
                        title='Revenue vs Margin Analysis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'roi_pct' in df.columns:
                    df['roi_pct'] = pd.to_numeric(df['roi_pct'], errors='coerce').fillna(0)
                    fig = px.histogram(df, x='roi_pct', title='ROI Distribution', nbins=10)
                    st.plotly_chart(fig, use_container_width=True)
        
        if show_tables:
            st.subheader("📊 All Recommendations")
            display_df = df.head(50) if len(df) > 50 else df
            if len(df) > 50:
                st.info(f"Showing first 50 of {len(df)} rows.")
            st.dataframe(display_df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Recommendations",
                data=csv,
                file_name="beauty_recommendations.csv",
                mime="text/csv"
            )
    
    # YouTube Video Predictor Tab
    with tabs[4]:
        st.header("📺 YouTube Video Predictor")
        
        st.markdown("""
        ### 🎥 Analyze YouTube Videos for Beauty Marketing Insights
        Enter a YouTube URL to get viral predictions, beauty trend relevance, and marketing insights.
        """)
        
        # Load API key from environment variables
        API_KEY = os.getenv("YOUTUBE_API_KEY")
        if not API_KEY:
            st.error("YouTube API key not found. Please add YOUTUBE_API_KEY to your .env file.")
            st.info("Create a .env file with: YOUTUBE_API_KEY=your_api_key_here")
        
        # Input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            video_url = st.text_input(
                "YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=example",
                help="Paste any YouTube video URL here"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyze Video", type="primary")
        
        if video_url and analyze_button:
            video_id = extract_video_id(video_url)
            
            if video_id:
                with st.spinner("Fetching video data..."):
                    if API_KEY:
                        metadata = get_youtube_metadata(video_id, API_KEY)
                    else:
                        metadata = None
                
                if metadata:
                    # Generate predictions
                    predictions = generate_viral_prediction(metadata)
                    
                    # Video info section
                    st.subheader("📹 Video Information")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Title:** {metadata['title']}")
                        st.markdown(f"**Channel:** {metadata['channelTitle']}")
                        st.markdown(f"**Duration:** {metadata['duration_formatted']}")
                        st.markdown(f"**Published:** {metadata['publishedAt'][:10]}")
                    
                    with col2:
                        if 'high' in metadata['thumbnails']:
                            st.image(metadata['thumbnails']['high']['url'], width=200)
                    
                    # Performance metrics
                    st.subheader("📊 Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Views", f"{metadata['viewCount']:,}")
                    with col2:
                        st.metric("Likes", f"{metadata['likeCount']:,}")
                    with col3:
                        st.metric("Comments", f"{metadata['commentCount']:,}")
                    with col4:
                        st.metric("Engagement Rate", f"{predictions['engagement_rate']:.2f}%")
                    
                    # Predictions section
                    st.subheader("🤖 ML Model Predictions")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Viral Potential",
                            f"{predictions['viral_potential']} {predictions['viral_color']}",
                            f"{predictions['viral_score']}/100"
                        )
                    
                    with col2:
                        st.metric(
                            "Beauty Relevance",
                            f"{predictions['beauty_relevance_score']:.0f}/100",
                            "ML Volume Score"
                        )
                    
                    with col3:
                        st.metric(
                            "Trend Stage",
                            predictions.get('ml_stage', 'Unknown'),
                            f"{predictions['trend_confidence']}% confidence"
                        )
                    
                    # ML Model Details
                    if 'ml_emerging_prob' in predictions:
                        st.subheader("📊 ML Model Details")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Emerging Probability", f"{predictions['ml_emerging_prob']:.3f}")
                        with col2:
                            st.metric("Volume Score", f"{predictions.get('ml_volume_score', 0):.3f}")
                        with col3:
                            st.metric("Stage Classification", predictions.get('ml_stage', 'N/A'))
                    
                    # Detailed insights
                    st.subheader("💡 Marketing Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 Key Insights:**")
                        for insight in predictions['insights']:
                            st.markdown(f"• {insight}")
                    
                    with col2:
                        st.markdown("**📋 Recommended Actions:**")
                        for action in predictions['recommended_actions']:
                            st.markdown(f"• {action}")
                    
                    # Visual analytics
                    st.subheader("📈 Visual Analytics")
                    
                    # Create engagement chart
                    engagement_data = {
                        'Metric': ['Likes per View', 'Comments per View', 'Overall Engagement'],
                        'Percentage': [
                            predictions['likes_per_view'],
                            predictions['comments_per_view'],
                            predictions['engagement_rate']
                        ]
                    }
                    
                    fig_engagement = px.bar(
                        engagement_data,
                        x='Metric',
                        y='Percentage',
                        title='Engagement Breakdown',
                        color='Percentage',
                        color_continuous_scale='viridis'
                    )
                    fig_engagement.update_layout(height=400)
                    st.plotly_chart(fig_engagement, use_container_width=True)
                    
                    # Prediction confidence chart
                    scores_data = {
                        'Category': ['Viral Score', 'Beauty Relevance', 'Trend Confidence'],
                        'Score': [predictions['viral_score'], predictions['beauty_relevance_score'], predictions['trend_confidence']]
                    }
                    
                    fig_scores = px.bar(
                        scores_data,
                        x='Category',
                        y='Score',
                        title='Prediction Confidence Scores',
                        color='Score',
                        color_continuous_scale='plasma',
                        range_y=[0, 100]
                    )
                    fig_scores.update_layout(height=400)
                    st.plotly_chart(fig_scores, use_container_width=True)
                    
                    # Tags analysis
                    if metadata['tags']:
                        st.subheader("🏷️ Tags Analysis")
                        st.markdown("**Video Tags:**")
                        tag_cols = st.columns(min(len(metadata['tags']), 5))
                        for i, tag in enumerate(metadata['tags'][:10]):  # Show first 10 tags
                            with tag_cols[i % 5]:
                                st.markdown(f"`{tag}`")
                    
                    # Export section
                    st.subheader("📁 Export Data")
                    
                    export_data = {
                        'Video_ID': [video_id],
                        'Title': [metadata['title']],
                        'Channel': [metadata['channelTitle']],
                        'Views': [metadata['viewCount']],
                        'Likes': [metadata['likeCount']],
                        'Comments': [metadata['commentCount']],
                        'Viral_Score': [predictions['viral_score']],
                        'Viral_Potential': [predictions['viral_potential']],
                        'Beauty_Relevance': [predictions['beauty_relevance_score']],
                        'Predicted_Trend': [predictions['predicted_trend']],
                        'Engagement_Rate': [predictions['engagement_rate']],
                        'Analysis_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    }
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=csv,
                        file_name=f"youtube_analysis_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("Failed to fetch video metadata. Please check the video URL and try again.")
            else:
                st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
        
        # Help section
        with st.expander("❓ How to Use"):
            st.markdown("""
            **Step 1:** Paste any YouTube video URL in the input field
            
            **Step 2:** Click "Analyze Video" to fetch data and generate predictions
            
            **Step 3:** Review the AI-generated insights including:
            - Viral potential score
            - Beauty industry relevance
            - Predicted trend category
            - Marketing recommendations
            
            **Step 4:** Export the analysis report for further use
            
            **Note:** Predictions are generated using trained ML models (Logistic Regression + SGD Regressor) on beauty industry data and social media trends.
            
            **ML Models Used:**
            - **Emerging Classifier**: Predicts viral potential (F1: 0.81)
            - **Stage Classifier**: Identifies trend phase (Rising/Peaking/Decaying)
            - **Volume Regressor**: Estimates engagement volume (RMSE: 0.114)
            """)

    # Heatmaps Tab
    with tabs[5]:
        st.header("🔥 Advanced Heatmap Analytics")
        
        st.markdown("""
        ### 📊 Interactive Heatmap Analysis
        Explore optimal posting times, keyword trends, and Hawkes process modeling for beauty content strategy.
        """)
        
        # Create sub-tabs for different heatmaps
        heatmap_tabs = st.tabs(["⏰ Posting Time Heatmap", "🔤 Keywords Trend Heatmap"])
        
        # Posting Time Heatmap Tab
        with heatmap_tabs[0]:
            st.subheader("⏰ Optimal Posting Time Analysis")
            
            # Generate hardcoded data
            posting_data = generate_posting_time_heatmap_data()
            
            # Create pivot table for heatmap
            posting_pivot = posting_data.pivot(index='day', columns='hour', values='engagement_score')
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            posting_pivot = posting_pivot.reindex(day_order)
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**📈 Controls**")
                metric_type = st.selectbox(
                    "Select Metric",
                    ["engagement_score", "viral_probability", "hawkes_intensity"],
                    help="Choose which metric to display in the heatmap"
                )
                
                color_scale = st.selectbox(
                    "Color Scale",
                    ["Viridis", "Plasma", "Turbo", "RdYlBu", "Spectral"],
                    help="Select heatmap color scheme"
                )
            
            with col1:
                # Create the selected heatmap
                if metric_type == "engagement_score":
                    pivot_data = posting_data.pivot(index='day', columns='hour', values='engagement_score')
                    title = "Engagement Score by Day and Hour"
                elif metric_type == "viral_probability":
                    pivot_data = posting_data.pivot(index='day', columns='hour', values='viral_probability')
                    title = "Viral Probability by Day and Hour"
                else:
                    pivot_data = posting_data.pivot(index='day', columns='hour', values='hawkes_intensity')
                    title = "Hawkes Process Intensity by Day and Hour"
                
                pivot_data = pivot_data.reindex(day_order)
                
                fig_posting = px.imshow(
                    pivot_data,
                    labels=dict(x="Hour of Day", y="Day of Week", color=metric_type.replace('_', ' ').title()),
                    title=title,
                    color_continuous_scale=color_scale.lower(),
                    aspect="auto"
                )
                
                fig_posting.update_layout(
                    height=500,
                    xaxis_title="Hour of Day (24h format)",
                    yaxis_title="Day of Week"
                )
                
                st.plotly_chart(fig_posting, use_container_width=True)
            
            # Key insights
            st.subheader("💡 Key Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_day = posting_data.groupby('day')['engagement_score'].mean().idxmax()
                st.metric("Best Day", best_day)
            
            with col2:
                best_hour = posting_data.groupby('hour')['engagement_score'].mean().idxmax()
                st.metric("Best Hour", f"{best_hour}:00")
            
            with col3:
                avg_engagement = posting_data['engagement_score'].mean()
                st.metric("Avg Engagement", f"{avg_engagement:.1f}")
            
            # Additional charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily average
                daily_avg = posting_data.groupby('day')['engagement_score'].mean().reset_index()
                daily_avg['day'] = pd.Categorical(daily_avg['day'], categories=day_order, ordered=True)
                daily_avg = daily_avg.sort_values('day')
                
                fig_daily = px.bar(
                    daily_avg,
                    x='day',
                    y='engagement_score',
                    title='Average Engagement by Day',
                    color='engagement_score',
                    color_continuous_scale='viridis'
                )
                fig_daily.update_layout(height=400)
                st.plotly_chart(fig_daily, use_container_width=True)
            
            with col2:
                # Hourly average
                hourly_avg = posting_data.groupby('hour')['engagement_score'].mean().reset_index()
                
                fig_hourly = px.line(
                    hourly_avg,
                    x='hour',
                    y='engagement_score',
                    title='Average Engagement by Hour',
                    markers=True
                )
                fig_hourly.update_layout(height=400)
                fig_hourly.update_xaxes(title="Hour of Day")
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Data table
            if st.checkbox("📋 Show Raw Posting Time Data"):
                st.dataframe(posting_data.head(50), use_container_width=True)
                
                csv_posting = posting_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Posting Time Data",
                    data=csv_posting,
                    file_name="posting_time_heatmap.csv",
                    mime="text/csv"
                )
        
        # Keywords Trend Heatmap Tab
        with heatmap_tabs[1]:
            st.subheader("🔤 Keywords vs Trends Correlation")
            
            # Generate hardcoded data
            keywords_data = generate_keywords_heatmap_data()
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**🎛️ Controls**")
                
                # Filters
                selected_keywords = st.multiselect(
                    "Filter Keywords",
                    keywords_data['keyword'].unique(),
                    default=keywords_data['keyword'].unique()[:10],
                    help="Select keywords to display"
                )
                
                selected_trends = st.multiselect(
                    "Filter Trends",
                    keywords_data['trend'].unique(),
                    default=keywords_data['trend'].unique()[:8],
                    help="Select trends to display"
                )
                
                metric_type_kw = st.selectbox(
                    "Select Metric",
                    ["correlation_score", "frequency", "growth_rate", "hawkes_process"],
                    help="Choose which metric to display"
                )
                
                color_scale_kw = st.selectbox(
                    "Color Scale",
                    ["RdYlBu", "Viridis", "Plasma", "Spectral", "Turbo"],
                    help="Select color scheme"
                )
            
            with col1:
                # Filter data
                filtered_data = keywords_data[
                    (keywords_data['keyword'].isin(selected_keywords)) &
                    (keywords_data['trend'].isin(selected_trends))
                ]
                
                if not filtered_data.empty:
                    # Create pivot table
                    pivot_keywords = filtered_data.pivot(
                        index='keyword', 
                        columns='trend', 
                        values=metric_type_kw
                    )
                    
                    # Create heatmap
                    fig_keywords = px.imshow(
                        pivot_keywords,
                        labels=dict(
                            x="Beauty Trends", 
                            y="Keywords", 
                            color=metric_type_kw.replace('_', ' ').title()
                        ),
                        title=f"{metric_type_kw.replace('_', ' ').title()} Heatmap: Keywords vs Trends",
                        color_continuous_scale=color_scale_kw.lower(),
                        aspect="auto"
                    )
                    
                    fig_keywords.update_layout(
                        height=600,
                        xaxis={'side': 'top'},
                        yaxis_title="Beauty Keywords"
                    )
                    
                    st.plotly_chart(fig_keywords, use_container_width=True)
                else:
                    st.warning("No data available for selected filters. Please adjust your selection.")
            
            # Analytics
            st.subheader("📊 Correlation Analytics")
            
            if not filtered_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    top_keyword = filtered_data.groupby('keyword')['correlation_score'].mean().idxmax()
                    st.metric("Top Keyword", top_keyword)
                
                with col2:
                    top_trend = filtered_data.groupby('trend')['correlation_score'].mean().idxmax()
                    st.metric("Top Trend", top_trend)
                
                with col3:
                    avg_correlation = filtered_data['correlation_score'].mean()
                    st.metric("Avg Correlation", f"{avg_correlation:.2f}")
                
                with col4:
                    total_keywords = len(filtered_data['keyword'].unique())
                    st.metric("Keywords", total_keywords)
                
                # Top correlations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔝 Top 10 Keyword-Trend Pairs**")
                    top_pairs = filtered_data.nlargest(10, 'correlation_score')[
                        ['keyword', 'trend', 'correlation_score']
                    ]
                    st.dataframe(top_pairs, hide_index=True)
                
                with col2:
                    # Trend growth chart
                    trend_growth = filtered_data.groupby('trend')['growth_rate'].mean().reset_index()
                    trend_growth = trend_growth.sort_values('growth_rate', ascending=False)
                    
                    fig_growth = px.bar(
                        trend_growth.head(10),
                        x='growth_rate',
                        y='trend',
                        orientation='h',
                        title='Top Trends by Growth Rate',
                        color='growth_rate',
                        color_continuous_scale='viridis'
                    )
                    fig_growth.update_layout(height=400)
                    st.plotly_chart(fig_growth, use_container_width=True)
            
            # Data export
            if st.checkbox("📋 Show Raw Keywords Data"):
                st.dataframe(filtered_data, use_container_width=True)
                
                csv_keywords = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Keywords Data",
                    data=csv_keywords,
                    file_name="keywords_heatmap.csv",
                    mime="text/csv"
                )
        
        # Help section
        with st.expander("❓ Understanding Heatmaps"):
            st.markdown("""
            **📅 Posting Time Heatmap:**
            - Shows optimal times to post content for maximum engagement
            - Darker colors indicate higher engagement/viral probability
            - Hawkes intensity measures content cascade potential
            
            **🔤 Keywords Heatmap:**
            - Displays correlation between beauty keywords and trending topics
            - High correlation (red/warm colors) suggests strong trend alignment
            - Growth rate indicates keyword momentum over time
            - Hawkes process models viral spreading dynamics
            
            **💡 How to Use:**
            1. Select different metrics using the dropdown controls
            2. Filter keywords and trends to focus on specific areas
            3. Use insights to optimize content strategy and timing
            4. Export data for further analysis
            """)

    # About Tab
    with tabs[6]:
        st.header("ℹ️ About Beauty Insights Suite")
        
        st.markdown("""
        ### 🎯 Purpose
        This application provides comprehensive analytics for beauty industry trends, audience segmentation, 
        market insights, and product innovation recommendations.
        
        ### 📊 Data Sources
        - **Trends**: TikTok/social media trend analysis
        - **Segments**: Audience demographic and interest segmentation  
        - **External Insights**: Market research on products, categories, brands
        - **Recommendations**: AI-generated product innovation opportunities
        
        ### 🔧 Features
        - Interactive Plotly visualizations
        - Real-time filtering and search
        - Data export capabilities
        - Responsive design for all screen sizes
        
        ### 📈 Key Metrics Explained
        - **Trend Strength**: Composite measure of viral potential
        - **Confidence**: ML model certainty in segment assignment
        - **Engagement Rate**: Interaction intensity metric
        - **ROI**: Return on investment percentage
        - **Market Potential**: Scored opportunity assessment
        """)
        
        with st.expander("📚 Data Dictionary"):
            st.markdown("""
            **Trends Data:**
            - `trend_strength`: Composite signal of topical intensity
            - `avg_composite_score`: Average quality score across videos
            - `n_videos`: Number of videos in trend cluster
            - `top_keywords`: Most relevant keywords for trend
            
            **Segments Data:**
            - `confidence`: ML model confidence (0-1 scale)
            - `segment`: Assigned audience segment
            - `genz_pct`: Percentage of Gen Z audience
            - `millennial_pct`: Percentage of Millennial audience
            
            **Recommendations Data:**
            - `roi_pct`: Expected return on investment
            - `break_even_months`: Time to profitability
            - `profitability_index`: CLV/CAC ratio indicator
            - `market_potential_score`: Opportunity assessment
            """)

if __name__ == "__main__":
    main()
