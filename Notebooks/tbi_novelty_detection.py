"""
Topological Burst Index (TBI) for Novelty Detection in Comment Activity

This script uses TBI to detect novelty and unusual patterns in video comment data
by analyzing topological features in reduced-dimensional space. TBI captures
sudden changes in data topology that indicate novel or burst-like behavior.

Features:
- Fast PCA/SVD dimensionality reduction
- Topological burst index calculation
- Novelty detection based on topological changes
- Efficient processing of large datasets

Author: Data Science Team
Purpose: Detect novel engagement patterns and topological bursts
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CONFIG = {
    'input_file': "/kaggle/input/aggregate-to-video-hourly-activity/video_hourly_activity.parquet",
    'output_file': "tbi_novelty_bursts.csv",
    'min_time_points': 7,        # Minimum time series length
    'min_total_activity': 10,    # Minimum total activity
    'dimension_reduction': 'pca', # 'pca', 'svd', or 'none'
    'n_components': 5,           # Reduced dimensions
    'window_size': 7,            # Sliding window for TBI calculation
    'novelty_threshold': 2.0,    # Z-score threshold for novelty
    'burst_threshold': 1.5,      # TBI threshold for burst detection
    'use_features': ['comment_count', 'unique_commenters', 'avg_emoji_per_comment', 'hashtag_density']
}

class TopologicalBurstDetector:
    """
    Fast TBI-based novelty and burst detection system.
    """
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.reducer = None
        self._setup_dimensionality_reduction()
    
    def _setup_dimensionality_reduction(self):
        """Initialize dimensionality reduction method."""
        if self.config['dimension_reduction'] == 'pca':
            self.reducer = PCA(n_components=self.config['n_components'], random_state=42)
        elif self.config['dimension_reduction'] == 'svd':
            self.reducer = TruncatedSVD(n_components=self.config['n_components'], random_state=42)
        else:
            self.reducer = None
    
    def compute_tbi(self, data_matrix):
        """
        Compute Topological Burst Index for a data matrix.
        
        Args:
            data_matrix: 2D array (time_points x features)
            
        Returns:
            Array of TBI values for each time point
        """
        n_points = len(data_matrix)
        if n_points < 3:
            return np.zeros(n_points)
        
        tbi_values = np.zeros(n_points)
        
        for i in range(1, n_points - 1):
            # Define local neighborhood
            start_idx = max(0, i - self.config['window_size'] // 2)
            end_idx = min(n_points, i + self.config['window_size'] // 2 + 1)
            
            local_data = data_matrix[start_idx:end_idx]
            
            if len(local_data) < 3:
                continue
            
            # Compute pairwise distances
            distances = pdist(local_data, metric='euclidean')
            
            if len(distances) == 0:
                continue
            
            # Topological features
            max_distance = np.max(distances)
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # Current point distance to local centroid
            local_centroid = np.mean(local_data, axis=0)
            current_point = data_matrix[i]
            centroid_distance = np.linalg.norm(current_point - local_centroid)
            
            # TBI calculation: normalized deviation from local topology
            if std_distance > 0:
                tbi = (centroid_distance - mean_distance) / (std_distance + 1e-8)
            else:
                tbi = 0
            
            # Additional topological complexity measure
            if max_distance > 0:
                complexity = std_distance / (max_distance + 1e-8)
                tbi *= (1 + complexity)  # Weight by local complexity
            
            tbi_values[i] = abs(tbi)  # Take absolute value for burst magnitude
        
        return tbi_values
    
    def detect_novelty_points(self, tbi_values):
        """
        Detect novelty points based on TBI values.
        
        Args:
            tbi_values: Array of TBI values
            
        Returns:
            Boolean array indicating novelty points
        """
        if len(tbi_values) < 3:
            return np.zeros(len(tbi_values), dtype=bool)
        
        # Z-score based novelty detection
        tbi_zscore = np.abs(zscore(tbi_values, nan_policy='omit'))
        novelty_mask = tbi_zscore > self.config['novelty_threshold']
        
        # Additional condition: TBI above burst threshold
        burst_mask = tbi_values > self.config['burst_threshold']
        
        return novelty_mask & burst_mask
    
    def process_video_data(self, video_data):
        """
        Process a single video's time series data for TBI analysis.
        
        Args:
            video_data: DataFrame with time series data for one video
            
        Returns:
            Dict with TBI results or None if insufficient data
        """
        try:
            # Sort by datetime
            video_data = video_data.sort_values('datetime')
            
            # Check if we have enough data points
            if len(video_data) < self.config['min_time_points']:
                return None
            
            # Check total activity threshold
            total_activity = video_data[self.config['use_features'][0]].sum()
            if total_activity < self.config['min_total_activity']:
                return None
            
            # Extract feature matrix
            feature_matrix = video_data[self.config['use_features']].values
            
            # Handle missing values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0, posinf=0, neginf=0)
            
            # Standardize features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Apply dimensionality reduction if specified and feasible
            if self.reducer is not None:
                n_samples, n_features = feature_matrix_scaled.shape
                max_components = min(n_samples, n_features)
                
                # Adaptive component selection
                if max_components < self.config['n_components']:
                    # Use fewer components or skip reduction if data is too small
                    if max_components >= 2:  # Need at least 2 components for meaningful reduction
                        # Create a new reducer with appropriate number of components
                        if self.config['dimension_reduction'] == 'pca':
                            from sklearn.decomposition import PCA
                            temp_reducer = PCA(n_components=max_components - 1, random_state=42)
                        else:  # svd
                            from sklearn.decomposition import TruncatedSVD
                            temp_reducer = TruncatedSVD(n_components=max_components - 1, random_state=42)
                        feature_matrix_reduced = temp_reducer.fit_transform(feature_matrix_scaled)
                    else:
                        # Skip dimensionality reduction for very small datasets
                        feature_matrix_reduced = feature_matrix_scaled
                else:
                    # Use the configured reducer
                    feature_matrix_reduced = self.reducer.fit_transform(feature_matrix_scaled)
            else:
                feature_matrix_reduced = feature_matrix_scaled
            
            # Compute TBI values
            tbi_values = self.compute_tbi(feature_matrix_reduced)
            
            # Detect novelty points
            novelty_points = self.detect_novelty_points(tbi_values)
            
            if not np.any(novelty_points):
                return None  # No novelty detected
            
            # Find the strongest novelty/burst
            novelty_indices = np.where(novelty_points)[0]
            strongest_idx = novelty_indices[np.argmax(tbi_values[novelty_indices])]
            
            # Get date information
            burst_date = video_data.iloc[strongest_idx]['datetime']
            
            # Calculate novelty burst characteristics
            novelty_strength = tbi_values[strongest_idx]
            novelty_duration = len(novelty_indices)
            
            # Additional statistics
            total_comments = video_data[self.config['use_features'][0]].sum()
            peak_activity = video_data[self.config['use_features'][0]].max()
            avg_tbi = np.mean(tbi_values)
            max_tbi = np.max(tbi_values)
            
            return {
                'novelty_date': burst_date,
                'novelty_day_index': strongest_idx,
                'novelty_strength': float(novelty_strength),
                'novelty_duration': int(novelty_duration),
                'total_activity': int(total_comments),
                'peak_activity': int(peak_activity),
                'avg_tbi': float(avg_tbi),
                'max_tbi': float(max_tbi),
                'time_series_length': len(video_data),
                'novelty_ratio': float(novelty_duration / len(video_data)),
                'components_used': feature_matrix_reduced.shape[1]
            }
            
        except Exception as e:
            # Silently skip errors to avoid cluttering output
            return None

def load_and_prepare_data(input_file):
    """Load and prepare data for TBI analysis."""
    print(f"📊 Loading data from: {input_file}")
    
    df = pd.read_parquet(input_file)
    print(f"   Raw data shape: {df.shape}")
    print(f"   Available columns: {df.columns.tolist()}")
    
    # Ensure datetime is properly formatted
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    # Aggregate to daily level
    print("🔄 Aggregating to daily level...")
    daily_data = (
        df.groupby(["videoId", pd.Grouper(key="datetime", freq="D")])
        [CONFIG['use_features']]
        .sum()
        .reset_index()
    )
    
    print(f"   Daily data shape: {daily_data.shape}")
    print(f"   Unique videos: {daily_data['videoId'].nunique()}")
    
    return daily_data

def filter_videos(daily_data, min_points, min_activity):
    """Filter videos with sufficient data for analysis."""
    print("🎯 Filtering videos...")
    
    video_stats = daily_data.groupby("videoId").agg({
        CONFIG['use_features'][0]: ['count', 'sum']
    })
    
    video_stats.columns = ['time_points', 'total_activity']
    video_stats = video_stats.reset_index()
    
    valid_videos = video_stats[
        (video_stats['time_points'] >= min_points) & 
        (video_stats['total_activity'] >= min_activity)
    ]['videoId'].tolist()
    
    print(f"   Videos meeting criteria: {len(valid_videos)} / {len(video_stats)}")
    return valid_videos

def main():
    """Main execution function."""
    print("=" * 70)
    print("🌊 TOPOLOGICAL BURST INDEX (TBI) NOVELTY DETECTION")
    print("=" * 70)
    
    # Load data
    daily_data = load_and_prepare_data(CONFIG['input_file'])
    
    # Filter videos
    valid_videos = filter_videos(
        daily_data, 
        CONFIG['min_time_points'], 
        CONFIG['min_total_activity']
    )
    
    if len(valid_videos) == 0:
        print("❌ No videos meet the criteria!")
        return
    
    # Initialize TBI detector
    print(f"🧠 Initializing TBI detector...")
    print(f"   Method: {CONFIG['dimension_reduction'].upper()}")
    print(f"   Components: {CONFIG['n_components']}")
    print(f"   Window size: {CONFIG['window_size']}")
    print(f"   Features: {CONFIG['use_features']}")
    
    detector = TopologicalBurstDetector(CONFIG)
    
    # Process videos
    print(f"🔍 Processing {len(valid_videos)} videos...")
    results = []
    
    for video_id in tqdm(valid_videos, desc="TBI Analysis"):
        video_data = daily_data[daily_data['videoId'] == video_id].copy()
        
        result = detector.process_video_data(video_data)
        if result is not None:
            result['videoId'] = video_id
            results.append(result)
    
    # Save and summarize results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(CONFIG['output_file'], index=False)
        
        print(f"\n✅ SUCCESS: Detected novelty in {len(results_df)} videos")
        print(f"📁 Results saved to: {CONFIG['output_file']}")
        print(f"🎯 Detection rate: {len(results_df)/len(valid_videos)*100:.1f}%")
        
        # Summary statistics
        print("\n📈 TBI NOVELTY ANALYSIS SUMMARY")
        print("-" * 50)
        print(f"Novelty strength stats:")
        print(f"   Mean: {results_df['novelty_strength'].mean():.3f}")
        print(f"   Median: {results_df['novelty_strength'].median():.3f}")
        print(f"   Max: {results_df['novelty_strength'].max():.3f}")
        
        print(f"\nNovelty duration stats:")
        print(f"   Mean: {results_df['novelty_duration'].mean():.1f} days")
        print(f"   Median: {results_df['novelty_duration'].median():.1f} days")
        
        print("\n🏆 Top 5 Strongest Novelty Events:")
        top_novelty = results_df.nlargest(5, 'novelty_strength')[
            ['videoId', 'novelty_strength', 'novelty_duration', 'peak_activity']
        ]
        print(top_novelty.to_string(index=False))
        
    else:
        print("⚠️  No novelty patterns detected!")
        print("💡 Try adjusting thresholds or window size")
        
        # Create empty results file
        empty_df = pd.DataFrame(columns=[
            'videoId', 'novelty_date', 'novelty_day_index', 'novelty_strength',
            'novelty_duration', 'total_activity', 'peak_activity', 'avg_tbi',
            'max_tbi', 'time_series_length', 'novelty_ratio'
        ])
        empty_df.to_csv(CONFIG['output_file'], index=False)
    
    print(f"\n🎉 TBI Analysis complete!")

if __name__ == "__main__":
    main()
