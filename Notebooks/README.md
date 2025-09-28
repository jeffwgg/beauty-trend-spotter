# TrendSpotter: Advanced Beauty Content Analytics Pipeline

## Project Overview

TrendSpotter is a comprehensive data science framework that combines **quantitative finance methodologies** with **social media analytics** to detect emerging trends, predict viral content, and segment audiences in the beauty industry. The system processes 4.7M YouTube comments across 92K videos using quantitative finance methodologies adapted for social media analytics.

### Core Technical Objectives
- **Self-Exciting Process Detection**: Hawkes process modeling of comment cascades with exponential decay kernels
- **Topological Novelty Detection**: PCA-based burst identification in reduced dimensional comment activity space
- **Probabilistic Audience Clustering**: GMM segmentation using engineered behavioral features (slang patterns, emoji usage, temporal activity)
- **Multi-Modal Revenue Forecasting**: Random Forest ensemble with business metrics integration

### System Architecture
The pipeline implements a **4-stage signal processing architecture**:

```
Stage 1: Data Preprocessing (text normalization, temporal aggregation)
    ↓
Stage 2: Parallel Signal Generation (Hawkes, TBI, Fundamentals, Decay)
    ↓
Stage 3: Multi-Signal Fusion (weighted composite scoring)
    ↓
Stage 4: Predictive Modeling (trend classification, revenue forecasting)
```

## Folder Structure

### Core Pipeline Files
- **`data-cleaning.ipynb`** - Data ingestion, text normalization, and feature engineering
- **`aggregate-to-video-hourly-activity.ipynb`** - Time-series aggregation for temporal analysis
- **`apply-hawkes-process-for-self-exciting-momentum.ipynb`** - Hawkes process momentum detection
- **`tbi_novelty_detection.py`** - Topological Burst Index for novelty detection
- **`fundamental-analysis-comments-likes-tags.ipynb`** - Content quality metrics
- **`trend-decay-analysis-multi-signal-combined.ipynb`** - Trend lifecycle analysis
- **`fusion-engine.ipynb`** - Multi-signal integration and composite scoring
- **`gmm-audience-segmentation.ipynb`** - Gaussian Mixture Model audience clustering
- **`cluster-into-trends.ipynb`** - Semantic trend clustering
- **`video-trend-predictor.ipynb`** - ML models for trend prediction
- **`merge-signals-to-master-csv.ipynb`** - Final data consolidation
- **`innovative-recommendation-engine.ipynb`** - Product recommendation system
- **`external-dataset-analysis.ipynb`** - Market intelligence integration

### Supporting Files
- **`system architecture.drawio.png`** - Visual system architecture diagram
- **`system architecture.drawio.html`** - Interactive architecture diagram

## Setup Instructions

### Dependencies
```bash
pip install pandas numpy scikit-learn
pip install hawkeslib sentence-transformers
pip install giotto-tda polars lightgbm xgboost
pip install prophet cmdstanpy
pip install ftfy langdetect emoji
pip install keybert yake hdbscan
pip install matplotlib seaborn plotly
```

### Environment
- **Python**: 3.11+
- **Memory**: 16GB+ recommended for full dataset processing
- **Storage**: 10GB+ for intermediate files

### Input Data Structure
```
/input/
├── datathon-loreal/
│   ├── videos.csv          # Video metadata
│   ├── comments1-5.csv     # Comment data (split files)
├── amazon-ratings/         # External market data
├── supply-chain-analysis/  # Supply chain metrics
└── most-used-beauty-cosmetics-products/
```

## Data Flow & Pipeline

### Stage 1: Data Preparation
```
Raw Comments + Videos → data-cleaning.ipynb → comments_enriched.parquet
                     ↓
            aggregate-to-video-hourly-activity.ipynb → video_hourly_activity.parquet
```

### Stage 2: Signal Generation
```
video_hourly_activity.parquet → [Parallel Processing]
                             ├── apply-hawkes-process → signal_hawkes.csv
                             ├── tbi_novelty_detection.py → tbi_novelty_bursts.csv
                             ├── fundamental-analysis → signal_fundamentals.csv
                             └── trend-decay-analysis → signal_decay.csv
```

### Stage 3: Fusion & Intelligence
```
All Signals → fusion-engine.ipynb → all_signals_combined.csv
           ↓
comments_enriched.parquet → gmm-audience-segmentation.ipynb → segments_labels.csv
                         ↓
all_signals_combined.csv → video-trend-predictor.ipynb → prediction_models.pkl
                        ↓
                    merge-signals-to-master-csv.ipynb → master.csv
```

### Stage 4: Business Applications
```
master.csv + External Data → innovative-recommendation-engine.ipynb → product_recommendations.csv
                           ↓
                       cluster-into-trends.ipynb → trend_clusters.csv
```

## Technical Implementation Details

### 1. Data Preprocessing Pipeline (`data-cleaning.ipynb`)
**Text Normalization**:
```python
def clean_text(s: pd.Series) -> pd.Series:
    s = s.str.replace(r"http\S+|www\.\S+", "", regex=True)  # URL removal
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()  # Whitespace normalization
    return s
```

**Feature Engineering**:
- **Temporal Features**: Hour-of-day, day-of-week, week-start alignment
- **Linguistic Features**: Emoji count, hashtag extraction, language detection (langdetect)
- **Engagement Features**: Like-to-comment ratios, commenter uniqueness

**Output**: `comments_enriched.parquet` (4.7M rows, 17 features)

### 2. Hawkes Process Implementation (`apply-hawkes-process-for-self-exciting-momentum.ipynb`)
**Mathematical Foundation**:
The intensity function models self-exciting comment cascades:

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}$$

Where:
- $\mu$ = baseline comment rate (background intensity)
- $\alpha$ = excitation strength (viral amplification factor)
- $\beta$ = decay rate (attention span parameter)
- $t_i$ = timestamps of previous comments

**Implementation Details**:
```python
# Momentum calculation with engineered features
momentum_components = {
    "growth_rate": max(0, -decay_rate),  # Negative decay = positive growth
    "momentum_ratio": late_activity / early_activity,  # Acceleration indicator
    "recent_momentum": recent_24h / previous_24h,  # Short-term momentum
    "acceleration": (q4_velocity - q1_velocity) / q1_velocity,  # Velocity change
    "velocity_normalized": velocity / log(duration_hours + 1)  # Time-normalized rate
}

# Weighted composite score
composite_momentum = sum(weights[k] * momentum_components[k] for k in weights.keys())
```

**Robust Scaling**: 5th-95th percentile clipping with min-max normalization to [0,10]

**Output**: `signal_hawkes.csv` (5,383 videos with momentum scores)

### 3. Topological Burst Index (`tbi_novelty_detection.py`)
**Algorithm**: PCA-based novelty detection in comment activity space

**Mathematical Framework**:
1. **Dimensionality Reduction**: PCA to 5 components
   $$X_{reduced} = PCA(X_{scaled})$$

2. **Local Topology Analysis**: Sliding window (7-hour) pairwise distances
   $$d_{ij} = ||x_i - x_j||_2$$

3. **TBI Calculation**: Normalized deviation from local centroid
   $$TBI_t = \frac{||x_t - \bar{x}_{local}|| - \mu_{dist}}{\sigma_{dist} + \epsilon}$$

4. **Novelty Detection**: Z-score thresholding (threshold = 2.0)
   $$\text{Novelty} = |zscore(TBI)| > 2.0 \land TBI > 1.5$$

**Implementation**:
```python
def compute_tbi(self, data_matrix):
    for i in range(1, n_points - 1):
        local_data = data_matrix[start_idx:end_idx]
        distances = pdist(local_data, metric='euclidean')
        
        # Topological complexity measure
        centroid_distance = np.linalg.norm(current_point - local_centroid)
        tbi = (centroid_distance - mean_distance) / (std_distance + 1e-8)
        
        # Weight by local complexity
        complexity = std_distance / (max_distance + 1e-8)
        tbi_values[i] = abs(tbi) * (1 + complexity)
```

**Output**: `tbi_novelty_bursts.csv` (5,387 videos with novelty metrics)

### 4. Fundamental Analysis (`fundamental-analysis-comments-likes-tags.ipynb`)
**Quality Metrics Computation**:

$$\text{Engagement Quality} = \min\left(\frac{\text{likes per comment}}{10}, 1\right)$$

$$\text{Depth Score} = \min\left(\frac{\text{unique commenters}}{\text{total comments}}, 1\right)$$

$$\text{Content Richness} = 0.4 \cdot \frac{\text{avg emojis}}{5} + 0.3 \cdot \frac{\text{avg hashtags}}{3} + 0.3 \cdot \frac{\text{avg text length}}{200}$$

$$\text{Fundamental Health} = 0.4 \cdot \text{Engagement} + 0.3 \cdot \text{Depth} + 0.3 \cdot \text{Richness}$$

**Saturation Analysis**:
$$\text{Saturation} = \frac{\text{total comments}}{\text{viewCount}/1000 + 1}$$

**Output**: `signal_fundamentals.csv` (39,938 videos with quality scores)

### 5. Trend Decay Analysis (`trend-decay-analysis-multi-signal-combined.ipynb`)
**Exponential Decay Modeling**:

$$y(t) = y_0 e^{-\lambda t}$$

Log-linear regression for decay parameter estimation:
$$\ln(y) = \ln(y_0) - \lambda t$$

**Half-life Calculation**:
$$t_{1/2} = \frac{\ln(2)}{\lambda}$$

**Decay Strength Score**:
$$\text{Strength} = \frac{(-\text{slope}) \cdot R^2}{\text{normalization factor}}$$

**Implementation**:
```python
def fit_exp_decay(post_df, value_col, peak_ts, eps=1e-6):
    t_hours = (post_df["ts"] - peak_ts).dt.total_seconds().values / 3600.0
    y = post_df[value_col].values + eps
    y_log = np.log(y)
    
    reg = LinearRegression().fit(t_hours.reshape(-1, 1), y_log)
    slope = float(reg.coef_[0])
    r2 = float(reg.score(t_hours.reshape(-1, 1), y_log))
    half_life = math.log(2) / -slope if slope < 0 else math.inf
    
    return slope, r2, half_life
```

**Output**: `signal_decay.csv` (39,938 videos with decay metrics)

### 6. Multi-Signal Fusion Engine (`fusion-engine.ipynb`)
**Signal Processing Architecture**:

```python
class SignalProcessor:
    def safe_normalize(self, series):
        # Min-Max normalization with outlier handling
        if series.std() == 0: return pd.Series([0.5] * len(series))
        return (series - series.min()) / (series.max() - series.min())
```

**Fusion Formula**:
$$\text{Composite Score} = \sum_{i=1}^{4} w_i \cdot \text{normalize}(S_i)$$

Where:
- $w_1 = 0.3$ (Hawkes weight)
- $w_2 = 0.25$ (TBI weight) 
- $w_3 = 0.25$ (Fundamental weight)
- $w_4 = 0.2$ (Decay weight)

**Performance Categories**:
- **Low**: Score < 0.25
- **Medium**: 0.25 ≤ Score < 0.5
- **High**: 0.5 ≤ Score < 0.75
- **Viral**: Score ≥ 0.75

**Output**: `all_signals_combined.csv` (39,938 videos with composite scores)

### 7. GMM Audience Segmentation (`gmm-audience-segmentation.ipynb`)
**Feature Engineering for Behavioral Clustering**:

```python
# Temporal behavior encoding
features['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
features['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Slang detection with regex patterns
slang_terms = ['bussin', 'rizz', 'cap', 'no cap', 'gyat', 'sigma', 'skibidi', ...]
df['slang_count'] = df['text_norm'].str.lower().apply(
    lambda text: sum(1 for term in slang_terms if term in text)
)

# Emoji density calculation
features['emoji_per_word'] = df['emoji_count'] / word_counts.replace(0, 1)
```

**GMM Implementation**:
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

**BIC-based Model Selection**:
$$BIC = -2 \ln(L) + k \ln(n)$$

Optimal components: K=4 (lowest BIC)

**Segment Assignment Logic**:
```python
# Gen Z: High slang, late night activity, low emoji density
genz_score = slang_count + night_activity_bonus - emoji_per_word

# Millennial: Daytime activity, moderate engagement
millennial_score = daytime_bonus + moderate_slang_bonus + emoji_bonus
```

**Output**: `segments_labels.csv` (4.7M comments with segment assignments)

### 8. Revenue Forecasting (`innovative-recommendation-engine.ipynb`)
**ML Pipeline Architecture**:

```python
# Feature engineering for revenue prediction
training_data['rating_review_interaction'] = rating * review_count
training_data['log_review_count'] = np.log1p(review_count)
training_data['revenue_per_review'] = revenue / (review_count + 1)
training_data['brand_category_interaction'] = brand_factor * category_factor
```

**Random Forest with Grid Search**:
```python
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

**Business Metrics Calculation**:
$$ROI = \frac{\text{Forecasted Annual Profit}}{\text{Fixed Cost}} \times 100$$

$$\text{Break-even Months} = \frac{\text{Fixed Cost}}{\text{Monthly Profit}}$$

$$\text{Profitability Index} = \text{Margin} \times \frac{ROI}{100}$$

**Output**: Product recommendations with financial projections

### 9. Predictive Modeling (`video-trend-predictor.ipynb`)
**Multi-target Classification/Regression**:

1. **Emerging Trend Classification** (Logistic Regression):
   - Target: Binary (top 20th percentile composite score)
   - Features: TF-IDF(title+description+tags) + metadata
   - Solver: SAGA with L2 regularization

2. **Trend Stage Classification** (Multinomial Logistic):
   - Targets: Rising/Peaking/Decaying
   - Logic: Momentum vs Decay signal comparison

3. **Volume Regression** (SGD with log transformation):
   - Target: log1p(predicted_future_score)
   - Inverse transform: expm1 for final predictions

**Pipeline Implementation**:
```python
ct = ColumnTransformer([
    ("text", TfidfVectorizer(max_features=12000, ngram_range=(1,2)), "text_all"),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler(with_mean=False))]), NUM_COLS)
])
```

## Usage Guide

### Step 1: Data Preparation
```python
# Run data cleaning
jupyter notebook data-cleaning.ipynb

# Aggregate to hourly activity
jupyter notebook aggregate-to-video-hourly-activity.ipynb
```

### Step 2: Signal Generation (Parallel)
```python
# Generate all signals simultaneously
jupyter notebook apply-hawkes-process-for-self-exciting-momentum.ipynb &
python tbi_novelty_detection.py &
jupyter notebook fundamental-analysis-comments-likes-tags.ipynb &
jupyter notebook trend-decay-analysis-multi-signal-combined.ipynb &
wait
```

### Step 3: Fusion & Analysis
```python
# Combine signals
jupyter notebook fusion-engine.ipynb

# Audience segmentation
jupyter notebook gmm-audience-segmentation.ipynb

# Train predictive models
jupyter notebook video-trend-predictor.ipynb
```

### Step 4: Business Intelligence
```python
# Generate final master dataset
jupyter notebook merge-signals-to-master-csv.ipynb

# Product recommendations
jupyter notebook innovative-recommendation-engine.ipynb

# Trend clustering
jupyter notebook cluster-into-trends.ipynb
```

### Expected Outputs
- **Performance Scores**: 0-1 composite scores for viral potential
- **Audience Segments**: Probabilistic user classifications
- **Trend Predictions**: Binary emerging/declining classifications
- **Revenue Forecasts**: Dollar projections with confidence intervals

## Business & Analytical Value

### Quantitative Finance Integration
The system applies **signal processing techniques** from quantitative finance:
- **Multi-signal fusion** for robust trend detection
- **Momentum indicators** adapted from technical analysis
- **Risk-adjusted scoring** for investment decisions

### Key Business Applications
1. **Early Trend Detection**: Identify emerging beauty trends 2-4 weeks before mainstream adoption
2. **Content Optimization**: Predict which video concepts will achieve viral status
3. **Audience Targeting**: Segment users for personalized marketing campaigns
4. **Product Development**: Forecast market demand for new beauty products
5. **Investment Decisions**: ROI analysis for content and product investments

### Competitive Advantages
- **Multi-modal Analysis**: Combines engagement, semantic, and temporal signals
- **Real-time Processing**: Hourly updates for rapid trend detection
- **Probabilistic Outputs**: Confidence intervals for risk management
- **Scalable Architecture**: Modular design for easy extension

## Future Extensions

### Technical Improvements
- **Multi-platform Support**: Extend to TikTok, Instagram, Twitter
- **Real-time Streaming**: Apache Kafka integration for live processing
- **Deep Learning**: Transformer models for semantic understanding
- **Graph Analytics**: Social network analysis for influence mapping

### Business Enhancements
- **Competitive Intelligence**: Brand comparison and market positioning
- **Supply Chain Integration**: Inventory optimization based on trend predictions
- **Creator Economy**: Influencer ROI analysis and partnership recommendations
- **Global Expansion**: Multi-language and cultural adaptation

### Scalability
- **Cloud Deployment**: AWS/GCP containerization
- **Distributed Computing**: Spark/Dask for large-scale processing
- **Model Serving**: MLflow/Kubeflow for production ML pipelines
- **API Development**: REST/GraphQL interfaces for business applications

---

*TrendSpotter represents the convergence of quantitative finance rigor with social media analytics, providing beauty brands and content creators with institutional-grade intelligence for trend detection and audience understanding.*