# 🎨 Beauty Insights Suite

A comprehensive Streamlit application for beauty industry analytics, featuring trend analysis, audience segmentation, market insights, and AI-powered product recommendations.

## 🚀 Features

- **📈 Trendspotting**: Analyze viral beauty trends from social media data
- **👥 Audience Segmentation**: Understand demographic patterns and engagement
- **🔍 External Insights**: Market analysis across products, categories, and brands
- **💡 Product Recommendations**: AI-generated innovation opportunities with financial forecasts
- **📊 Interactive Visualizations**: Dynamic charts with Plotly
- **🔍 Advanced Filtering**: Real-time search and filter capabilities
- **📥 Data Export**: Download filtered datasets as CSV

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd beauty-insights-suite
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place all CSV files in the `data/` directory
   - Ensure all required files are present (see Data Requirements below)

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 📁 Data Requirements

The application expects the following CSV files in the data directory:

### Required Files:
- `top_trends_clean.csv` - Social media trend analysis
- `segments_labels.csv` - Comment-level audience segments
- `segments_video.csv` - Video-level segment distributions
- `product_gaps.csv` - Market gap analysis
- `top_categories.csv` - Product category insights
- `successful_products.csv` - High-performing product data
- `top_supply_types.csv` - Supply chain analysis
- `top_brands.csv` - Brand performance metrics
- `trending_ingredients.csv` - Ingredient trend data
- `beauty_innovation_recommendation.csv` - AI product recommendations

### Data Schema

#### `top_trends_clean.csv`
```
trend_id, trend_name, n_keywords, n_videos, total_composite_score, 
avg_composite_score, max_composite_score, avg_keyword_score, 
trend_strength, top_keywords, all_keywords, rank
```

#### `segments_labels.csv`
```
commentId, videoId, segment_id, segment, confidence
```

#### `segments_video.csv`
```
videoId, total_comments, genz_pct, millennial_pct, 
interest_pct, other_pct, top_interest
```

#### `beauty_innovation_recommendation.csv`
```
product_name, category, key_ingredients, target_market, 
innovation_description, forecasted_yearly_revenue, 
forecasted_margin_pct, forecasted_yearly_profit, 
forecast_confidence, model_version, break_even_months, 
roi_pct, market_potential_score, risk_level, 
investment_recommendation, [additional context fields...]
```

## 🎯 Usage Guide

### Navigation
The app features five main sections accessible via tabs:

1. **📈 Trends**: Analyze viral beauty trends and their performance metrics
2. **👥 Audience Segments**: Explore demographic patterns and engagement
3. **🔍 External Insights**: Market analysis across multiple dimensions
4. **💡 Recommendations**: AI-powered product innovation pipeline
5. **ℹ️ About**: Documentation and data dictionary

### Key Features

#### Global Controls (Sidebar)
- **Data Directory**: Specify path to your CSV files
- **Top N Slider**: Control number of items shown in charts (5-50)
- **Search**: Filter current page data
- **Toggle Tables**: Show/hide data tables below charts
- **Reset Filters**: Clear all applied filters

#### Trend Analysis
- Filter by minimum videos, trend strength, or keywords
- Interactive charts showing trend performance matrices
- Detailed keyword analysis with visual tags
- Export capabilities for trend data

#### Audience Segmentation
- Comment-level and video-level analysis tabs
- Confidence distribution analysis
- Demographic composition visualizations
- Weighted audience metrics

#### External Insights
- Multi-tab interface for different data sources
- Product gap analysis with engagement metrics
- Category performance comparisons
- Brand and ingredient trend analysis

#### Product Recommendations
- Card-based product display with risk indicators
- Financial forecasting with ROI analysis
- Advanced filtering by category, risk, and revenue
- Short-listing capabilities with export options

### Performance Features
- **Caching**: Automatic data caching for fast load times
- **Error Handling**: Graceful handling of missing files
- **Responsive Design**: Works on desktop and tablet devices
- **Export Options**: Download filtered data as CSV

## 🔧 Configuration

### Data Directory
By default, the app looks for data in:
```
/Users/jeffwg/Documents/Competition/DATATHON/L'OREAL/UI/data
```

You can change this path using the sidebar input field.

### Customization
The app includes custom CSS styling that can be modified in the `app.py` file:
- Metric cards styling
- Product cards with risk indicators
- Trend tags and badges
- Color schemes for risk levels

## 📊 Key Metrics Explained

### Trend Metrics
- **Trend Strength**: Composite measure of viral potential and engagement
- **Composite Score**: Content quality assessment
- **Keyword Score**: Relevance and trending potential of associated keywords

### Audience Metrics
- **Confidence**: ML model certainty in segment assignment (0-1 scale)
- **Engagement**: Interaction rates across different demographics
- **Distribution**: Percentage breakdown of audience segments

### Financial Metrics
- **ROI**: Return on Investment percentage
- **Break-even**: Months until profitability
- **Market Potential**: Scored opportunity assessment (Low/Medium/High)
- **Profitability Index**: CLV/CAC ratio indicator

## 🐛 Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Ensure all CSV files are in the correct directory
   - Check file names match exactly (case-sensitive)
   - Verify data directory path in sidebar

2. **Empty Charts**
   - Check if filters are too restrictive
   - Use "Reset Filters" button to clear all filters
   - Verify data contains required columns

3. **Performance Issues**
   - Reduce "Top N" slider value for faster rendering
   - Clear browser cache if charts don't update
   - Check data file sizes (app optimized for <1M rows per file)

4. **Display Issues**
   - Refresh the page if layouts appear broken
   - Try different browser (Chrome/Firefox recommended)
   - Ensure screen width is sufficient for wide layout

### Data Issues

- **Missing Columns**: App will show warnings and skip affected visualizations
- **Data Types**: Ensure numeric columns don't contain text values
- **Encoding**: Save CSV files with UTF-8 encoding for special characters

## 🔄 Updates and Maintenance

### Adding New Data
1. Place new CSV files in the data directory
2. Refresh the app (it will auto-detect new files)
3. Check sidebar for data loading status

### Modifying Filters
The app automatically detects available columns and creates appropriate filters. No code changes needed for standard data updates.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your data matches the expected schema
3. Ensure all dependencies are correctly installed

## 🎨 Customization

The app is designed to be easily customizable:
- Modify chart types in the respective `create_*_charts()` functions
- Add new KPIs in the `display_kpis()` function
- Customize styling in the CSS section at the top of `app.py`
- Add new filter options by modifying the filter sections

---

**Built with ❤️ using Streamlit, Plotly, and Pandas**
