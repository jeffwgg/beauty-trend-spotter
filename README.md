# 🎨 Beauty Insights Suite

A comprehensive Streamlit application for beauty industry analytics, featuring trend analysis, audience segmentation, market insights, and AI-powered product recommendations.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Run
```bash
# Clone the repository
git clone <repository-url>
cd datathonlorealFrontEnd

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🌐 Web3 Storage Integration

This application uses **Storacha (Web3.Storage)** for decentralized data storage:
- **Global Access**: Data accessible from anywhere via IPFS
- **Fallback System**: Automatic fallback to local files if Web3 fails
- **Sample Data**: Demo mode for testing without real data

## 📊 Features

- **📈 Trendspotting**: Analyze viral beauty trends from social media data
- **👥 Audience Segmentation**: Understand demographic patterns and engagement
- **🔍 External Insights**: Market analysis across products, categories, and brands
- **💡 Product Recommendations**: AI-generated innovation opportunities with financial forecasts
- **📺 YouTube Predictor**: Viral potential analysis for beauty content
- **🔥 Heatmap Analytics**: Posting time optimization and keyword correlation analysis

## 📁 Project Structure

```
datathonlorealFrontEnd/
├── app.py                      # Main Streamlit application
├── web3_data_manager.py        # Web3/IPFS data handling
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/                 # Streamlit configuration
│   ├── config.toml
│   └── secrets.toml.example
├── data/                       # Local CSV files (backup)
├── model/                      # ML model files
├── docs/                       # Documentation
│   ├── APP_SUMMARY.md          # Competition presentation summary
│   └── TECHNICAL_OVERVIEW.md   # Technical documentation
└── UI/                         # Legacy folder (can be removed)
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Storage**: Web3.Storage (IPFS) + Local fallback
- **APIs**: YouTube Data API v3
- **ML Models**: Scikit-learn

## 📖 Documentation

- **[App Summary](docs/APP_SUMMARY.md)**: Competition presentation and business overview
- **[Technical Overview](docs/TECHNICAL_OVERVIEW.md)**: Detailed technical documentation

## 🌍 Deployment

### Streamlit Community Cloud
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Set main file path: `app.py`
4. Deploy automatically

### Local Development
```bash
streamlit run app.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
IPFS_CID=your_ipfs_cid_here
IPFS_GATEWAY=https://w3s.link/ipfs
```

### Data Sources
The app supports three data source modes:
- **🌐 Web3 Storage**: Global IPFS storage (default)
- **💾 Local Files**: Bundled CSV files (backup)
- **🎯 Sample Data**: Demo data for testing

## 📊 Data Requirements

The application expects these CSV files:
- `beauty_innovation_recommendation.csv`
- `product_gaps.csv`
- `segments_labels.csv`
- `segments_video.csv`
- `successful_products.csv`
- `top_brands.csv`
- `top_categories.csv`
- `top_supply_types.csv`
- `refined_trends.csv`
- `trending_ingredients.csv`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is part of the L'Oréal Datathon competition.

---

**Built with ❤️ using Streamlit, Web3.Storage, and modern data science tools**