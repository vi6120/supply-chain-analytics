# Supply Chain & Inventory Optimization Analytics

A comprehensive analytics dashboard for supply chain optimization featuring demand forecasting, inventory analysis, and reorder point optimization.

## Features

### 📊 Dashboard Overview
- Key Performance Indicators (KPIs)
- Inventory turnover analysis
- Stockout rate monitoring
- Delivery performance metrics
- Interactive visualizations

### 📈 Demand Forecasting
- Prophet time series forecasting
- ARIMA modeling
- Seasonal pattern detection
- Confidence intervals
- Multi-period forecasts

### 🎯 Inventory Analysis
- Real-time inventory tracking
- Demand vs inventory correlation
- Stockout frequency analysis
- Category-wise performance

### ⚡ Reorder Optimization
- Dynamic reorder point calculation
- Safety stock optimization
- Service level configuration
- Inventory simulation
- Automated reorder alerts

### 🚚 Supplier Performance
- Delivery delay tracking
- Lead time analysis
- Performance scoring
- Supplier rankings

## Quick Start

### Option 1: Direct Run (Recommended)
```bash
python supply_chain_app.py
```

### Option 2: Using Launcher
```bash
python run_app.py
```

The application will automatically:
1. Check for required dependencies
2. Install missing packages
3. Generate synthetic supply chain data
4. Launch the Streamlit dashboard at http://localhost:8501

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Forecasting**: Prophet, Statsmodels (ARIMA)
- **Analytics**: Scikit-learn

## Key Metrics Calculated

### Inventory KPIs
- **Inventory Turnover**: Cost of Goods Sold ÷ Average Inventory
- **Stockout Rate**: Percentage of days with stockouts
- **Service Level**: Probability of not stocking out
- **Carrying Cost**: Cost of holding inventory

### Reorder Calculations
- **Reorder Point (ROP)**: Demand during Lead Time + Safety Stock
- **Safety Stock**: Buffer inventory for demand variability
- **Economic Order Quantity**: Optimal order size

### Supplier Metrics
- **Delivery Performance**: On-time delivery percentage
- **Lead Time Variability**: Consistency in delivery times
- **Quality Score**: Overall supplier performance rating

## Data Structure

The application generates realistic synthetic data including:
- Product information (ID, category, supplier)
- Daily demand patterns with seasonality
- Inventory levels and movements
- Supplier performance metrics
- Cost and pricing data

## Business Value

This analytics platform demonstrates:
- **Predictive Analytics**: Forecast future demand patterns
- **Optimization**: Minimize inventory costs while maintaining service levels
- **Risk Management**: Identify potential stockouts and supply disruptions
- **Performance Monitoring**: Track supplier and inventory KPIs
- **Decision Support**: Data-driven recommendations for inventory management

## Portfolio Highlights

- Real-world business problem solving
- Advanced time series forecasting
- Interactive dashboard development
- Supply chain domain expertise
- End-to-end analytics pipeline
- Professional data visualization

## Requirements

All dependencies are automatically installed when running the application:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- statsmodels
- prophet
- scikit-learn

## Usage Tips

1. **Dashboard Overview**: Start here for high-level KPIs and trends
2. **Inventory Analysis**: Deep dive into specific product performance
3. **Demand Forecasting**: Generate predictions for planning
4. **Reorder Optimization**: Get actionable reorder recommendations
5. **Supplier Performance**: Evaluate and compare supplier metrics

## Future Enhancements

- Machine learning-based demand forecasting
- Multi-echelon inventory optimization
- Real-time data integration
- Advanced supplier risk assessment
- Cost optimization algorithms
- Mobile-responsive design

---

**Author**: Supply Chain Analytics Team  
**Version**: 1.0  
**License**: MIT