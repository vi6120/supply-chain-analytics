# Supply Chain & Inventory Optimization Analytics

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
- Service level configuration (85%-99.9%)
- **Service-Level Driven Inventory Planning**
  - Interactive cost vs service trade-off slider
  - Multi-scenario comparison (85%, 90%, 95%, 99%, 99.9%)
  - Real-time safety stock and cost calculations
  - Optimal service level identification
- Inventory simulation
- Automated reorder alerts

### 🚚 Supplier Performance
- Delivery delay tracking
- Lead time analysis
- Performance scoring
- Supplier rankings

### 💰 Cost & Financial Integration
- Carrying cost simulation ($/unit/day)
- Lost sales estimation from stockouts
- Dynamic margin analysis
- ROI calculations
- Financial impact visualization
- Cost optimization recommendations

### 📈 Enhanced Inventory Analysis
- Inventory threshold management
- Upper and lower limit monitoring
- Threshold violation tracking
- Utilization percentage analysis
- Real-time threshold alerts

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
- **Service Level**: Probability of not stocking out (85%-99.9%)
- **Carrying Cost**: Cost of holding inventory

### Reorder Calculations
- **Reorder Point (ROP)**: Demand during Lead Time + Safety Stock
- **Safety Stock**: Buffer inventory for demand variability (service-level driven)
- **Service Level Scenarios**: Multi-level analysis with cost trade-offs
- **Optimal Service Level**: Cost-minimizing service level calculation

### Supplier Metrics
- **Delivery Performance**: On-time delivery percentage
- **Lead Time Variability**: Consistency in delivery times
- **Quality Score**: Overall supplier performance rating

### Financial Metrics
- **Carrying Cost per Unit**: Daily holding cost calculation
- **Lost Sales Impact**: Revenue loss from stockouts
- **Margin Loss**: Profit impact from delays and variability
- **Inventory ROI**: Return on inventory investment
- **Total Cost Impact**: Comprehensive financial analysis

### Inventory Thresholds
- **Maximum Inventory**: Upper limit (30 days demand)
- **Minimum Inventory**: Lower limit (5 days demand)
- **Reorder Point**: Trigger level with safety buffer
- **Threshold Utilization**: Current vs maximum capacity
- **Violation Tracking**: Days exceeding limits

## Data Structure

The application generates enhanced realistic synthetic data including:
- Product information (ID, category, supplier)
- Daily demand patterns with seasonality and weekly cycles
- Inventory levels with realistic restocking logic
- Inventory thresholds (max, min, reorder points)
- Enhanced stockout scenarios (25% more realistic)
- Supplier performance with variable delays (25% chance)
- Cost and pricing data with profit margins
- Financial impact calculations
- Threshold violation tracking

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
2. **Inventory Analysis**: Deep dive into specific product performance with threshold monitoring
3. **Demand Forecasting**: Generate predictions for planning with Prophet/ARIMA
4. **Reorder Optimization**: Get actionable reorder recommendations with safety stock
   - **Service-Level Planning**: Use the interactive slider to balance cost vs customer service
   - **Scenario Comparison**: Compare 5 service levels (85%-99.9%) with cost implications
5. **Supplier Performance**: Evaluate and compare supplier metrics with scoring
6. **Cost & Financial Integration**: Analyze financial impact and ROI of inventory decisions
7. **Threshold Management**: Monitor inventory limits and prevent overstocking/understocking

## Future Enhancements

- Machine learning-based demand forecasting
- Multi-echelon inventory optimization
- Real-time data integration
- Advanced supplier risk assessment
- Cost optimization algorithms
- Mobile-responsive design

---

**Author**: Arvind Dharanidharan & Vikas Ramaswamy

**Version**: 1.0

**License**: MIT



A comprehensive analytics dashboard for supply chain optimization featuring demand forecasting, inventory analysis, and reorder point optimization.