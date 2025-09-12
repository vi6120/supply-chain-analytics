#!/usr/bin/env python3
"""
Supply Chain & Inventory Optimization Analytics
A comprehensive analytics dashboard for supply chain optimization
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    .stSelectbox > div > div {
        background-color: white !important;
        color: black !important;
    }
    .stSelectbox > div > div > div {
        background-color: white !important;
        color: black !important;
    }
    .stSelectbox label {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

class SupplyChainAnalytics:
    def __init__(self):
        self.data = None
        self.forecasts = {}
        
    def generate_synthetic_data(self, n_products=50, n_days=365):
        """Generate realistic supply chain data with inventory thresholds"""
        np.random.seed(42)
        
        # Product categories and their characteristics
        categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
        suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D', 'Supplier_E']
        
        data = []
        start_date = datetime.now() - timedelta(days=n_days)
        
        for product_id in range(1, n_products + 1):
            category = np.random.choice(categories)
            supplier = np.random.choice(suppliers)
            
            # Base demand varies by category
            base_demand = {
                'Electronics': 100, 'Clothing': 150, 'Food': 200, 
                'Home': 80, 'Sports': 120
            }[category]
            
            # Lead time varies by supplier
            lead_time = np.random.randint(3, 15)
            unit_cost = np.random.uniform(10, 500)
            
            # Inventory thresholds
            max_inventory = base_demand * 30  # 30 days of demand
            min_inventory = base_demand * 5   # 5 days of demand
            reorder_point = base_demand * lead_time * 1.5  # Safety buffer
            
            current_inventory = np.random.randint(int(min_inventory), int(max_inventory))
            
            for day in range(n_days):
                date = start_date + timedelta(days=day)
                
                # Seasonal patterns
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 365)
                
                # Weekly patterns (lower on weekends)
                weekly_factor = 0.7 if date.weekday() >= 5 else 1.0
                
                # Random noise
                noise = np.random.normal(0, 0.2)
                
                demand = max(0, int(base_demand * seasonal_factor * weekly_factor * (1 + noise)))
                
                # Inventory simulation with realistic restocking
                if day == 0:
                    inventory = current_inventory
                else:
                    prev_inventory = current_inventory
                    
                    # Consume inventory based on demand
                    inventory_after_demand = max(0, prev_inventory - demand)
                    
                    # Restock if below reorder point (with some randomness)
                    if inventory_after_demand <= reorder_point and np.random.random() < 0.7:
                        # Restock to 80-95% of max capacity
                        restock_amount = np.random.randint(int(max_inventory * 0.8), int(max_inventory * 0.95)) - inventory_after_demand
                        inventory = min(max_inventory, inventory_after_demand + restock_amount)
                    else:
                        inventory = inventory_after_demand
                    
                    current_inventory = inventory
                
                # Enhanced stockout logic with more realistic scenarios
                stockout = 0
                if inventory < demand:
                    stockout = 1
                elif inventory <= min_inventory * 0.3 and np.random.random() < 0.4:
                    stockout = 1  # Random stockouts when very low inventory
                elif delivery_delay > 2 and inventory <= reorder_point and np.random.random() < 0.6:
                    stockout = 1  # Stockouts due to delivery delays
                
                # More realistic supplier delays
                delivery_delay = 0
                if np.random.random() < 0.25:  # 25% chance of delay
                    delivery_delay = np.random.randint(1, 5)
                
                data.append({
                    'date': date,
                    'product_id': f'PROD_{product_id:03d}',
                    'category': category,
                    'supplier': supplier,
                    'demand': demand,
                    'inventory': inventory,
                    'unit_cost': unit_cost,
                    'lead_time': lead_time,
                    'stockout': stockout,
                    'delivery_delay': delivery_delay,
                    'max_inventory': max_inventory,
                    'min_inventory': min_inventory,
                    'reorder_point': reorder_point
                })
        
        return pd.DataFrame(data)
    
    def calculate_kpis(self, df):
        """Calculate key supply chain KPIs"""
        kpis = {}
        
        # Inventory Turnover = COGS / Average Inventory
        df['cogs'] = df['demand'] * df['unit_cost']
        product_metrics = df.groupby('product_id').agg({
            'inventory': 'mean',
            'cogs': 'sum',
            'stockout': 'sum',
            'delivery_delay': 'mean',
            'demand': 'sum'
        }).reset_index()
        
        product_metrics['inventory_turnover'] = product_metrics['cogs'] / product_metrics['inventory']
        product_metrics['stockout_rate'] = product_metrics['stockout'] / len(df['date'].unique()) * 100
        
        kpis['avg_inventory_turnover'] = product_metrics['inventory_turnover'].mean()
        kpis['avg_stockout_rate'] = product_metrics['stockout_rate'].mean()
        kpis['avg_delivery_delay'] = product_metrics['delivery_delay'].mean()
        kpis['total_demand'] = df['demand'].sum()
        kpis['total_inventory_value'] = (df['inventory'] * df['unit_cost']).sum()
        
        return kpis, product_metrics
    
    def forecast_demand(self, df, product_id, method='prophet', periods=30):
        """Forecast demand using Prophet or ARIMA"""
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data.groupby('date')['demand'].sum().reset_index()
        
        if len(product_data) < 30:
            return None
        
        if method == 'prophet':
            # Prophet forecasting
            prophet_data = product_data.rename(columns={'date': 'ds', 'demand': 'y'})
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_data)
            
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return {
                'method': 'Prophet',
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods),
                'model': model
            }
        
        elif method == 'arima':
            # ARIMA forecasting
            try:
                model = ARIMA(product_data['demand'], order=(1, 1, 1))
                fitted_model = model.fit()
                
                forecast = fitted_model.forecast(steps=periods)
                conf_int = fitted_model.get_forecast(steps=periods).conf_int()
                
                future_dates = pd.date_range(
                    start=product_data['date'].max() + timedelta(days=1),
                    periods=periods,
                    freq='D'
                )
                
                forecast_df = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': forecast,
                    'yhat_lower': conf_int.iloc[:, 0],
                    'yhat_upper': conf_int.iloc[:, 1]
                })
                
                return {
                    'method': 'ARIMA',
                    'forecast': forecast_df,
                    'model': fitted_model
                }
            except:
                return None
    
    def calculate_reorder_point(self, avg_demand, lead_time, service_level=0.95):
        """Calculate reorder point with safety stock"""
        # Safety stock calculation (simplified)
        z_score = 1.96 if service_level == 0.95 else 1.65  # 95% or 90% service level
        demand_std = avg_demand * 0.3  # Assume 30% coefficient of variation
        
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        reorder_point = (avg_demand * lead_time) + safety_stock
        
        return {
            'reorder_point': int(reorder_point),
            'safety_stock': int(safety_stock),
            'avg_demand_during_lead_time': int(avg_demand * lead_time)
        }
    
    def calculate_carrying_costs(self, df, product_id, carrying_rate=0.25):
        """Calculate detailed carrying costs"""
        product_data = df[df['product_id'] == product_id]
        unit_cost = product_data['unit_cost'].iloc[0]
        avg_inventory = product_data['inventory'].mean()
        
        # Carrying cost components
        daily_carrying_rate = carrying_rate / 365
        cost_per_unit_per_day = unit_cost * daily_carrying_rate
        total_daily_carrying_cost = avg_inventory * cost_per_unit_per_day
        annual_carrying_cost = total_daily_carrying_cost * 365
        
        return {
            'cost_per_unit_per_day': cost_per_unit_per_day,
            'total_daily_carrying_cost': total_daily_carrying_cost,
            'annual_carrying_cost': annual_carrying_cost,
            'avg_inventory_value': avg_inventory * unit_cost
        }
    
    def calculate_lost_sales(self, df, product_id, profit_margin=0.30):
        """Calculate lost sales from stockouts"""
        product_data = df[df['product_id'] == product_id]
        unit_cost = product_data['unit_cost'].iloc[0]
        selling_price = unit_cost / (1 - profit_margin)
        unit_profit = selling_price - unit_cost
        
        # Calculate lost sales
        stockout_days = product_data[product_data['stockout'] == 1]
        total_lost_units = stockout_days['demand'].sum()
        lost_revenue = total_lost_units * selling_price
        lost_profit = total_lost_units * unit_profit
        
        return {
            'stockout_days': len(stockout_days),
            'total_lost_units': total_lost_units,
            'lost_revenue': lost_revenue,
            'lost_profit': lost_profit,
            'unit_profit': unit_profit,
            'selling_price': selling_price
        }
    
    def calculate_margin_impact(self, df, product_id, profit_margin=0.30):
        """Calculate margin impact from supplier delays and demand fluctuations"""
        product_data = df[df['product_id'] == product_id]
        unit_cost = product_data['unit_cost'].iloc[0]
        selling_price = unit_cost / (1 - profit_margin)
        
        # Delay impact
        delayed_deliveries = product_data[product_data['delivery_delay'] > 0]
        delay_impact_units = delayed_deliveries['demand'].sum()
        delay_cost = delay_impact_units * unit_cost * 0.05  # 5% cost increase per delay
        
        # Demand variability impact
        demand_std = product_data['demand'].std()
        demand_cv = demand_std / product_data['demand'].mean()
        variability_cost = demand_cv * product_data['demand'].sum() * unit_cost * 0.02
        
        total_margin_loss = delay_cost + variability_cost
        
        return {
            'delay_impact_units': delay_impact_units,
            'delay_cost': delay_cost,
            'demand_variability_cost': variability_cost,
            'total_margin_loss': total_margin_loss,
            'margin_loss_percentage': (total_margin_loss / (product_data['demand'].sum() * selling_price)) * 100
        }
    
    def calculate_inventory_thresholds(self, df, product_id):
        """Calculate inventory threshold analysis"""
        product_data = df[df['product_id'] == product_id]
        
        current_inventory = product_data['inventory'].iloc[-1]
        max_threshold = product_data['max_inventory'].iloc[0]
        min_threshold = product_data['min_inventory'].iloc[0]
        reorder_point = product_data['reorder_point'].iloc[0]
        avg_inventory = product_data['inventory'].mean()
        
        # Calculate threshold violations
        over_max = len(product_data[product_data['inventory'] > max_threshold])
        under_min = len(product_data[product_data['inventory'] < min_threshold])
        
        return {
            'current_inventory': current_inventory,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'reorder_point': reorder_point,
            'avg_inventory': avg_inventory,
            'over_max_days': over_max,
            'under_min_days': under_min,
            'threshold_utilization': (avg_inventory / max_threshold) * 100
        }

def show_kpi_catalog():
    """Display KPI definitions and explanations"""
    with st.expander("📚 KPI Catalog & Technical Definitions"):
        st.markdown("""
        ### 📊 **Inventory KPIs**
        - **Inventory Turnover**: Measures how efficiently inventory is used. Higher values indicate better efficiency.
        - **Stockout Rate**: Percentage of time when demand cannot be met. Lower is better for customer service.
        - **Service Level**: Probability of meeting demand without stockouts. Target: 95-99%.
        - **Carrying Cost**: Cost of holding inventory including storage, insurance, and capital costs.
        
        ### 💰 **Financial Metrics**
        - **Cost per Unit per Day**: Daily holding cost for each inventory unit.
        - **Lost Sales**: Revenue lost when customers cannot purchase due to stockouts.
        - **ROI (Return on Investment)**: Profit generated per dollar invested in inventory.
        - **Margin Loss**: Profit reduction due to operational inefficiencies.
        
        ### 📈 **Threshold Management**
        - **Maximum Inventory**: Upper safety limit to prevent overstocking (30 days demand).
        - **Minimum Inventory**: Lower safety limit to prevent stockouts (5 days demand).
        - **Reorder Point**: Inventory level that triggers new orders.
        - **Threshold Utilization**: Current inventory as percentage of maximum capacity.
        
        ### 🚚 **Supplier Metrics**
        - **Lead Time**: Time between order placement and delivery.
        - **Delivery Delay**: Days beyond promised delivery date.
        - **Performance Score**: Composite metric combining reliability and speed.
        """)

def main():
    # Initialize the analytics class
    try:
        if 'analytics' not in st.session_state:
            st.session_state.analytics = SupplyChainAnalytics()
            st.session_state.data_loaded = False
        
        analytics = st.session_state.analytics
    except:
        # Fallback for non-streamlit execution
        analytics = SupplyChainAnalytics()
        analytics.data = analytics.generate_synthetic_data()
        return
    
    # Header
    st.markdown('<h1 class="main-header">Supply Chain & Inventory Optimization Analytics</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "Dashboard Overview",
        "Inventory Analysis", 
        "Demand Forecasting",
        "Reorder Optimization",
        "Supplier Performance",
        "Cost & Financial Integration"
    ])
    
    # Add KPI Catalog to sidebar
    st.sidebar.markdown("---")
    show_kpi_catalog()
    
    # Load or generate data
    try:
        if not st.session_state.data_loaded:
            with st.spinner("Generating supply chain data..."):
                analytics.data = analytics.generate_synthetic_data()
                st.session_state.data_loaded = True
            st.success("Data loaded successfully!")
        
        df = analytics.data
    except:
        # Fallback data loading
        if analytics.data is None:
            analytics.data = analytics.generate_synthetic_data()
        df = analytics.data
    
    if page == "Dashboard Overview":
        st.header("Executive Dashboard")
        
        # Calculate KPIs
        kpis, product_metrics = analytics.calculate_kpis(df)
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Inventory Turnover</h3>
                <h2>{kpis['avg_inventory_turnover']:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Stockout Rate</h3>
                <h2>{kpis['avg_stockout_rate']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Delivery Delay</h3>
                <h2>{kpis['avg_delivery_delay']:.1f} days</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Inventory Value</h3>
                <h2>${kpis['total_inventory_value']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI Explanations
        st.markdown("""
        ### 📊 **KPI Explanations**
        
        - **Inventory Turnover {:.2f}**: Higher values (>6) indicate efficient inventory management
        - **Stockout Rate {:.1f}%**: Target <5% for good customer service
        - **Delivery Delay {:.1f} days**: Lower values indicate reliable suppliers
        - **Inventory Value ${:,.0f}**: Total capital tied up in inventory
        """.format(kpis['avg_inventory_turnover'], kpis['avg_stockout_rate'], 
                   kpis['avg_delivery_delay'], kpis['total_inventory_value']))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Demand Trend")
            daily_demand = df.groupby('date')['demand'].sum().reset_index()
            fig = px.line(daily_demand, x='date', y='demand', title="Total Daily Demand")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Demand by Category")
            category_demand = df.groupby('category')['demand'].sum().reset_index()
            fig = px.pie(category_demand, values='demand', names='category', title="Demand Distribution")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Chart Explanations
        st.markdown("""
        ### 📈 **Chart Explanations**
        
        **Daily Demand Trend**: Shows total demand across all products over time. Look for:
        - Seasonal patterns (peaks and valleys)
        - Weekly cycles (lower weekend demand)
        - Growth or decline trends
        
        **Demand Distribution**: Pie chart showing which product categories drive the most demand. Use this to:
        - Identify high-volume categories requiring more attention
        - Allocate resources based on demand patterns
        - Plan category-specific inventory strategies
        """)
    
    elif page == "Inventory Analysis":
        st.header("Inventory Analysis")
        
        # Product selection
        products = df['product_id'].unique()
        selected_product = st.selectbox("Select Product", products)
        
        product_data = df[df['product_id'] == selected_product]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Inventory Levels Over Time")
            fig = px.line(product_data, x='date', y='inventory', title=f"Inventory for {selected_product}")
            fig.add_hline(y=product_data['inventory'].mean(), line_dash="dash", 
                         annotation_text="Average Level")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Inventory Thresholds")
            threshold_data = analytics.calculate_inventory_thresholds(df, selected_product)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=product_data['date'], y=product_data['inventory'], 
                                   name='Inventory', line=dict(color='blue')))
            fig.add_hline(y=threshold_data['max_threshold'], line_dash="dash", 
                         line_color="red", annotation_text="Max Threshold")
            fig.add_hline(y=threshold_data['min_threshold'], line_dash="dash", 
                         line_color="orange", annotation_text="Min Threshold")
            fig.add_hline(y=threshold_data['reorder_point'], line_dash="dot", 
                         line_color="green", annotation_text="Reorder Point")
            fig.update_layout(title=f"Inventory Thresholds for {selected_product}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Inventory metrics
        st.subheader("Inventory Metrics & Thresholds")
        threshold_data = analytics.calculate_inventory_thresholds(df, selected_product)
        
        col1, col2, col3, col4 = st.columns(4)
        
        stockout_days = product_data['stockout'].sum()
        turnover = (product_data['demand'].sum() * product_data['unit_cost'].iloc[0]) / threshold_data['avg_inventory']
        
        col1.metric("Current Inventory", f"{threshold_data['current_inventory']:.0f} units")
        col2.metric("Stockout Days", f"{stockout_days} days")
        col3.metric("Threshold Utilization", f"{threshold_data['threshold_utilization']:.1f}%")
        col4.metric("Inventory Turnover", f"{turnover:.2f}")
        
        # Threshold Analysis
        st.markdown("""
        ### 📈 **Threshold Analysis**
        
        **Current Inventory**: {:.0f} units ({:.1f}% of max capacity)
        
        **Threshold Status**:
        - Max Threshold: {:.0f} units (30 days demand)
        - Min Threshold: {:.0f} units (5 days demand)
        - Reorder Point: {:.0f} units (includes safety stock)
        
        **Performance**: {} stockout days detected in the analysis period.
        """.format(
            threshold_data['current_inventory'], threshold_data['threshold_utilization'],
            threshold_data['max_threshold'], threshold_data['min_threshold'], 
            threshold_data['reorder_point'], stockout_days
        ))
        
        # Threshold violations
        col1, col2 = st.columns(2)
        with col1:
            if threshold_data['over_max_days'] > 0:
                st.warning(f"⚠️ Exceeded max threshold {threshold_data['over_max_days']} days")
            else:
                st.success("✅ No max threshold violations")
        
        with col2:
            if threshold_data['under_min_days'] > 0:
                st.error(f"🚨 Below min threshold {threshold_data['under_min_days']} days")
            else:
                st.success("✅ No min threshold violations")
        
        # Chart Explanations
        st.markdown("""
        ### 📈 **Chart Explanations**
        
        **Inventory Levels Over Time**: Shows how inventory changes daily. Key insights:
        - Inventory depletion patterns
        - Restocking frequency and timing
        - Seasonal inventory adjustments
        
        **Inventory Thresholds**: Displays inventory against safety limits:
        - **Red Line (Max)**: Overstocking risk above this line
        - **Orange Line (Min)**: Stockout risk below this line  
        - **Green Line (Reorder)**: Optimal reorder trigger point
        - **Blue Line (Actual)**: Current inventory trajectory
        """)
    
    elif page == "Demand Forecasting":
        st.header("Demand Forecasting")
        
        # Product and method selection
        col1, col2 = st.columns(2)
        with col1:
            products = df['product_id'].unique()
            selected_product = st.selectbox("Select Product", products)
        with col2:
            method = st.selectbox("Forecasting Method", ["prophet", "arima"])
        
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                forecast_result = analytics.forecast_demand(df, selected_product, method, forecast_days)
            
            if forecast_result:
                st.success(f"Forecast generated using {forecast_result['method']}")
                
                # Historical data
                historical = df[df['product_id'] == selected_product].groupby('date')['demand'].sum().reset_index()
                
                # Plot forecast
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical['date'], 
                    y=historical['demand'],
                    name='Historical Demand',
                    line=dict(color='blue')
                ))
                
                # Forecast
                forecast_df = forecast_result['forecast']
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'], 
                    y=forecast_df['yhat'],
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'], 
                    y=forecast_df['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'], 
                    y=forecast_df['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='Confidence Interval',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                
                fig.update_layout(
                    title=f"Demand Forecast for {selected_product}",
                    xaxis_title="Date",
                    yaxis_title="Demand",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                st.subheader("Forecast Summary")
                col1, col2, col3 = st.columns(3)
                
                avg_forecast = forecast_df['yhat'].mean()
                max_forecast = forecast_df['yhat'].max()
                min_forecast = forecast_df['yhat'].min()
                
                col1.metric("Average Forecasted Demand", f"{avg_forecast:.0f}")
                col2.metric("Peak Demand", f"{max_forecast:.0f}")
                col3.metric("Minimum Demand", f"{min_forecast:.0f}")
                
                # Chart Explanations
                st.markdown("""
                ### 📈 **Forecast Chart Explanation**
                
                **Blue Line (Historical)**: Past demand data showing actual patterns
                **Red Dashed Line (Forecast)**: Predicted future demand using {}
                **Gray Shaded Area**: Confidence interval (uncertainty range)
                
                **Key Insights**:
                - Wider confidence intervals indicate higher uncertainty
                - Seasonal patterns help predict future peaks/valleys
                - Use forecasts for procurement planning and capacity management
                """.format(forecast_result['method']))
            else:
                st.error("Unable to generate forecast. Insufficient data.")
        
        # Forecasting Method Explanations
        st.markdown("""
        ### 🤖 **Forecasting Methods**
        
        **Prophet**: Advanced ML model that handles:
        - Seasonal patterns (yearly, weekly)
        - Holiday effects and trend changes
        - Missing data and outliers
        
        **ARIMA**: Statistical model that captures:
        - Autoregressive patterns
        - Moving averages
        - Trend and seasonality components
        """)
    
    elif page == "Reorder Optimization":
        st.header("Reorder Point Optimization")
        
        # Product selection
        products = df['product_id'].unique()
        selected_product = st.selectbox("Select Product", products)
        
        product_data = df[df['product_id'] == selected_product]
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            service_level = st.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)
        with col2:
            lead_time = st.number_input("Lead Time (days)", 
                                      value=int(product_data['lead_time'].iloc[0]), 
                                      min_value=1, max_value=30)
        
        # Calculate metrics
        avg_demand = product_data['demand'].mean()
        current_inventory = product_data['inventory'].iloc[-1]
        
        # Calculate reorder point
        reorder_calc = analytics.calculate_reorder_point(avg_demand, lead_time, service_level)
        
        # Display results
        st.subheader("Reorder Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current Inventory", f"{current_inventory:.0f} units")
        col2.metric("Reorder Point", f"{reorder_calc['reorder_point']:.0f} units")
        col3.metric("Safety Stock", f"{reorder_calc['safety_stock']:.0f} units")
        
        # Reorder status
        if current_inventory <= reorder_calc['reorder_point']:
            st.error(f"🚨 REORDER NOW! Current inventory ({current_inventory}) is below reorder point ({reorder_calc['reorder_point']})")
            days_until_stockout = current_inventory / avg_demand
            st.warning(f"Estimated days until stockout: {days_until_stockout:.1f} days")
        else:
            days_until_reorder = (current_inventory - reorder_calc['reorder_point']) / avg_demand
            st.success(f"✅ Inventory OK. Reorder in approximately {days_until_reorder:.1f} days")
        
        # Visualization
        st.subheader("Inventory Simulation")
        
        # Create simulation data
        days = 60
        sim_dates = pd.date_range(start=product_data['date'].max(), periods=days, freq='D')
        sim_inventory = [current_inventory]
        
        for i in range(1, days):
            daily_demand = np.random.poisson(avg_demand)
            new_inventory = max(0, sim_inventory[-1] - daily_demand)
            sim_inventory.append(new_inventory)
        
        sim_df = pd.DataFrame({
            'date': sim_dates,
            'inventory': sim_inventory
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_df['date'], y=sim_df['inventory'], 
                               name='Projected Inventory', line=dict(color='blue')))
        fig.add_hline(y=reorder_calc['reorder_point'], line_dash="dash", 
                     line_color="red", annotation_text="Reorder Point")
        fig.add_hline(y=reorder_calc['safety_stock'], line_dash="dot", 
                     line_color="orange", annotation_text="Safety Stock")
        
        fig.update_layout(
            title="Inventory Projection",
            xaxis_title="Date",
            yaxis_title="Inventory Level",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Chart Explanations
        st.markdown("""
        ### 📈 **Inventory Projection Explanation**
        
        **Blue Line**: Projected inventory levels over next 60 days
        **Red Dashed Line**: Reorder point - triggers new orders
        **Orange Dotted Line**: Safety stock - minimum buffer level
        
        **Reorder Logic**:
        - Order when inventory hits reorder point
        - Safety stock prevents stockouts during lead time
        - Service level determines safety stock size
        
        **Current Status**: {}
        
        **Recommendations**:
        - Monitor inventory daily
        - Adjust reorder points based on demand changes
        - Consider supplier reliability in safety stock calculations
        """.format(
            "REORDER NOW!" if current_inventory <= reorder_calc['reorder_point'] else "Inventory OK"
        ))
    
    elif page == "Supplier Performance":
        st.header("Supplier Performance Analysis")
        
        # Supplier metrics
        supplier_metrics = df.groupby('supplier').agg({
            'delivery_delay': ['mean', 'std'],
            'lead_time': 'mean',
            'demand': 'sum',
            'stockout': 'sum'
        }).round(2)
        
        supplier_metrics.columns = ['Avg_Delivery_Delay', 'Delay_Std', 'Avg_Lead_Time', 'Total_Demand', 'Total_Stockouts']
        supplier_metrics = supplier_metrics.reset_index()
        
        # Calculate performance score (lower is better)
        supplier_metrics['Performance_Score'] = (
            supplier_metrics['Avg_Delivery_Delay'] * 0.4 + 
            supplier_metrics['Delay_Std'] * 0.3 + 
            supplier_metrics['Avg_Lead_Time'] * 0.3
        )
        
        supplier_metrics = supplier_metrics.sort_values('Performance_Score')
        
        # Display metrics table
        st.subheader("Supplier Performance Metrics")
        st.dataframe(supplier_metrics, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Delivery Performance")
            fig = px.bar(supplier_metrics, x='supplier', y='Avg_Delivery_Delay',
                        title="Average Delivery Delay by Supplier")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Lead Time Comparison")
            fig = px.bar(supplier_metrics, x='supplier', y='Avg_Lead_Time',
                        title="Average Lead Time by Supplier")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Best/Worst suppliers
        st.subheader("Supplier Rankings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🏆 Best Performer: {supplier_metrics.iloc[0]['supplier']}")
            st.write(f"Performance Score: {supplier_metrics.iloc[0]['Performance_Score']:.2f}")
        
        with col2:
            st.error(f"⚠️ Needs Improvement: {supplier_metrics.iloc[-1]['supplier']}")
            st.write(f"Performance Score: {supplier_metrics.iloc[-1]['Performance_Score']:.2f}")
        
        # Chart Explanations
        st.markdown("""
        ### 📈 **Supplier Analysis Explanation**
        
        **Delivery Performance Chart**: Shows average delays by supplier
        - Lower bars indicate more reliable suppliers
        - Consistent performers have minimal delays
        
        **Lead Time Comparison**: Shows promised delivery times
        - Shorter lead times enable faster response to demand
        - Consider lead time vs reliability trade-offs
        
        **Performance Score Calculation**:
        - 40% Delivery Delay Weight
        - 30% Delay Variability Weight  
        - 30% Lead Time Weight
        - Lower scores indicate better performance
        
        **Supplier Selection Criteria**:
        - Reliability (consistent delivery)
        - Speed (short lead times)
        - Quality (low defect rates)
        """)
    
    elif page == "Cost & Financial Integration":
        st.header("Cost & Financial Analysis")
        
        # Product selection
        products = df['product_id'].unique()
        selected_product = st.selectbox("Select Product for Financial Analysis", products)
        
        # Financial parameters
        col1, col2 = st.columns(2)
        with col1:
            carrying_rate = st.slider("Annual Carrying Cost Rate", 0.15, 0.40, 0.25, 0.01)
        with col2:
            profit_margin = st.slider("Profit Margin", 0.10, 0.50, 0.30, 0.01)
        
        # Calculate financial metrics
        carrying_costs = analytics.calculate_carrying_costs(df, selected_product, carrying_rate)
        lost_sales = analytics.calculate_lost_sales(df, selected_product, profit_margin)
        margin_impact = analytics.calculate_margin_impact(df, selected_product, profit_margin)
        
        # Carrying Cost Analysis
        st.subheader("💰 Carrying Cost Simulation")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Cost per Unit per Day", f"${carrying_costs['cost_per_unit_per_day']:.2f}")
        col2.metric("Daily Carrying Cost", f"${carrying_costs['total_daily_carrying_cost']:.2f}")
        col3.metric("Annual Carrying Cost", f"${carrying_costs['annual_carrying_cost']:,.0f}")
        col4.metric("Avg Inventory Value", f"${carrying_costs['avg_inventory_value']:,.0f}")
        
        # Lost Sales Analysis
        st.subheader("📉 Lost Sales Estimation")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Stockout Days", f"{lost_sales['stockout_days']} days")
        col2.metric("Lost Units", f"{lost_sales['total_lost_units']:,.0f}")
        col3.metric("Lost Revenue", f"${lost_sales['lost_revenue']:,.0f}")
        col4.metric("Lost Profit", f"${lost_sales['lost_profit']:,.0f}")
        
        # Margin Impact Analysis
        st.subheader("📊 Dynamic Margin Analysis")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Delay Cost Impact", f"${margin_impact['delay_cost']:,.0f}")
        col2.metric("Variability Cost", f"${margin_impact['demand_variability_cost']:,.0f}")
        col3.metric("Total Margin Loss", f"{margin_impact['margin_loss_percentage']:.2f}%")
        
        # Financial Impact Visualization
        st.subheader("Financial Impact Breakdown")
        
        # Create financial impact chart
        financial_data = {
            'Cost Type': ['Carrying Costs', 'Lost Sales', 'Delay Impact', 'Variability Impact'],
            'Annual Cost': [
                carrying_costs['annual_carrying_cost'],
                lost_sales['lost_profit'],
                margin_impact['delay_cost'] * 12,
                margin_impact['demand_variability_cost'] * 12
            ]
        }
        
        fig = px.bar(pd.DataFrame(financial_data), x='Cost Type', y='Annual Cost',
                    title="Annual Financial Impact by Cost Type",
                    color='Cost Type')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost optimization recommendations
        st.subheader("💡 Cost Optimization Recommendations")
        
        total_annual_cost = (carrying_costs['annual_carrying_cost'] + 
                           lost_sales['lost_profit'] + 
                           margin_impact['total_margin_loss'] * 12)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Total Annual Cost Impact: ${total_annual_cost:,.0f}**")
            
            if lost_sales['stockout_days'] > 10:
                st.warning("🚨 High stockout frequency detected. Consider increasing safety stock.")
            
            if carrying_costs['annual_carrying_cost'] > lost_sales['lost_profit']:
                st.warning("💰 Carrying costs exceed lost sales. Consider reducing inventory levels.")
            else:
                st.success("✅ Inventory levels appear optimized for cost vs service trade-off.")
        
        with col2:
            # ROI calculation for inventory investment
            inventory_investment = carrying_costs['avg_inventory_value']
            annual_profit_impact = lost_sales['lost_profit'] - carrying_costs['annual_carrying_cost']
            roi = (annual_profit_impact / inventory_investment) * 100 if inventory_investment > 0 else 0
            
            st.metric("Inventory ROI", f"{roi:.1f}%")
            
            if margin_impact['delay_impact_units'] > 0:
                st.warning(f"⏰ Supplier delays affecting {margin_impact['delay_impact_units']:,.0f} units")
            
            if margin_impact['margin_loss_percentage'] > 5:
                st.error("📈 High margin loss detected. Review supplier performance.")
        
        # Chart Explanations
        st.markdown("""
        ### 📈 **Financial Analysis Explanation**
        
        **Annual Financial Impact Chart**: Shows cost breakdown by category:
        - **Carrying Costs**: Cost of holding inventory (storage, insurance, capital)
        - **Lost Sales**: Revenue lost due to stockouts
        - **Delay Impact**: Additional costs from supplier delays
        - **Variability Impact**: Costs from demand uncertainty
        
        **Key Financial Metrics**:
        - **Cost per Unit per Day**: ${:.2f} - Daily holding cost for each unit
        - **Total Annual Cost**: ${:,.0f} - Complete financial impact
        - **ROI**: {:.1f}% - Return on inventory investment
        
        **Optimization Insights**:
        - High carrying costs suggest overstocking
        - High lost sales indicate understocking
        - Balance these costs for optimal inventory levels
        """.format(
            carrying_costs['cost_per_unit_per_day'],
            total_annual_cost,
            roi
        ))

if __name__ == "__main__":
    main()