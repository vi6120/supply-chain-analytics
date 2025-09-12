#!/usr/bin/env python3
"""
Supply Chain & Inventory Optimization Analytics
A comprehensive analytics dashboard for supply chain optimization
"""

import subprocess
import sys
import os

# Required packages
REQUIRED_PACKAGES = [
    'streamlit',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'plotly',
    'statsmodels',
    'prophet',
    'scikit-learn'
]

def install_packages():
    """Install required packages if not available"""
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages before importing
install_packages()

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
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

class SupplyChainAnalytics:
    def __init__(self):
        self.data = None
        self.forecasts = {}
        
    def generate_synthetic_data(self, n_products=50, n_days=365):
        """Generate realistic supply chain data"""
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
            
            for day in range(n_days):
                date = start_date + timedelta(days=day)
                
                # Seasonal patterns
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 365)
                
                # Weekly patterns (lower on weekends)
                weekly_factor = 0.7 if date.weekday() >= 5 else 1.0
                
                # Random noise
                noise = np.random.normal(0, 0.2)
                
                demand = max(0, int(base_demand * seasonal_factor * weekly_factor * (1 + noise)))
                
                # Inventory simulation
                if day == 0:
                    inventory = np.random.randint(200, 1000)
                else:
                    prev_inventory = data[-1]['inventory'] if data and data[-1]['product_id'] == product_id else inventory
                    inventory = max(0, prev_inventory - demand + np.random.randint(0, 200))
                
                # Stock out if inventory is low
                stockout = 1 if inventory < demand * 0.1 else 0
                
                # Supplier performance
                delivery_delay = np.random.poisson(1) if np.random.random() < 0.15 else 0
                
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
                    'delivery_delay': delivery_delay
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
        "Supplier Performance"
    ])
    
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
            st.subheader("Demand vs Inventory")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=product_data['date'], y=product_data['demand'], 
                                   name='Demand', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=product_data['date'], y=product_data['inventory'], 
                                   name='Inventory', line=dict(color='blue')))
            fig.update_layout(title=f"Demand vs Inventory for {selected_product}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Inventory metrics
        st.subheader("Inventory Metrics")
        col1, col2, col3 = st.columns(3)
        
        avg_inventory = product_data['inventory'].mean()
        stockout_days = product_data['stockout'].sum()
        turnover = (product_data['demand'].sum() * product_data['unit_cost'].iloc[0]) / avg_inventory
        
        col1.metric("Average Inventory", f"{avg_inventory:.0f} units")
        col2.metric("Stockout Days", f"{stockout_days} days")
        col3.metric("Inventory Turnover", f"{turnover:.2f}")
    
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
            else:
                st.error("Unable to generate forecast. Insufficient data.")
    
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

if __name__ == "__main__":
    main()