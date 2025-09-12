import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Supply Chain Analytics", layout="wide")

@st.cache_data
def generate_data():
    np.random.seed(42)
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for product_id in range(1, 21):
        for day in range(365):
            date = start_date + timedelta(days=day)
            demand = max(0, int(100 + 50 * np.sin(2 * np.pi * day / 365) + np.random.normal(0, 20)))
            inventory = max(0, 500 - demand + np.random.randint(-50, 100))
            
            data.append({
                'date': date,
                'product_id': f'PROD_{product_id:03d}',
                'demand': demand,
                'inventory': inventory,
                'unit_cost': np.random.uniform(10, 100)
            })
    
    return pd.DataFrame(data)

def main():
    st.title("Supply Chain Analytics Dashboard")
    
    df = generate_data()
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Product Analysis", "Forecasting"])
    
    with tab1:
        st.header("Dashboard Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_demand = df['demand'].sum()
            st.metric("Total Demand", f"{total_demand:,}")
        
        with col2:
            avg_inventory = df['inventory'].mean()
            st.metric("Avg Inventory", f"{avg_inventory:.0f}")
        
        with col3:
            total_value = (df['inventory'] * df['unit_cost']).sum()
            st.metric("Inventory Value", f"${total_value:,.0f}")
        
        # Daily demand chart
        daily_demand = df.groupby('date')['demand'].sum().reset_index()
        fig = px.line(daily_demand, x='date', y='demand', title="Daily Demand Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Product Analysis")
        
        products = df['product_id'].unique()
        selected_product = st.selectbox("Select Product", products)
        
        product_data = df[df['product_id'] == selected_product]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(product_data, x='date', y='inventory', title=f"Inventory - {selected_product}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(product_data, x='date', y='demand', title=f"Demand - {selected_product}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Simple Forecasting")
        
        products = df['product_id'].unique()
        selected_product = st.selectbox("Product for Forecast", products, key="forecast")
        
        product_data = df[df['product_id'] == selected_product]
        
        # Simple moving average forecast
        window = 30
        product_data['forecast'] = product_data['demand'].rolling(window=window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=product_data['date'], y=product_data['demand'], name='Actual Demand'))
        fig.add_trace(go.Scatter(x=product_data['date'], y=product_data['forecast'], name='Forecast'))
        fig.update_layout(title=f"Demand Forecast - {selected_product}")
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()