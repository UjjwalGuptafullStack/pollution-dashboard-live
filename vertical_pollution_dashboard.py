import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Vertical Pollution Dispersion Live Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 5px 0;
    }
    .status-live {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .header-section {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants for both nodes
NODES_CONFIG = {
    'Node-1 (35m)': {
        'channel_id': '3111437',
        'api_key': 'YPFZP7D18YEMQXWG',
        'height': 35,
        'url': 'https://api.thingspeak.com/channels/3111437/feeds.json'
    },
    'Node-2 (25m)': {
        'channel_id': '2839248',
        'api_key': 'OWWYZK5OXTZBC65U',
        'height': 25,
        'url': 'https://api.thingspeak.com/channels/2839248/feeds.json'
    }
}

# Field mapping for ThingSpeak
FIELD_MAPPING = {
    'field1': 'Temperature_C',
    'field2': 'Humidity_%',
    'field3': 'PM2_5_ugm3',
    'field4': 'PM10_ugm3',
    'field5': 'CO2_ppm',
    'field6': 'CO_ppm',
    'field7': 'NO2_ppb'
}

# Cache data for 5 minutes
@st.cache_data(ttl=300)
def fetch_thingspeak_data(node_name, results=2000):
    """Fetch data from ThingSpeak API for specified node"""
    try:
        node_config = NODES_CONFIG[node_name]
        params = {'results': results}
        response = requests.get(node_config['url'], params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'feeds' not in data or not data['feeds']:
            st.error(f"No data found in {node_name} ThingSpeak channel")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data['feeds'])
        
        # Parse timestamp and handle timezone properly
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df['created_at_local'] = df['created_at'] + pd.Timedelta(hours=5, minutes=30)  # Convert to IST
        # Remove timezone info to avoid comparison issues
        df['created_at_local'] = df['created_at_local'].dt.tz_localize(None)
        
        # Add node information
        df['node_name'] = node_name
        df['height_m'] = node_config['height']
        
        # Rename fields and convert to numeric
        for field, name in FIELD_MAPPING.items():
            if field in df.columns:
                df[name] = pd.to_numeric(df[field], errors='coerce')
        
        # Drop original field columns and NaN rows
        df = df.drop(columns=[f'field{i}' for i in range(1, 9) if f'field{i}' in df.columns])
        df = df.dropna(subset=list(FIELD_MAPPING.values()))
        
        return df.sort_values('created_at')
        
    except Exception as e:
        st.error(f"Error fetching data from {node_name}: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_both_nodes_data(results=2000):
    """Fetch data from both nodes and combine for comparison"""
    node1_data = fetch_thingspeak_data('Node-1 (35m)', results)
    node2_data = fetch_thingspeak_data('Node-2 (25m)', results)
    
    return node1_data, node2_data

def calculate_vertical_gradient(df1, df2, parameter, time_window_minutes=30):
    """Calculate vertical gradient between two nodes"""
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return None
    
    # Synchronize timestamps (within time window)
    gradients = []
    
    for idx, row1 in df1.iterrows():
        # Find closest measurement from node 2
        time_diff = abs(df2['created_at_local'] - row1['created_at_local'])
        closest_idx = time_diff.idxmin()
        
        if time_diff[closest_idx].total_seconds() <= time_window_minutes * 60:
            row2 = df2.loc[closest_idx]
            
            # Calculate gradient (concentration difference / height difference)
            height_diff = row1['height_m'] - row2['height_m']  # 35m - 25m = 10m
            conc_diff = row1[parameter] - row2[parameter]
            
            if height_diff != 0:
                gradient = conc_diff / height_diff  # units per meter
                gradients.append({
                    'timestamp': row1['created_at_local'],
                    'gradient': gradient,
                    'node1_value': row1[parameter],
                    'node2_value': row2[parameter],
                    'height_diff': height_diff
                })
    
    return pd.DataFrame(gradients)

def create_dual_node_comparison(df1, df2, parameter):
    """Create comparison visualization between two nodes"""
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return None
    
    fig = go.Figure()
    
    # Node 1 (35m)
    fig.add_trace(go.Scatter(
        x=df1['created_at_local'],
        y=df1[parameter],
        mode='lines',
        name='Node-1 (35m)',
        line=dict(color='blue', width=2),
        opacity=0.8
    ))
    
    # Node 2 (25m)
    fig.add_trace(go.Scatter(
        x=df2['created_at_local'],
        y=df2[parameter],
        mode='lines',
        name='Node-2 (25m)',
        line=dict(color='red', width=2),
        opacity=0.8
    ))
    
    fig.update_layout(
        title=f"Dual Node Comparison - {parameter}",
        xaxis_title="Time (IST)",
        yaxis_title=f"{parameter}",
        height=400,
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    return fig

def create_vertical_gradient_plot(gradient_df, parameter):
    """Create vertical gradient visualization"""
    if gradient_df is None or gradient_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=gradient_df['timestamp'],
        y=gradient_df['gradient'],
        mode='lines+markers',
        name=f'Vertical Gradient',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Vertical Gradient - {parameter} (35m - 25m)",
        xaxis_title="Time (IST)",
        yaxis_title=f"Gradient ({parameter}/m)",
        height=400,
        hovermode='x'
    )
    
    return fig

def create_height_profile_plot(df1, df2, parameter, time_point=None):
    """Create vertical profile at specific time or latest values"""
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return None
    
    # Use latest values if no specific time given
    if time_point is None:
        val_35m = df1[parameter].iloc[-1]
        val_25m = df2[parameter].iloc[-1]
        time_str = df1['created_at_local'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
    else:
        # Find closest values to specified time
        idx1 = (abs(df1['created_at_local'] - time_point)).idxmin()
        idx2 = (abs(df2['created_at_local'] - time_point)).idxmin()
        val_35m = df1.loc[idx1, parameter]
        val_25m = df2.loc[idx2, parameter]
        time_str = time_point.strftime('%Y-%m-%d %H:%M:%S')
    
    heights = [25, 35]
    values = [val_25m, val_35m]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=values,
        y=heights,
        mode='markers+lines',
        name=f'{parameter} Profile',
        marker=dict(size=12, color=['red', 'blue']),
        line=dict(width=3, color='purple')
    ))
    
    # Add labels
    fig.add_annotation(x=val_25m, y=25, text="Node-2 (25m)", 
                      showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="red")
    fig.add_annotation(x=val_35m, y=35, text="Node-1 (35m)", 
                      showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="blue")
    
    fig.update_layout(
        title=f"Vertical Profile - {parameter}<br><sub>{time_str} IST</sub>",
        xaxis_title=f"{parameter}",
        yaxis_title="Height (meters)",
        height=500,
        showlegend=False
    )
    
    return fig

def get_data_status(df, node_name="Node"):
    """Get current data collection status"""
    if df is None or df.empty:
        return f"❌ No Data ({node_name})", "No data available", "danger"
    
    last_update = df['created_at_local'].iloc[-1]
    time_diff = datetime.now() - last_update.replace(tzinfo=None)
    
    if time_diff < timedelta(minutes=5):
        return f"🟢 Live ({node_name})", f"Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S IST')}", "success"
    elif time_diff < timedelta(hours=1):
        return f"🟡 Recent ({node_name})", f"Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S IST')}", "warning"
    else:
        return f"🔴 Offline ({node_name})", f"Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S IST')}", "danger"

def filter_data_by_timerange(df, timerange):
    """Filter dataframe by selected time range"""
    if df is None or df.empty:
        return df
        
    # Get timezone-aware datetime for comparison
    if not df['created_at_local'].empty:
        # Use the timezone from the data if available, otherwise assume UTC+5:30 (IST)
        if df['created_at_local'].dt.tz is not None:
            now = pd.Timestamp.now(tz=df['created_at_local'].dt.tz)
        else:
            # Convert to timezone-naive for comparison
            now = pd.Timestamp.now().tz_localize(None)
    else:
        now = pd.Timestamp.now().tz_localize(None)
    
    if timerange == "Last 6 hours":
        cutoff = now - timedelta(hours=6)
    elif timerange == "Last 24 hours":
        cutoff = now - timedelta(hours=24)
    elif timerange == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif timerange == "Last 30 days":
        cutoff = now - timedelta(days=30)
    else:
        return df
    
    # Ensure both sides of comparison have same timezone handling
    if df['created_at_local'].dt.tz is not None and cutoff.tz is None:
        cutoff = cutoff.tz_localize(df['created_at_local'].dt.tz.zone)
    elif df['created_at_local'].dt.tz is None and hasattr(cutoff, 'tz') and cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)
    
    return df[df['created_at_local'] > cutoff]

def calculate_diurnal_variation(df, parameter):
    """Calculate hourly averages for diurnal analysis"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df['hour'] = df['created_at_local'].dt.hour
    hourly_stats = df.groupby('hour')[parameter].agg(['mean', 'std', 'count']).reset_index()
    return hourly_stats

def calculate_day_night_comparison(df, parameter):
    """Compare day (6 AM - 6 PM) vs night (6 PM - 6 AM) values"""
    if df is None or df.empty:
        return {}
    
    df['hour'] = df['created_at_local'].dt.hour
    day_data = df[(df['hour'] >= 6) & (df['hour'] < 18)][parameter]
    night_data = df[(df['hour'] >= 18) | (df['hour'] < 6)][parameter]
    
    return {
        'day_mean': day_data.mean(),
        'day_std': day_data.std(),
        'day_count': len(day_data),
        'night_mean': night_data.mean(),
        'night_std': night_data.std(),
        'night_count': len(night_data)
    }

def calculate_weekly_trends(df, parameter):
    """Calculate weekly trends (day of week analysis)"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df['day_of_week'] = df['created_at_local'].dt.day_name()
    weekly_stats = df.groupby('day_of_week')[parameter].agg(['mean', 'std', 'count']).reset_index()
    
    # Order days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_stats['day_of_week'] = pd.Categorical(weekly_stats['day_of_week'], categories=day_order, ordered=True)
    weekly_stats = weekly_stats.sort_values('day_of_week')
    
    return weekly_stats

def create_correlation_heatmap(df):
    """Create correlation heatmap for all parameters"""
    if df is None or df.empty:
        return None
    
    numeric_cols = list(FIELD_MAPPING.values())
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return None
    
    corr_matrix = df[available_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Parameter Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    
    fig.update_layout(
        title_x=0.5,
        height=400,
        width=600
    )
    
    return fig

def create_vertical_profile_simulation(df, parameter):
    """Create simulated vertical profile for demonstration"""
    if df is None or df.empty:
        return None
    
    # Current node at 12m
    current_value = df[parameter].iloc[-1] if not df.empty else 50
    
    # Simulate values at different heights (demonstration)
    heights = [2, 12, 25, 50]
    
    # Simple model: surface concentrations higher, decreasing with height
    if 'PM' in parameter:
        # PM decreases with height due to settling and mixing
        multipliers = [1.3, 1.0, 0.8, 0.6]
    elif 'CO' in parameter:
        # CO slightly higher at surface
        multipliers = [1.2, 1.0, 0.9, 0.7]
    elif 'NO2' in parameter:
        # NO2 varies with height and source proximity
        multipliers = [1.4, 1.0, 0.7, 0.5]
    else:
        # Default profile
        multipliers = [1.1, 1.0, 0.9, 0.8]
    
    concentrations = [current_value * mult for mult in multipliers]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=concentrations,
        y=heights,
        mode='markers+lines',
        name=f'{parameter} Profile',
        marker=dict(size=10),
        line=dict(width=3)
    ))
    
    # Highlight current node
    fig.add_trace(go.Scatter(
        x=[current_value],
        y=[12],
        mode='markers',
        name='Current Node (12m)',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title=f"Vertical Profile - {parameter}",
        xaxis_title=f"{parameter}",
        yaxis_title="Height (meters)",
        height=400,
        showlegend=True
    )
    
    return fig

def create_time_series_plot(df, parameter, smoothing_window):
    """Create time series plot with optional smoothing"""
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    
    # Raw data
    fig.add_trace(go.Scatter(
        x=df['created_at_local'],
        y=df[parameter],
        mode='lines',
        name=f'{parameter} (Raw)',
        opacity=0.7
    ))
    
    # Smoothed data if window > 1
    if smoothing_window > 1:
        smoothed = df[parameter].rolling(window=smoothing_window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df['created_at_local'],
            y=smoothed,
            mode='lines',
            name=f'{parameter} (Smoothed)',
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title=f"{parameter} Time Series",
        xaxis_title="Time (IST)",
        yaxis_title=f"{parameter}",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="header-section">
        <h1>🌍 Vertical Pollution Dispersion Live Analysis</h1>
        <p>Real-time air quality monitoring and vertical profile estimation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Node selection
        analysis_mode = st.radio(
            "🏗️ Analysis Mode",
            ["Single Node", "Dual Node Comparison"],
            index=1  # Default to dual node comparison
        )
        
        if analysis_mode == "Single Node":
            selected_node = st.selectbox(
                "📡 Select Node",
                list(NODES_CONFIG.keys())
            )
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (10 min)", value=True)
        if auto_refresh:
            time.sleep(0.1)  # Small delay for smooth UX
        
        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Time range selection
        timerange = st.selectbox(
            "📅 Time Range",
            ["Last 6 hours", "Last 24 hours", "Last 7 days", "Last 30 days"]
        )
        
        # Parameter selection
        parameter = st.selectbox(
            "📊 Parameter",
            list(FIELD_MAPPING.values())
        )
        
        # Smoothing window
        smoothing_window = st.slider(
            "📈 Smoothing Window",
            min_value=1,
            max_value=50,
            value=10
        )
        
        st.divider()
        
        # Quick links
        st.markdown("### 🔗 Quick Links")
        st.markdown("📊 [Diwali Analysis](https://ujjwalguptafullstack-esw-dashboard-ycciky.streamlit.app/)")
        st.markdown("🌐 [GCD Dashboard](https://gcd-dashboard.netlify.app/)")
    
    # Fetch data based on analysis mode
    with st.spinner("Fetching live data from ThingSpeak..."):
        if analysis_mode == "Single Node":
            df = fetch_thingspeak_data(selected_node, 2000)
            df1, df2 = df, None
        else:
            df1, df2 = fetch_both_nodes_data(2000)
    
    if analysis_mode == "Single Node":
        if df is None:
            st.error("Unable to fetch data. Please check your connection.")
            return
        
        # Single node analysis (existing logic)
        status, status_text, status_type = get_data_status(df, selected_node)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Status:** {status} {status_text}")
        with col2:
            st.markdown(f"**Total Records:** {len(df):,}")
        with col3:
            if not df.empty:
                duration = df['created_at_local'].iloc[-1] - df['created_at_local'].iloc[0]
                st.markdown(f"**Duration:** {duration.days} days")
        
        node_height = NODES_CONFIG[selected_node]['height']
        st.markdown(f"""
        <div class="info-box">
            <strong>📍 Node Information:</strong> Live data from {selected_node} ({node_height}m height) | Running continuously since mid-Nov 2025
        </div>
        """, unsafe_allow_html=True)
        
        # Filter data by time range
        filtered_df = filter_data_by_timerange(df, timerange)
        
    else:
        # Dual node comparison analysis
        if df1 is None and df2 is None:
            st.error("Unable to fetch data from both nodes. Please check your connections.")
            return
        elif df1 is None:
            st.error("Unable to fetch data from Node-1 (35m). Showing Node-2 only.")
            df, filtered_df = df2, filter_data_by_timerange(df2, timerange)
        elif df2 is None:
            st.error("Unable to fetch data from Node-2 (25m). Showing Node-1 only.")
            df, filtered_df = df1, filter_data_by_timerange(df1, timerange)
        else:
            # Both nodes available - show dual status
            status1, status_text1, _ = get_data_status(df1, "Node-1 (35m)")
            status2, status_text2, _ = get_data_status(df2, "Node-2 (25m)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Node-1 Status:** {status1}")
                st.caption(status_text1)
                if not df1.empty:
                    st.markdown(f"**Records:** {len(df1):,}")
            
            with col2:
                st.markdown(f"**Node-2 Status:** {status2}")
                st.caption(status_text2)
                if not df2.empty:
                    st.markdown(f"**Records:** {len(df2):,}")
            
            st.markdown("""
            <div class="info-box">
                <strong>📍 Dual Node Setup:</strong> Node-1 at 35m height | Node-2 at 25m height | 10m vertical separation for gradient analysis
            </div>
            """, unsafe_allow_html=True)
            
            # Filter data by time range for both nodes
            filtered_df1 = filter_data_by_timerange(df1, timerange)
            filtered_df2 = filter_data_by_timerange(df2, timerange)
            
            if filtered_df1.empty and filtered_df2.empty:
                st.warning(f"No data available for {timerange} from either node")
                return
    
    # Current readings (live metrics)
    if analysis_mode == "Single Node":
        if filtered_df.empty:
            st.warning(f"No data available for {timerange}")
            return
            
        st.header("📊 Current Readings")
        
        if not filtered_df.empty:
            latest_data = filtered_df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'Temperature_C' in latest_data:
                    st.metric(
                        "🌡️ Temperature",
                        f"{latest_data['Temperature_C']:.1f}°C",
                        delta=f"{latest_data['Temperature_C'] - filtered_df['Temperature_C'].mean():.1f}°C"
                    )
                if 'PM2_5_ugm3' in latest_data:
                    st.metric(
                        "💨 PM2.5",
                        f"{latest_data['PM2_5_ugm3']:.1f} µg/m³",
                        delta=f"{latest_data['PM2_5_ugm3'] - filtered_df['PM2_5_ugm3'].mean():.1f}"
                    )
            
            with col2:
                if 'Humidity_%' in latest_data:
                    st.metric(
                        "💧 Humidity",
                        f"{latest_data['Humidity_%']:.1f}%",
                        delta=f"{latest_data['Humidity_%'] - filtered_df['Humidity_%'].mean():.1f}%"
                    )
                if 'PM10_ugm3' in latest_data:
                    st.metric(
                        "🌫️ PM10",
                        f"{latest_data['PM10_ugm3']:.1f} µg/m³",
                        delta=f"{latest_data['PM10_ugm3'] - filtered_df['PM10_ugm3'].mean():.1f}"
                    )
            
            with col3:
                if 'CO2_ppm' in latest_data:
                    st.metric(
                        "🏭 CO₂",
                        f"{latest_data['CO2_ppm']:.0f} ppm",
                        delta=f"{latest_data['CO2_ppm'] - filtered_df['CO2_ppm'].mean():.0f}"
                    )
                if 'CO_ppm' in latest_data:
                    st.metric(
                        "☁️ CO",
                        f"{latest_data['CO_ppm']:.1f} ppm",
                        delta=f"{latest_data['CO_ppm'] - filtered_df['CO_ppm'].mean():.1f}"
                    )
            
            with col4:
                if 'NO2_ppb' in latest_data:
                    st.metric(
                        "🚗 NO₂",
                        f"{latest_data['NO2_ppb']:.1f} ppb",
                        delta=f"{latest_data['NO2_ppb'] - filtered_df['NO2_ppb'].mean():.1f}"
                    )
    
    else:  # Dual node comparison mode
        st.header("📊 Dual Node Current Readings")
        
        col_node1, col_node2 = st.columns(2)
        
        # Node 1 readings
        with col_node1:
            st.subheader("🔵 Node-1 (35m)")
            if not filtered_df1.empty:
                latest_data1 = filtered_df1.iloc[-1]
                
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    if 'Temperature_C' in latest_data1:
                        st.metric("🌡️ Temperature", f"{latest_data1['Temperature_C']:.1f}°C")
                    if 'PM2_5_ugm3' in latest_data1:
                        st.metric("💨 PM2.5", f"{latest_data1['PM2_5_ugm3']:.1f} µg/m³")
                    if 'CO2_ppm' in latest_data1:
                        st.metric("🏭 CO₂", f"{latest_data1['CO2_ppm']:.0f} ppm")
                
                with sub_col2:
                    if 'Humidity_%' in latest_data1:
                        st.metric("💧 Humidity", f"{latest_data1['Humidity_%']:.1f}%")
                    if 'PM10_ugm3' in latest_data1:
                        st.metric("🌫️ PM10", f"{latest_data1['PM10_ugm3']:.1f} µg/m³")
                    if 'NO2_ppb' in latest_data1:
                        st.metric("🚗 NO₂", f"{latest_data1['NO2_ppb']:.1f} ppb")
            else:
                st.warning("No recent data from Node-1")
        
        # Node 2 readings
        with col_node2:
            st.subheader("🔴 Node-2 (25m)")
            if not filtered_df2.empty:
                latest_data2 = filtered_df2.iloc[-1]
                
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    if 'Temperature_C' in latest_data2:
                        st.metric("🌡️ Temperature", f"{latest_data2['Temperature_C']:.1f}°C")
                    if 'PM2_5_ugm3' in latest_data2:
                        st.metric("💨 PM2.5", f"{latest_data2['PM2_5_ugm3']:.1f} µg/m³")
                    if 'CO2_ppm' in latest_data2:
                        st.metric("🏭 CO₂", f"{latest_data2['CO2_ppm']:.0f} ppm")
                
                with sub_col2:
                    if 'Humidity_%' in latest_data2:
                        st.metric("💧 Humidity", f"{latest_data2['Humidity_%']:.1f}%")
                    if 'PM10_ugm3' in latest_data2:
                        st.metric("🌫️ PM10", f"{latest_data2['PM10_ugm3']:.1f} µg/m³")
                    if 'NO2_ppb' in latest_data2:
                        st.metric("🚗 NO₂", f"{latest_data2['NO2_ppb']:.1f} ppb")
            else:
                st.warning("No recent data from Node-2")
        
        # Calculate and display current gradient
        if not filtered_df1.empty and not filtered_df2.empty:
            st.header("⚡ Current Vertical Gradient")
            latest1 = filtered_df1.iloc[-1]
            latest2 = filtered_df2.iloc[-1]
            
            gradient_cols = st.columns(len(FIELD_MAPPING))
            for i, (field, param) in enumerate(FIELD_MAPPING.items()):
                with gradient_cols[i]:
                    if param in latest1 and param in latest2:
                        val1, val2 = latest1[param], latest2[param]
                        gradient = (val1 - val2) / 10  # 35m - 25m = 10m difference
                        st.metric(
                            param.replace('_', ' ').replace('ugm3', 'µg/m³').replace('ppm', ' ppm').replace('ppb', ' ppb'),
                            f"{gradient:+.3f}/m",
                            delta=f"35m: {val1:.1f} | 25m: {val2:.1f}"
                        )
    
    # Time series visualization
    st.header("📈 Time Series Analysis")
    
    if analysis_mode == "Single Node":
        fig_ts = create_time_series_plot(filtered_df, parameter, smoothing_window)
        if fig_ts:
            st.plotly_chart(fig_ts, use_container_width=True)
    else:
        # Dual node comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dual Node Comparison")
            fig_dual = create_dual_node_comparison(filtered_df1, filtered_df2, parameter)
            if fig_dual:
                st.plotly_chart(fig_dual, use_container_width=True)
        
        with col2:
            st.subheader("📏 Current Vertical Profile")
            fig_profile = create_height_profile_plot(filtered_df1, filtered_df2, parameter)
            if fig_profile:
                st.plotly_chart(fig_profile, use_container_width=True)
        
        # Vertical gradient analysis
        st.subheader("⚡ Vertical Gradient Analysis")
        gradient_df = calculate_vertical_gradient(filtered_df1, filtered_df2, parameter, 30)
        
        if gradient_df is not None and not gradient_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_gradient = create_vertical_gradient_plot(gradient_df, parameter)
                if fig_gradient:
                    st.plotly_chart(fig_gradient, use_container_width=True)
            
            with col2:
                st.subheader("📊 Gradient Statistics")
                mean_gradient = gradient_df['gradient'].mean()
                std_gradient = gradient_df['gradient'].std()
                max_gradient = gradient_df['gradient'].max()
                min_gradient = gradient_df['gradient'].min()
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Mean Gradient", f"{mean_gradient:+.4f}/m")
                    st.metric("Max Gradient", f"{max_gradient:+.4f}/m")
                with col_b:
                    st.metric("Std Deviation", f"{std_gradient:.4f}/m")
                    st.metric("Min Gradient", f"{min_gradient:+.4f}/m")
                
                # Interpretation
                if abs(mean_gradient) < 0.001:
                    st.info("🟢 **Well-mixed conditions** - Minimal vertical stratification")
                elif mean_gradient > 0:
                    st.warning("🟡 **Surface accumulation** - Higher concentrations at 35m")
                else:
                    st.error("🔴 **Ground-level buildup** - Higher concentrations at 25m")
        else:
            st.warning("Insufficient synchronized data for gradient calculation")

    # Analysis tabs - adapt based on mode
    if analysis_mode == "Single Node":
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🕒 Diurnal Analysis",
            "🌅 Day vs Night", 
            "📅 Weekly Trends",
            "🔗 Correlations",
            "📏 Vertical Profile"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🕒 Diurnal Comparison",
            "🌅 Day vs Night",
            "📅 Weekly Trends", 
            "🔗 Correlations",
            "📏 Height Profiles",
            "⚡ Gradient Analysis"
        ])
    
    with tab1:
        if analysis_mode == "Single Node":
            st.subheader(f"Diurnal Variation - {parameter}")
            diurnal_data = calculate_diurnal_variation(filtered_df, parameter)
            
            if not diurnal_data.empty:
                fig = px.bar(
                    diurnal_data,
                    x='hour',
                    y='mean',
                    error_y='std',
                    title=f"Hourly Average {parameter}",
                    labels={'hour': 'Hour of Day', 'mean': f'Average {parameter}'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                with st.expander("📊 View Hourly Statistics"):
                    st.dataframe(diurnal_data, use_container_width=True)
        else:
            st.subheader(f"Diurnal Comparison - {parameter}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🔵 Node-1 (35m)")
                if not filtered_df1.empty:
                    diurnal_data1 = calculate_diurnal_variation(filtered_df1, parameter)
                    if not diurnal_data1.empty:
                        fig1 = px.bar(
                            diurnal_data1,
                            x='hour',
                            y='mean',
                            error_y='std',
                            title=f"Node-1: Hourly Average {parameter}",
                            labels={'hour': 'Hour of Day', 'mean': f'Average {parameter}'},
                            color_discrete_sequence=['blue']
                        )
                        fig1.update_layout(height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        with st.expander("📊 Node-1 Hourly Stats"):
                            st.dataframe(diurnal_data1, use_container_width=True)
            
            with col2:
                st.markdown("##### 🔴 Node-2 (25m)")
                if not filtered_df2.empty:
                    diurnal_data2 = calculate_diurnal_variation(filtered_df2, parameter)
                    if not diurnal_data2.empty:
                        fig2 = px.bar(
                            diurnal_data2,
                            x='hour',
                            y='mean',
                            error_y='std',
                            title=f"Node-2: Hourly Average {parameter}",
                            labels={'hour': 'Hour of Day', 'mean': f'Average {parameter}'},
                            color_discrete_sequence=['red']
                        )
                        fig2.update_layout(height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        with st.expander("📊 Node-2 Hourly Stats"):
                            st.dataframe(diurnal_data2, use_container_width=True)
    
    with tab2:
        st.subheader(f"Day vs Night Comparison - {parameter}")
        day_night_stats = calculate_day_night_comparison(filtered_df, parameter)
        
        if day_night_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "🌅 Day Average (6AM-6PM)",
                    f"{day_night_stats['day_mean']:.2f}",
                    delta=f"σ = {day_night_stats['day_std']:.2f}"
                )
                st.caption(f"Based on {day_night_stats['day_count']} readings")
            
            with col2:
                st.metric(
                    "🌙 Night Average (6PM-6AM)",
                    f"{day_night_stats['night_mean']:.2f}",
                    delta=f"σ = {day_night_stats['night_std']:.2f}"
                )
                st.caption(f"Based on {day_night_stats['night_count']} readings")
            
            # Comparison chart
            fig = go.Figure(data=[
                go.Bar(name='Day', x=['Mean', 'Std Dev'], y=[day_night_stats['day_mean'], day_night_stats['day_std']]),
                go.Bar(name='Night', x=['Mean', 'Std Dev'], y=[day_night_stats['night_mean'], day_night_stats['night_std']])
            ])
            fig.update_layout(title=f"Day vs Night Statistics - {parameter}", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader(f"Weekly Trends - {parameter}")
        weekly_data = calculate_weekly_trends(filtered_df, parameter)
        
        if not weekly_data.empty:
            fig = px.bar(
                weekly_data,
                x='day_of_week',
                y='mean',
                error_y='std',
                title=f"Average {parameter} by Day of Week",
                labels={'day_of_week': 'Day of Week', 'mean': f'Average {parameter}'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            with st.expander("📊 View Weekly Statistics"):
                st.dataframe(weekly_data, use_container_width=True)
    
    with tab4:
        st.subheader("Parameter Correlation Matrix")
        corr_fig = create_correlation_heatmap(filtered_df)
        
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # Correlation insights
        if not filtered_df.empty:
            numeric_cols = list(FIELD_MAPPING.values())
            available_cols = [col for col in numeric_cols if col in filtered_df.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = filtered_df[available_cols].corr()
                
                st.subheader("🔍 Correlation Insights")
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        corr_pairs.append({
                            'Parameter 1': corr_matrix.columns[i],
                            'Parameter 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
                
                st.dataframe(corr_df.head(10), use_container_width=True)
    
    with tab5:
        st.subheader("Simulated Vertical Profile")
        st.info("This demonstrates potential vertical dispersion analysis with multi-height nodes")
        
        vertical_fig = create_vertical_profile_simulation(filtered_df, parameter)
        if vertical_fig:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(vertical_fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                **🎯 Future Multi-Node Setup:**
                - **2m**: Surface level
                - **12m**: Current node ⭐
                - **25m**: Mid-height
                - **50m**: Upper level
                
                **📊 Analysis Potential:**
                - Vertical gradients (Δc/Δh)
                - Mixing height variation
                - Inversion layer detection
                - Dispersion coefficients
                """)
        
        # Mock gradient calculation
        if not filtered_df.empty:
            current_val = filtered_df[parameter].iloc[-1]
            st.subheader("📊 Simulated Gradient Analysis")
            
            gradients = {
                "Surface to 12m": f"{(current_val * 0.3) / 10:.3f} units/m",
                "12m to 25m": f"{(current_val * -0.2) / 13:.3f} units/m",
                "25m to 50m": f"{(current_val * -0.2) / 25:.3f} units/m"
            }
            
            for layer, gradient in gradients.items():
                st.metric(f"Gradient: {layer}", gradient)
    
    # Export functionality
    st.header("📄 Export & Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Summary Report"):
            # Create summary statistics
            summary_stats = {}
            for param in FIELD_MAPPING.values():
                if param in filtered_df.columns:
                    summary_stats[param] = {
                        'Mean': filtered_df[param].mean(),
                        'Std': filtered_df[param].std(),
                        'Min': filtered_df[param].min(),
                        'Max': filtered_df[param].max(),
                        'Count': len(filtered_df[param].dropna())
                    }
            
            summary_df = pd.DataFrame(summary_stats).T
            st.subheader(f"Summary Statistics - {timerange}")
            st.dataframe(summary_df, use_container_width=True)
    
    with col2:
        if st.button("💾 Download Data CSV"):
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"pollution_data_{timerange.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col3:
        st.markdown("**🔄 Last Updated:**")
        st.caption(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    
    # Auto-refresh mechanism
    if auto_refresh:
        # Set up periodic refresh every 10 minutes
        st.markdown("""
        <script>
        setTimeout(function(){
            window.location.reload(1);
        }, 600000); // 10 minutes
        </script>
        """, unsafe_allow_html=True)
        
        st.info("🔄 Auto-refresh enabled (10 minutes)")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌍 Vertical Pollution Dispersion Analysis Dashboard | 
    Node-1 Data via ThingSpeak Channel 3111437 | 
    Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()