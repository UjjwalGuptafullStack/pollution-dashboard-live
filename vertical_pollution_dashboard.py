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
        'url': 'https://api.thingspeak.com/channels/3111437/feeds.json',
        'field_mapping': {
            'field1': 'Temperature_C',
            'field2': 'Humidity_%',
            'field3': 'PM2_5_ugm3',
            'field4': 'PM10_ugm3',
            'field5': 'CO2_ppm',
            'field6': 'CO_ppm',
            'field7': 'NO2_ppb'
        }
    },
    'Node-2 (25m)': {
        'channel_id': '2839248',
        'api_key': 'OWWYZK5OXTZBC65U',
        'height': 25,
        'url': 'https://api.thingspeak.com/channels/2839248/feeds.json',
        'field_mapping': {
            'field1': 'Temperature_C',
            'field2': 'Humidity_%',
            'field3': 'CO2_ppm',
            'field4': 'CO_ppm',
            'field5': 'Temperature_C_2',  # Secondary temperature sensor
            'field6': 'Humidity_%_2',    # Secondary humidity sensor
            'field7': 'NO2_ppb'
        }
    }
}

# Combined field mapping for display purposes (Node-1 fields as reference)
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
        
        # Rename fields and convert to numeric using node-specific mapping
        node_field_mapping = node_config['field_mapping']
        for field, name in node_field_mapping.items():
            if field in df.columns:
                df[name] = pd.to_numeric(df[field], errors='coerce')
        
        # Drop original field columns and NaN rows
        df = df.drop(columns=[f'field{i}' for i in range(1, 9) if f'field{i}' in df.columns])
        
        # Only drop NaN rows for fields that actually exist in this node
        available_fields = [name for name in node_field_mapping.values() if name in df.columns]
        if available_fields:
            df = df.dropna(subset=available_fields)
        
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

def get_available_parameters(node_name):
    """Get available parameters for a specific node"""
    if node_name in NODES_CONFIG:
        return list(NODES_CONFIG[node_name]['field_mapping'].values())
    return list(FIELD_MAPPING.values())

def get_common_parameters(df1, df2):
    """Get parameters that are common between both nodes"""
    if df1 is None or df2 is None:
        return []
    
    params1 = set(df1.columns)
    params2 = set(df2.columns)
    
    # Common parameters (excluding metadata columns)
    exclude_cols = {'created_at', 'created_at_local', 'node_name', 'height_m', 'entry_id'}
    common = (params1 & params2) - exclude_cols
    
    return list(common)

def create_parameter_info_display():
    """Create an informative display about parameter availability"""
    st.subheader("📊 Node Parameter Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🔵 Node-1 (35m) - Full Environmental Suite")
        node1_params = get_available_parameters('Node-1 (35m)')
        for param in node1_params:
            if 'PM' in param:
                st.markdown(f"• 💨 **{param}** - Particulate Matter")
            elif 'Temperature' in param:
                st.markdown(f"• 🌡️ **{param}** - Temperature")
            elif 'Humidity' in param:
                st.markdown(f"• 💧 **{param}** - Humidity")
            elif 'CO2' in param:
                st.markdown(f"• 🏭 **{param}** - Carbon Dioxide")
            elif 'CO_ppm' in param:
                st.markdown(f"• ☁️ **{param}** - Carbon Monoxide")
            elif 'NO2' in param:
                st.markdown(f"• 🚗 **{param}** - Nitrogen Dioxide")
    
    with col2:
        st.markdown("##### 🔴 Node-2 (25m) - Gas Monitoring Focus")
        node2_params = get_available_parameters('Node-2 (25m)')
        for param in node2_params:
            if 'Temperature' in param and '_2' in param:
                st.markdown(f"• 🌡️ **{param}** - Secondary Temperature")
            elif 'Temperature' in param:
                st.markdown(f"• 🌡️ **{param}** - Primary Temperature")
            elif 'Humidity' in param and '_2' in param:
                st.markdown(f"• 💧 **{param}** - Secondary Humidity")
            elif 'Humidity' in param:
                st.markdown(f"• 💧 **{param}** - Primary Humidity")
            elif 'CO2' in param:
                st.markdown(f"• 🏭 **{param}** - Carbon Dioxide")
            elif 'CO_ppm' in param:
                st.markdown(f"• ☁️ **{param}** - Carbon Monoxide")
            elif 'NO2' in param:
                st.markdown(f"• 🚗 **{param}** - Nitrogen Dioxide")
    
    # Common parameters info
    st.info("📊 **Common Parameters for Comparison**: Temperature, Humidity, CO₂, CO, NO₂")
    st.warning("⚠️ **Note**: PM2.5 and PM10 are only available on Node-1 (35m). Vertical gradient analysis will focus on common parameters.")

def calculate_vertical_gradient(df1, df2, parameter, time_window_minutes=30):
    """Calculate vertical gradient between two nodes"""
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return None
    
    # Check if parameter exists in both dataframes
    if parameter not in df1.columns or parameter not in df2.columns:
        return None
    
    # Synchronize timestamps (within time window)
    gradients = []
    
    for idx, row1 in df1.iterrows():
        # Skip if parameter value is missing
        if pd.isna(row1[parameter]):
            continue
            
        # Find closest measurement from node 2
        time_diff = abs(df2['created_at_local'] - row1['created_at_local'])
        closest_idx = time_diff.idxmin()
        
        if time_diff[closest_idx].total_seconds() <= time_window_minutes * 60:
            row2 = df2.loc[closest_idx]
            
            # Skip if parameter value is missing
            if pd.isna(row2[parameter]):
                continue
            
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
    
    # Check if parameter exists in both dataframes
    param_in_df1 = parameter in df1.columns
    param_in_df2 = parameter in df2.columns
    
    if not param_in_df1 and not param_in_df2:
        return None
    
    fig = go.Figure()
    
    # Node 1 (35m)
    if param_in_df1:
        fig.add_trace(go.Scatter(
            x=df1['created_at_local'],
            y=df1[parameter],
            mode='lines',
            name='Node-1 (35m)',
            line=dict(color='blue', width=2),
            opacity=0.8
        ))
    
    # Node 2 (25m)
    if param_in_df2:
        fig.add_trace(go.Scatter(
            x=df2['created_at_local'],
            y=df2[parameter],
            mode='lines',
            name='Node-2 (25m)',
            line=dict(color='red', width=2),
            opacity=0.8
        ))
    
    # Add annotation if parameter is missing from one node
    title_suffix = ""
    if not param_in_df1:
        title_suffix = " (Node-2 only - not available on Node-1)"
    elif not param_in_df2:
        title_suffix = " (Node-1 only - not available on Node-2)"
    
    fig.update_layout(
        title=f"Dual Node Comparison - {parameter}{title_suffix}",
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

def create_comprehensive_multi_parameter_plot(df1, df2, timerange="All Time"):
    """Create comprehensive multi-parameter visualization for both nodes"""
    if df1 is None and df2 is None:
        return None
    
    # Create subplot with secondary y-axis for different parameter scales
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Temperature (°C)', 'Humidity (%)', 
            'PM2.5 & PM10 (µg/m³)', 'CO₂ (ppm)',
            'CO (ppm)', 'NO₂ (ppb)',
            'Vertical Gradients', 'Data Availability'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # Color scheme
    node1_color = '#1f77b4'  # Blue
    node2_color = '#d62728'  # Red
    
    # Row 1: Temperature and Humidity
    if df1 is not None and not df1.empty and 'Temperature_C' in df1.columns:
        fig.add_trace(
            go.Scatter(x=df1['created_at_local'], y=df1['Temperature_C'], 
                      name='Node-1 (35m)', line=dict(color=node1_color, width=2),
                      showlegend=True),
            row=1, col=1
        )
    
    if df2 is not None and not df2.empty and 'Temperature_C' in df2.columns:
        fig.add_trace(
            go.Scatter(x=df2['created_at_local'], y=df2['Temperature_C'], 
                      name='Node-2 (25m)', line=dict(color=node2_color, width=2),
                      showlegend=True),
            row=1, col=1
        )
        
    if df1 is not None and not df1.empty and 'Humidity_%' in df1.columns:
        fig.add_trace(
            go.Scatter(x=df1['created_at_local'], y=df1['Humidity_%'], 
                      name='Node-1 (35m)', line=dict(color=node1_color, width=2),
                      showlegend=False),
            row=1, col=2
        )
    
    if df2 is not None and not df2.empty and 'Humidity_%' in df2.columns:
        fig.add_trace(
            go.Scatter(x=df2['created_at_local'], y=df2['Humidity_%'], 
                      name='Node-2 (25m)', line=dict(color=node2_color, width=2),
                      showlegend=False),
            row=1, col=2
        )
    
    # Row 2: PM2.5 & PM10, CO₂
    # PM data - only available on Node-1
    if df1 is not None and not df1.empty:
        if 'PM2_5_ugm3' in df1.columns:
            fig.add_trace(
                go.Scatter(x=df1['created_at_local'], y=df1['PM2_5_ugm3'], 
                          name='PM2.5-35m', line=dict(color=node1_color, dash='solid'),
                          showlegend=False),
                row=2, col=1
            )
        if 'PM10_ugm3' in df1.columns:
            fig.add_trace(
                go.Scatter(x=df1['created_at_local'], y=df1['PM10_ugm3'], 
                          name='PM10-35m', line=dict(color=node1_color, dash='dash'),
                          showlegend=False),
                row=2, col=1
            )
        if 'CO2_ppm' in df1.columns:
            fig.add_trace(
                go.Scatter(x=df1['created_at_local'], y=df1['CO2_ppm'], 
                          name='CO₂-35m', line=dict(color=node1_color, width=2),
                          showlegend=False),
                row=2, col=2
            )
    
    if df2 is not None and not df2.empty:
        # Note: PM sensors not available on Node-2
        if 'CO2_ppm' in df2.columns:
            fig.add_trace(
                go.Scatter(x=df2['created_at_local'], y=df2['CO2_ppm'], 
                          name='CO₂-25m', line=dict(color=node2_color, width=2),
                          showlegend=False),
                row=2, col=2
            )
    
    # Row 3: CO, NO₂
    if df1 is not None and not df1.empty and 'CO_ppm' in df1.columns:
        fig.add_trace(
            go.Scatter(x=df1['created_at_local'], y=df1['CO_ppm'], 
                      name='CO-35m', line=dict(color=node1_color, width=2),
                      showlegend=False),
            row=3, col=1
        )
    
    if df2 is not None and not df2.empty and 'CO_ppm' in df2.columns:
        fig.add_trace(
            go.Scatter(x=df2['created_at_local'], y=df2['CO_ppm'], 
                      name='CO-25m', line=dict(color=node2_color, width=2),
                      showlegend=False),
            row=3, col=1
        )
        
    if df1 is not None and not df1.empty and 'NO2_ppb' in df1.columns:
        fig.add_trace(
            go.Scatter(x=df1['created_at_local'], y=df1['NO2_ppb'], 
                      name='NO₂-35m', line=dict(color=node1_color, width=2),
                      showlegend=False),
            row=3, col=2
        )
    
    if df2 is not None and not df2.empty and 'NO2_ppb' in df2.columns:
        fig.add_trace(
            go.Scatter(x=df2['created_at_local'], y=df2['NO2_ppb'], 
                      name='NO₂-25m', line=dict(color=node2_color, width=2),
                      showlegend=False),
            row=3, col=2
        )
    
    # Row 4: Vertical Gradients and Data Availability
    if df1 is not None and df2 is not None and not df1.empty and not df2.empty:
        # Calculate gradients for common parameters only
        gradient_data = []
        
        for idx, row1 in df1.iterrows():
            time_diff = abs(df2['created_at_local'] - row1['created_at_local'])
            if len(time_diff) == 0:
                continue
                
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx].total_seconds() <= 30 * 60:  # 30 minutes window
                row2 = df2.loc[closest_idx]
                
                # Calculate gradients for common parameters only
                gradient_entry = {'time': row1['created_at_local']}
                
                if 'Temperature_C' in df1.columns and 'Temperature_C' in df2.columns:
                    temp_gradient = (row1['Temperature_C'] - row2['Temperature_C']) / 10
                    gradient_entry['temp_grad'] = temp_gradient
                
                if 'CO2_ppm' in df1.columns and 'CO2_ppm' in df2.columns:
                    co2_gradient = (row1['CO2_ppm'] - row2['CO2_ppm']) / 10
                    gradient_entry['co2_grad'] = co2_gradient
                
                if 'CO_ppm' in df1.columns and 'CO_ppm' in df2.columns:
                    co_gradient = (row1['CO_ppm'] - row2['CO_ppm']) / 10
                    gradient_entry['co_grad'] = co_gradient
                    
                gradient_data.append(gradient_entry)
        
        if gradient_data:
            grad_df = pd.DataFrame(gradient_data)
            
            # Add temperature gradient if available
            if 'temp_grad' in grad_df.columns:
                fig.add_trace(
                    go.Scatter(x=grad_df['time'], y=grad_df['temp_grad'], 
                              name='Temp Gradient', line=dict(color='orange', width=2),
                              showlegend=False),
                    row=4, col=1
                )
            
            # Add CO2 gradient if available
            if 'co2_grad' in grad_df.columns:
                fig.add_trace(
                    go.Scatter(x=grad_df['time'], y=grad_df['co2_grad'], 
                              name='CO₂ Gradient', line=dict(color='green', width=2),
                              showlegend=False),
                    row=4, col=1
                )
            
            # Add zero line for gradients
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=4, col=1)
    
    # Data availability heatmap (simplified)
    if df1 is not None and not df1.empty:
        data_availability_1 = df1.groupby(df1['created_at_local'].dt.date).size()
        fig.add_trace(
            go.Scatter(x=data_availability_1.index, y=data_availability_1.values, 
                      mode='markers+lines', name='Node-1 Data Points', 
                      marker=dict(color=node1_color, size=6),
                      showlegend=False),
            row=4, col=2
        )
    
    if df2 is not None and not df2.empty:
        data_availability_2 = df2.groupby(df2['created_at_local'].dt.date).size()
        fig.add_trace(
            go.Scatter(x=data_availability_2.index, y=data_availability_2.values, 
                      mode='markers+lines', name='Node-2 Data Points', 
                      marker=dict(color=node2_color, size=6),
                      showlegend=False),
            row=4, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title={
            'text': f'🌍 Comprehensive Multi-Parameter Analysis: Dual-Node Vertical Monitoring<br><sub>Node-1 (35m) vs Node-2 (25m) | {timerange}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update x-axis labels
    for i in range(1, 5):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Time (IST)", row=i, col=j, showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)
    
    # Specific y-axis labels
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_yaxes(title_text="µg/m³", row=2, col=1)
    fig.update_yaxes(title_text="ppm", row=2, col=2)
    fig.update_yaxes(title_text="ppm", row=3, col=1)
    fig.update_yaxes(title_text="ppb", row=3, col=2)
    fig.update_yaxes(title_text="units/m", row=4, col=1)
    fig.update_yaxes(title_text="data points/day", row=4, col=2)
    
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
        
        # Parameter selection with availability info
        if analysis_mode == "Single Node":
            available_params = get_available_parameters(selected_node) if 'selected_node' in locals() else list(FIELD_MAPPING.values())
        else:
            # Dual node mode - show parameter availability
            common_params = get_common_parameters()
            node1_only = [p for p in get_available_parameters(1) if p not in common_params]
            node2_only = [p for p in get_available_parameters(2) if p not in common_params]
            
            # Show parameter availability info in expander
            with st.expander("📊 Parameter Availability Info", expanded=False):
                if common_params:
                    st.success(f"🔗 **Common parameters (both nodes):** {', '.join(common_params)}")
                if node1_only:
                    st.info(f"🔵 **Node-1 only (35m):** {', '.join(node1_only)}")
                if node2_only:
                    st.info(f"🔴 **Node-2 only (25m):** {', '.join(node2_only)}")
            
            # Combine all available parameters
            all_params = common_params + node1_only + node2_only
            available_params = all_params if all_params else list(FIELD_MAPPING.values())
        
        parameter = st.selectbox(
            "📊 Parameter",
            available_params
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
            
            # Show parameter configuration
            create_parameter_info_display()
            
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
    
    # Comprehensive Multi-Parameter Overview
    if analysis_mode == "Dual Node" and df1 is not None and df2 is not None:
        st.header("🌍 Grand Overview: Complete Multi-Parameter Analysis")
        
        # Create comprehensive plot with all available data
        with st.spinner("Generating comprehensive analysis..."):
            # Get full dataset for grand overview (more data points)
            full_df1, full_df2 = fetch_both_nodes_data(5000)  # Get more historical data
            
            if full_df1 is not None and full_df2 is not None:
                # Show data range information
                col1, col2, col3 = st.columns(3)
                with col1:
                    if not full_df1.empty:
                        start_date = full_df1['created_at_local'].min().strftime('%Y-%m-%d')
                        end_date = full_df1['created_at_local'].max().strftime('%Y-%m-%d')
                        st.info(f"**Node-1 Data Range:** {start_date} to {end_date}")
                
                with col2:
                    if not full_df2.empty:
                        start_date = full_df2['created_at_local'].min().strftime('%Y-%m-%d')
                        end_date = full_df2['created_at_local'].max().strftime('%Y-%m-%d')
                        st.info(f"**Node-2 Data Range:** {start_date} to {end_date}")
                
                with col3:
                    total_hours = 0
                    if not full_df1.empty and not full_df2.empty:
                        total_duration = max(full_df1['created_at_local'].max(), full_df2['created_at_local'].max()) - \
                                       min(full_df1['created_at_local'].min(), full_df2['created_at_local'].min())
                        total_hours = total_duration.total_seconds() / 3600
                    st.success(f"**Total Monitoring:** {total_hours:.1f} hours")
                
                # Time range selector for comprehensive view
                comp_timerange = st.selectbox(
                    "📊 Comprehensive Analysis Time Range:",
                    ["All Available Data", "Last 7 Days", "Last 30 Days", "Last 3 Months"],
                    index=0
                )
                
                # Filter data based on comprehensive time range
                if comp_timerange == "Last 7 Days":
                    comp_df1 = filter_data_by_timerange(full_df1, "Last 7 days")
                    comp_df2 = filter_data_by_timerange(full_df2, "Last 7 days")
                elif comp_timerange == "Last 30 Days":
                    comp_df1 = filter_data_by_timerange(full_df1, "Last 30 days")
                    comp_df2 = filter_data_by_timerange(full_df2, "Last 30 days")
                elif comp_timerange == "Last 3 Months":
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=90)
                    comp_df1 = full_df1[full_df1['created_at_local'] > cutoff_date] if not full_df1.empty else full_df1
                    comp_df2 = full_df2[full_df2['created_at_local'] > cutoff_date] if not full_df2.empty else full_df2
                else:  # All Available Data
                    comp_df1, comp_df2 = full_df1, full_df2
                
                # Generate comprehensive plot
                fig_comprehensive = create_comprehensive_multi_parameter_plot(comp_df1, comp_df2, comp_timerange)
                if fig_comprehensive:
                    st.plotly_chart(fig_comprehensive, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📊 Comprehensive Data Summary")
                    
                    if not comp_df1.empty and not comp_df2.empty:
                        summary_cols = st.columns(4)
                        
                        # Data quality metrics
                        with summary_cols[0]:
                            st.metric("Node-1 Data Points", f"{len(comp_df1):,}")
                            uptime_1 = len(comp_df1) / (total_hours * 3) * 100 if total_hours > 0 else 0  # Assuming ~3 readings per hour
                            st.metric("Node-1 Uptime", f"{uptime_1:.1f}%")
                        
                        with summary_cols[1]:
                            st.metric("Node-2 Data Points", f"{len(comp_df2):,}")
                            uptime_2 = len(comp_df2) / (total_hours * 3) * 100 if total_hours > 0 else 0
                            st.metric("Node-2 Uptime", f"{uptime_2:.1f}%")
                        
                        with summary_cols[2]:
                            # Average gradient calculation
                            gradients = []
                            for param in ['PM2_5_ugm3', 'Temperature_C', 'Humidity_%']:
                                if param in comp_df1.columns and param in comp_df2.columns:
                                    avg_diff = (comp_df1[param].mean() - comp_df2[param].mean()) / 10
                                    gradients.append(avg_diff)
                            
                            if gradients:
                                avg_gradient = np.mean([abs(g) for g in gradients])
                                st.metric("Avg Vertical Gradient", f"{avg_gradient:.3f} units/m")
                            
                            # Synchronization quality
                            sync_quality = min(len(comp_df1), len(comp_df2)) / max(len(comp_df1), len(comp_df2)) * 100
                            st.metric("Data Synchronization", f"{sync_quality:.1f}%")
                        
                        with summary_cols[3]:
                            # Environmental conditions summary
                            if 'Temperature_C' in comp_df1.columns and 'Temperature_C' in comp_df2.columns:
                                temp_range_1 = comp_df1['Temperature_C'].max() - comp_df1['Temperature_C'].min()
                                temp_range_2 = comp_df2['Temperature_C'].max() - comp_df2['Temperature_C'].min()
                                avg_temp_range = (temp_range_1 + temp_range_2) / 2
                                st.metric("Avg Temperature Range", f"{avg_temp_range:.1f}°C")
                            
                            if 'PM2_5_ugm3' in comp_df1.columns and 'PM2_5_ugm3' in comp_df2.columns:
                                max_pm25 = max(comp_df1['PM2_5_ugm3'].max(), comp_df2['PM2_5_ugm3'].max())
                                st.metric("Peak PM2.5", f"{max_pm25:.1f} µg/m³")
                
                # Download option for comprehensive data
                col_download1, col_download2, col_download3 = st.columns(3)
                
                with col_download1:
                    if st.button("📥 Download Comprehensive Report"):
                        # Create comprehensive CSV
                        if not comp_df1.empty and not comp_df2.empty:
                            # Merge data on timestamp for comparison
                            merged_data = pd.merge_asof(
                                comp_df1.sort_values('created_at_local'),
                                comp_df2.sort_values('created_at_local'),
                                on='created_at_local',
                                suffixes=('_35m', '_25m'),
                                tolerance=pd.Timedelta('30 minutes')
                            )
                            
                            csv_data = merged_data.to_csv(index=False)
                            st.download_button(
                                label="Download Merged Dataset",
                                data=csv_data,
                                file_name=f"comprehensive_dual_node_data_{comp_timerange.replace(' ', '_').lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                
                with col_download2:
                    st.info("💡 **Tip:** Use the comprehensive view to identify long-term trends and patterns in vertical pollution dispersion.")
                
                with col_download3:
                    if st.button("🔄 Refresh Comprehensive Data"):
                        st.cache_data.clear()
                        st.rerun()
            
            else:
                st.error("Unable to fetch comprehensive data for analysis.")

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
        
        if analysis_mode == "Single Node":
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
        else:
            # Dual node day vs night comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🔵 Node-1 (35m) - Day vs Night")
                if not filtered_df1.empty:
                    day_night_stats1 = calculate_day_night_comparison(filtered_df1, parameter)
                    if day_night_stats1:
                        sub_col1, sub_col2 = st.columns(2)
                        with sub_col1:
                            st.metric("🌅 Day", f"{day_night_stats1['day_mean']:.2f}", f"σ={day_night_stats1['day_std']:.2f}")
                        with sub_col2:
                            st.metric("🌙 Night", f"{day_night_stats1['night_mean']:.2f}", f"σ={day_night_stats1['night_std']:.2f}")
                        
                        # Comparison chart for Node 1
                        fig1 = go.Figure(data=[
                            go.Bar(name='Day', x=['Mean'], y=[day_night_stats1['day_mean']], marker_color='orange'),
                            go.Bar(name='Night', x=['Mean'], y=[day_night_stats1['night_mean']], marker_color='navy')
                        ])
                        fig1.update_layout(title=f"Node-1: Day vs Night - {parameter}", height=300)
                        st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("##### 🔴 Node-2 (25m) - Day vs Night")
                if not filtered_df2.empty:
                    day_night_stats2 = calculate_day_night_comparison(filtered_df2, parameter)
                    if day_night_stats2:
                        sub_col1, sub_col2 = st.columns(2)
                        with sub_col1:
                            st.metric("🌅 Day", f"{day_night_stats2['day_mean']:.2f}", f"σ={day_night_stats2['day_std']:.2f}")
                        with sub_col2:
                            st.metric("🌙 Night", f"{day_night_stats2['night_mean']:.2f}", f"σ={day_night_stats2['night_std']:.2f}")
                        
                        # Comparison chart for Node 2
                        fig2 = go.Figure(data=[
                            go.Bar(name='Day', x=['Mean'], y=[day_night_stats2['day_mean']], marker_color='orange'),
                            go.Bar(name='Night', x=['Mean'], y=[day_night_stats2['night_mean']], marker_color='navy')
                        ])
                        fig2.update_layout(title=f"Node-2: Day vs Night - {parameter}", height=300)
                        st.plotly_chart(fig2, use_container_width=True)
    
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
        if analysis_mode == "Single Node":
            st.subheader("Simulated Vertical Profile")
            st.info("This demonstrates potential vertical dispersion analysis with multi-height nodes")
            
            vertical_fig = create_vertical_profile_simulation(filtered_df, parameter)
            if vertical_fig:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(vertical_fig, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    **🎯 Current Multi-Node Setup:**
                    - **25m**: Node-2 ⭐
                    - **35m**: Node-1 ⭐
                    
                    **📊 Analysis Capabilities:**
                    - Real vertical gradients (Δc/Δh)
                    - Mixing height variation
                    - Inversion layer detection
                    - Dispersion coefficients
                    """)
            
            # Mock gradient calculation
            if not filtered_df.empty:
                current_val = filtered_df[parameter].iloc[-1]
                st.subheader("📊 Simulated Gradient Analysis")
                
                gradients = {
                    "Surface to 25m": f"{(current_val * 0.3) / 25:.3f} units/m",
                    "25m to 35m": f"{(current_val * -0.1) / 10:.3f} units/m",
                    "35m to 50m": f"{(current_val * -0.2) / 15:.3f} units/m"
                }
                
                for layer, gradient in gradients.items():
                    st.metric(f"Gradient: {layer}", gradient)
        else:
            st.subheader("Real Height Profiles Analysis")
            
            if not filtered_df1.empty and not filtered_df2.empty:
                # Time selector for profile analysis
                profile_time = st.selectbox(
                    "Select time for profile analysis:",
                    ["Latest", "1 hour ago", "6 hours ago", "1 day ago"]
                )
                
                if profile_time == "Latest":
                    time_point = None
                elif profile_time == "1 hour ago":
                    time_point = pd.Timestamp.now() - pd.Timedelta(hours=1)
                elif profile_time == "6 hours ago":
                    time_point = pd.Timestamp.now() - pd.Timedelta(hours=6)
                else:  # 1 day ago
                    time_point = pd.Timestamp.now() - pd.Timedelta(days=1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    profile_fig = create_height_profile_plot(filtered_df1, filtered_df2, parameter, time_point)
                    if profile_fig:
                        st.plotly_chart(profile_fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Profile Statistics")
                    
                    # Get values for the selected time
                    if time_point is None:
                        val1 = filtered_df1[parameter].iloc[-1] if not filtered_df1.empty else 0
                        val2 = filtered_df2[parameter].iloc[-1] if not filtered_df2.empty else 0
                    else:
                        idx1 = (abs(filtered_df1['created_at_local'] - time_point)).idxmin() if not filtered_df1.empty else None
                        idx2 = (abs(filtered_df2['created_at_local'] - time_point)).idxmin() if not filtered_df2.empty else None
                        val1 = filtered_df1.loc[idx1, parameter] if idx1 is not None else 0
                        val2 = filtered_df2.loc[idx2, parameter] if idx2 is not None else 0
                    
                    gradient_val = (val1 - val2) / 10  # 35m - 25m = 10m
                    
                    st.metric("35m Value", f"{val1:.2f}")
                    st.metric("25m Value", f"{val2:.2f}")
                    st.metric("Vertical Gradient", f"{gradient_val:+.4f}/m")
                    
                    if gradient_val > 0.01:
                        st.success("🔼 **Higher at 35m** - Possible elevated source or inversion")
                    elif gradient_val < -0.01:
                        st.warning("🔽 **Higher at 25m** - Surface source or good mixing")
                    else:
                        st.info("⚖️ **Well mixed** - Minimal vertical stratification")
            else:
                st.warning("Need data from both nodes for profile analysis")

    if analysis_mode == "Dual Node":
        with tab6:
            st.subheader("⚡ Advanced Gradient Analysis")
            
            if not filtered_df1.empty and not filtered_df2.empty:
                # Parameter selector for gradient analysis - only common parameters
                common_params = get_common_parameters()
                if common_params:
                    gradient_params = st.multiselect(
                        "Select parameters for gradient analysis:",
                        common_params,
                        default=common_params[:2] if len(common_params) >= 2 else common_params
                    )
                else:
                    st.warning("No common parameters available for gradient analysis between the two nodes.")
                    gradient_params = []
                
                if gradient_params:
                    # Calculate gradients for multiple parameters
                    fig_multi_gradient = go.Figure()
                    
                    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                    
                    for i, param in enumerate(gradient_params):
                        gradient_df = calculate_vertical_gradient(filtered_df1, filtered_df2, param, 30)
                        
                        if gradient_df is not None and not gradient_df.empty:
                            fig_multi_gradient.add_trace(
                                go.Scatter(
                                    x=gradient_df['timestamp'],
                                    y=gradient_df['gradient'],
                                    mode='lines',
                                    name=f'{param} Gradient',
                                    line=dict(color=colors[i % len(colors)], width=2)
                                )
                            )
                    
                    fig_multi_gradient.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_multi_gradient.update_layout(
                        title="Multi-Parameter Vertical Gradients",
                        xaxis_title="Time (IST)",
                        yaxis_title="Gradient (units/m)",
                        height=500
                    )
                    
                    st.plotly_chart(fig_multi_gradient, use_container_width=True)
                    
                    # Gradient statistics table
                    st.subheader("📊 Gradient Statistics Summary")
                    
                    gradient_stats = []
                    for param in gradient_params:
                        gradient_df = calculate_vertical_gradient(filtered_df1, filtered_df2, param, 30)
                        if gradient_df is not None and not gradient_df.empty:
                            gradient_stats.append({
                                'Parameter': param,
                                'Mean Gradient': gradient_df['gradient'].mean(),
                                'Std Deviation': gradient_df['gradient'].std(),
                                'Max Gradient': gradient_df['gradient'].max(),
                                'Min Gradient': gradient_df['gradient'].min(),
                                'Data Points': len(gradient_df)
                            })
                    
                    if gradient_stats:
                        gradient_stats_df = pd.DataFrame(gradient_stats)
                        st.dataframe(gradient_stats_df, use_container_width=True)
                        
                        # Download gradient data
                        csv_gradients = gradient_stats_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Gradient Statistics",
                            data=csv_gradients,
                            file_name=f"gradient_statistics_{timerange.replace(' ', '_').lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("Please select parameters for gradient analysis")
            else:
                st.warning("Need data from both nodes for advanced gradient analysis")
    
    # Export functionality
    st.header("📄 Export & Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Summary Report"):
            # Create summary statistics
            if analysis_mode == "Single Node":
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
            else:
                # Dual node summary
                st.subheader(f"Dual Node Summary - {timerange}")
                
                col_sum1, col_sum2 = st.columns(2)
                
                with col_sum1:
                    st.markdown("##### 🔵 Node-1 (35m) Statistics")
                    if not filtered_df1.empty:
                        summary_stats1 = {}
                        for param in FIELD_MAPPING.values():
                            if param in filtered_df1.columns:
                                summary_stats1[param] = {
                                    'Mean': filtered_df1[param].mean(),
                                    'Std': filtered_df1[param].std(),
                                    'Min': filtered_df1[param].min(),
                                    'Max': filtered_df1[param].max()
                                }
                        summary_df1 = pd.DataFrame(summary_stats1).T
                        st.dataframe(summary_df1, use_container_width=True)
                
                with col_sum2:
                    st.markdown("##### 🔴 Node-2 (25m) Statistics")
                    if not filtered_df2.empty:
                        summary_stats2 = {}
                        for param in FIELD_MAPPING.values():
                            if param in filtered_df2.columns:
                                summary_stats2[param] = {
                                    'Mean': filtered_df2[param].mean(),
                                    'Std': filtered_df2[param].std(),
                                    'Min': filtered_df2[param].min(),
                                    'Max': filtered_df2[param].max()
                                }
                        summary_df2 = pd.DataFrame(summary_stats2).T
                        st.dataframe(summary_df2, use_container_width=True)
    
    with col2:
        if st.button("💾 Download Data CSV"):
            if analysis_mode == "Single Node":
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"pollution_data_{timerange.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                # Dual node download options
                download_option = st.selectbox(
                    "Select download option:",
                    ["Both Nodes (Separate)", "Node-1 Only", "Node-2 Only", "Merged Dataset"]
                )
                
                if download_option == "Node-1 Only" and not filtered_df1.empty:
                    csv_data = filtered_df1.to_csv(index=False)
                    filename = f"node1_35m_data_{timerange.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                elif download_option == "Node-2 Only" and not filtered_df2.empty:
                    csv_data = filtered_df2.to_csv(index=False)
                    filename = f"node2_25m_data_{timerange.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                elif download_option == "Merged Dataset" and not filtered_df1.empty and not filtered_df2.empty:
                    # Merge on timestamp
                    merged_data = pd.merge_asof(
                        filtered_df1.sort_values('created_at_local'),
                        filtered_df2.sort_values('created_at_local'),
                        on='created_at_local',
                        suffixes=('_35m', '_25m'),
                        tolerance=pd.Timedelta('30 minutes')
                    )
                    csv_data = merged_data.to_csv(index=False)
                    filename = f"merged_dual_node_data_{timerange.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                else:  # Both Nodes (Separate)
                    if not filtered_df1.empty:
                        csv_data1 = filtered_df1.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Node-1 Data",
                            data=csv_data1,
                            file_name=f"node1_35m_data_{timerange.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    if not filtered_df2.empty:
                        csv_data2 = filtered_df2.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Node-2 Data",
                            data=csv_data2,
                            file_name=f"node2_25m_data_{timerange.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    return  # Skip the single download button
                
                if download_option != "Both Nodes (Separate)":
                    st.download_button(
                        label=f"Download {download_option}",
                        data=csv_data,
                        file_name=filename,
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