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

# Constants
THINGSPEAK_CHANNEL_ID = "3111437"
THINGSPEAK_API_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"

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
def fetch_thingspeak_data(results=2000):
    """Fetch data from ThingSpeak API"""
    try:
        params = {'results': results}
        response = requests.get(THINGSPEAK_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'feeds' not in data or not data['feeds']:
            st.error("No data found in ThingSpeak channel")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data['feeds'])
        
        # Parse timestamp and handle timezone properly
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df['created_at_local'] = df['created_at'] + pd.Timedelta(hours=5, minutes=30)  # Convert to IST
        # Remove timezone info to avoid comparison issues
        df['created_at_local'] = df['created_at_local'].dt.tz_localize(None)
        
        # Rename fields and convert to numeric
        for field, name in FIELD_MAPPING.items():
            if field in df.columns:
                df[name] = pd.to_numeric(df[field], errors='coerce')
        
        # Drop original field columns and NaN rows
        df = df.drop(columns=[f'field{i}' for i in range(1, 9) if f'field{i}' in df.columns])
        df = df.dropna(subset=list(FIELD_MAPPING.values()))
        
        return df.sort_values('created_at')
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def get_data_status(df):
    """Get current data collection status"""
    if df is None or df.empty:
        return "❌ No Data", "No data available", "danger"
    
    last_update = df['created_at_local'].iloc[-1]
    
    # Handle timezone-aware comparison
    if hasattr(last_update, 'tz') and last_update.tz is not None:
        # If data has timezone info, use it
        current_time = pd.Timestamp.now(tz=last_update.tz)
        time_diff = current_time - last_update
    else:
        # If no timezone info, treat as naive datetime
        current_time = pd.Timestamp.now().tz_localize(None)
        if hasattr(last_update, 'tz_localize'):
            last_update = last_update.tz_localize(None)
        time_diff = current_time - last_update
    
    # Convert to timedelta if it's not already
    if hasattr(time_diff, 'total_seconds'):
        time_diff_seconds = time_diff.total_seconds()
    else:
        time_diff_seconds = time_diff.total_seconds()
    
    if time_diff_seconds < 300:  # 5 minutes
        return "🟢 Live", f"Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S IST')}", "success"
    elif time_diff_seconds < 3600:  # 1 hour
        return "🟡 Recent", f"Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S IST')}", "warning"
    else:
        return "🔴 Offline", f"Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S IST')}", "danger"

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
    
    # Fetch data
    with st.spinner("Fetching live data from ThingSpeak..."):
        df = fetch_thingspeak_data(2000)
    
    if df is None:
        st.error("Unable to fetch data. Please check your connection.")
        return
    
    # Data status
    status, status_text, status_type = get_data_status(df)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Status:** {status} {status_text}")
    with col2:
        st.markdown(f"**Total Records:** {len(df):,}")
    with col3:
        if not df.empty:
            duration = df['created_at_local'].iloc[-1] - df['created_at_local'].iloc[0]
            st.markdown(f"**Duration:** {duration.days} days")
    
    st.markdown("""
    <div class="info-box">
        <strong>📍 Node Information:</strong> Live data from Node-1 (~12 m height) | Running continuously since mid-Nov 2025
    </div>
    """, unsafe_allow_html=True)
    
    # Filter data by time range
    filtered_df = filter_data_by_timerange(df, timerange)
    
    if filtered_df.empty:
        st.warning(f"No data available for {timerange}")
        return
    
    # Current readings (live metrics)
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
    
    # Time series visualization
    st.header("📈 Time Series Analysis")
    
    fig_ts = create_time_series_plot(filtered_df, parameter, smoothing_window)
    if fig_ts:
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕒 Diurnal Analysis",
        "🌅 Day vs Night",
        "📅 Weekly Trends",
        "🔗 Correlations",
        "📏 Vertical Profile"
    ])
    
    with tab1:
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