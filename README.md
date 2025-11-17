# Vertical Pollution Dispersion Live Analysis Dashboard

## 🌍 Overview

This Streamlit dashboard provides real-time analysis of air quality data from a continuous monitoring node deployed at ~12m height. The dashboard fetches live data from ThingSpeak Channel 3111437 and performs comprehensive vertical pollution dispersion analysis.

## 📊 Features

### Real-time Data Collection
- Live data fetching from ThingSpeak API every 5 minutes
- Multi-parameter monitoring: Temperature, Humidity, PM2.5, PM10, CO₂, CO, NO₂
- Auto-refresh capability (10-minute intervals)
- Data collection status monitoring

### Analysis Capabilities
- **Diurnal Analysis**: Hourly averages showing daily pollution patterns
- **Day vs Night Comparison**: Comparative analysis between daytime (6 AM-6 PM) and nighttime (6 PM-6 AM)
- **Weekly Trends**: Day-of-week pollution pattern analysis
- **Correlation Analysis**: Parameter correlation matrix and insights
- **Time Series Visualization**: Interactive plots with smoothing options

### Vertical Profile Simulation
- Demonstrates potential multi-height node analysis
- Simulated vertical concentration gradients
- Framework for future multi-node deployments

### Export & Reporting
- Summary statistics generation
- CSV data export
- PDF-ready visualizations

## 🚀 Deployment

### Local Deployment

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run vertical_pollution_dashboard.py
   ```

3. **Access**: Open `http://localhost:8501` in your browser

### Cloud Deployment (Streamlit Cloud)

1. Push this repository to GitHub
2. Connect to [Streamlit Cloud](https://share.streamlit.io/)
3. Deploy directly from GitHub repository
4. Access your live dashboard URL

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "vertical_pollution_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📡 Data Source

- **ThingSpeak Channel**: 3111437
- **API Endpoint**: `https://api.thingspeak.com/channels/3111437/feeds.json`
- **Update Frequency**: ~18-20 seconds
- **Node Height**: ~12 meters
- **Location**: Continuous deployment since mid-Nov 2025

### Data Fields
- Field 1: Temperature (°C)
- Field 2: Humidity (%)
- Field 3: PM2.5 (µg/m³)
- Field 4: PM10 (µg/m³)
- Field 5: CO₂ (ppm)
- Field 6: CO (ppm)
- Field 7: NO₂ (ppb)

## 🔧 Configuration

### Time Range Options
- Last 6 hours
- Last 24 hours
- Last 7 days
- Last 30 days

### Smoothing Options
- Rolling window: 1-50 data points
- Real-time vs smoothed visualization

### Auto-refresh Settings
- Configurable auto-refresh (default: 10 minutes)
- Manual refresh capability
- Data caching (5-minute TTL)

## 🎯 Future Enhancements

### Multi-Node Integration
When additional nodes are deployed at different heights (2m, 25m, 50m), the dashboard will automatically:
- Calculate real vertical gradients (∆c/∆h)
- Detect mixing height variations
- Identify temperature inversion events
- Estimate dispersion coefficients

### Advanced Analytics
- Machine learning trend predictions
- Air quality index calculations
- Weather correlation analysis
- Event detection algorithms

## 🔗 Integration

### Related Dashboards
- **Diwali 2025 Analysis**: [https://ujjwalguptafullstack-esw-dashboard-ycciky.streamlit.app/](https://ujjwalguptafullstack-esw-dashboard-ycciky.streamlit.app/)
- **GCD Main Hub**: [https://gcd-dashboard.netlify.app/](https://gcd-dashboard.netlify.app/)

### API Integration
The dashboard can be extended to integrate with:
- Weather APIs for meteorological correlation
- Air quality standards for compliance monitoring
- Alert systems for pollution threshold breaches

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit with Plotly visualizations
- **Data Source**: ThingSpeak IoT platform
- **Processing**: Pandas for data manipulation
- **Caching**: Streamlit native caching (5-min TTL)
- **Refresh**: JavaScript-based auto-refresh

### Performance
- Optimized data fetching (2000 records max)
- Efficient correlation calculations
- Responsive design for mobile/desktop
- Progressive data loading

## 📈 Analytics Framework

### Statistical Measures
- Mean, standard deviation, min/max values
- Hourly, daily, and weekly aggregations
- Correlation coefficients and p-values
- Trend analysis and seasonality detection

### Visualization Types
- Time series plots (raw + smoothed)
- Bar charts for diurnal/weekly patterns
- Heatmaps for correlation analysis
- Scatter plots for vertical profiles
- Metric cards for live readings

## 🚨 Monitoring & Alerts

### Data Quality Monitoring
- Connection status indicators
- Last update timestamps
- Data completeness checks
- Anomaly detection flags

### Status Indicators
- 🟢 Live: Updated within 5 minutes
- 🟡 Recent: Updated within 1 hour
- 🔴 Offline: No updates for >1 hour

## 📋 Usage Instructions

1. **Select Time Range**: Choose analysis period from sidebar
2. **Choose Parameter**: Select pollutant or meteorological variable
3. **Adjust Smoothing**: Set rolling window for trend visualization
4. **Explore Tabs**: Navigate through different analysis views
5. **Export Data**: Download CSV or generate summary reports
6. **Monitor Status**: Check live data collection status

## 🎨 Customization

The dashboard theme and colors can be customized via `.streamlit/config.toml`:
- Primary color: `#667eea`
- Background: `#FFFFFF`
- Secondary background: `#f0f2f6`

## 📞 Support

For technical support or feature requests:
- Check data source: ThingSpeak Channel 3111437
- Verify network connectivity
- Review console logs for API errors
- Contact system administrator for node issues

---

**Last Updated**: November 17, 2025  
**Version**: 1.0  
**Compatibility**: Python 3.8+, Streamlit 1.28+