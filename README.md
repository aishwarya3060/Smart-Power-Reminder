# Smart Power Reminder - Green AI Desktop App

## Project Overview

Smart Power Reminder is a Green AI-powered desktop application that uses machine learning to detect idle applications and remind users to shut them down to save electricity. The application features real-time monitoring, intelligent idle detection, and power savings calculations.

## Features

### 🤖 Machine Learning Backend
- Lightweight Random Forest model for idle detection
- Real-time system monitoring and resource tracking
- Power consumption estimation and savings calculation
- Green AI principles: optimized for low energy consumption

### 🖥️ Desktop UI (Streamlit)
- Real-time dashboard showing running applications
- Active vs Idle application classification
- Power consumption monitoring and savings visualization
- Historical usage trends and analytics
- Customizable idle time thresholds and settings
- Energy-saving tips and recommendations

### 🌱 Green AI Focus
- Lightweight ML models optimized for energy efficiency
- Power consumption tracking and optimization
- Estimated savings calculations (power and cost)
- Sustainable computing recommendations

## Project Structure

```
smart_power_reminder/
├── data/
│   └── app_usage_data.csv          # Training dataset
├── models/
│   ├── idle_detection_model.pkl    # Trained ML model
│   └── scaler.pkl                  # Feature scaler
├── src/
│   ├── backend/
│   │   └── ml_backend.py           # ML model and system monitoring
│   └── frontend/
│       └── app.py                  # Streamlit dashboard application
├── requirements.txt                # Python dependencies
├── run_app.py                     # Application launcher
└── README.md                      # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Windows, macOS, or Linux operating system
- At least 4GB RAM
- Internet connection for initial setup

### Step-by-Step Installation

1. **Clone or download the project:**
   ```bash
   # If using git
   git clone <repository-url>
   cd smart_power_reminder

   # Or extract the downloaded ZIP file and navigate to the directory
   cd smart_power_reminder
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # On Windows
   python -m venv smart_power_env
   smart_power_env\Scripts\activate

   # On macOS/Linux
   python3 -m venv smart_power_env
   source smart_power_env/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import streamlit, sklearn, pandas; print('All dependencies installed successfully!')"
   ```

### Alternative Installation (using conda)

1. **Create conda environment:**
   ```bash
   conda create -n smart_power_env python=3.9
   conda activate smart_power_env
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Method 1: Using the Launcher Script (Recommended)
```bash
python run_app.py
```

### Method 2: Direct Streamlit Command
```bash
streamlit run src/frontend/app.py
```

### Method 3: From VS Code
1. Open the project folder in VS Code
2. Open terminal in VS Code
3. Run: `python run_app.py`

## Application Usage

### Initial Launch
1. The application will automatically open in your default web browser
2. The URL will typically be: `http://localhost:8501`
3. If it doesn't open automatically, copy the URL from the terminal

### Dashboard Features

#### Main Dashboard
- **Active Apps**: Shows currently running and active applications
- **Idle Apps**: Displays applications detected as idle
- **Power Usage**: Real-time power consumption monitoring
- **System Metrics**: CPU usage, memory usage, and system statistics

#### Settings Panel (Sidebar)
- **Idle Time Threshold**: Adjust when apps are considered idle (5-60 minutes)
- **Auto-shutdown**: Enable automatic closing of idle apps
- **Notifications**: Toggle reminder notifications
- **App Filters**: Exclude specific apps from monitoring

#### Action Buttons
- **🔔 Remind**: Send notification about idle app
- **😴 Sleep**: Put application in sleep mode
- **❌ Close**: Close the idle application

#### Analytics
- **Power Consumption Charts**: Visual representation of app power usage
- **Usage History**: 7-day trend analysis
- **Savings Calculator**: Estimated power and cost savings

### Energy Saving Tips
The app provides intelligent recommendations for:
- Optimizing system settings
- Reducing power consumption
- Implementing green computing practices

## Machine Learning Model

### Model Architecture
- **Algorithm**: Random Forest Classifier (lightweight configuration)
- **Features**: 8 key metrics for idle detection
  - CPU usage percentage
  - Memory usage (MB)
  - Mouse clicks per minute
  - Keyboard strokes per minute
  - Network activity (KB)
  - Window focus time (seconds)
  - Time since last interaction (minutes)
  - Power consumption (watts)

### Model Performance
- **Accuracy**: >99% on test dataset
- **Precision/Recall**: Balanced performance for both Active and Idle classes
- **Inference Time**: <50ms per prediction (optimized for real-time use)

### Green AI Optimizations
- Limited to 50 decision trees for efficiency
- Single-threaded processing to reduce power consumption
- Compressed model size for faster loading
- Optimized feature scaling for minimal computation

## Dataset Information

### Training Data
- **Size**: 1,000 samples
- **Features**: 8 numerical features + app names + timestamps
- **Target Distribution**: 60% Active, 40% Idle (balanced)
- **Data Source**: Simulated based on real-world app usage patterns

### Feature Engineering
- Normalized resource usage metrics
- Time-based interaction features
- Power consumption estimates
- System performance indicators

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` when running the app
**Solution**:
```bash
pip install -r requirements.txt
# Or upgrade packages
pip install --upgrade streamlit pandas scikit-learn
```

#### 2. Port Already in Use
**Problem**: "Port 8501 is already in use"
**Solution**:
```bash
streamlit run src/frontend/app.py --server.port 8502
```

#### 3. Model Loading Error
**Problem**: "Pre-trained model not found"
**Solution**:
```bash
cd src/backend
python ml_backend.py  # This will retrain the model
```

#### 4. Permission Errors (Windows)
**Problem**: System monitoring access denied
**Solution**: Run as administrator or adjust Windows security settings

#### 5. Slow Performance
**Problem**: App running slowly
**Solutions**:
- Reduce refresh rate in settings
- Close other resource-intensive applications
- Check available RAM (minimum 4GB recommended)

### Performance Optimization

#### For Better Performance:
1. **Increase refresh interval** in settings (30-60 seconds)
2. **Exclude system apps** from monitoring
3. **Use SSD storage** for faster model loading
4. **Ensure adequate RAM** (8GB+ recommended for smooth operation)

#### For Lower Power Consumption:
1. **Reduce monitoring frequency**
2. **Enable power saving mode** on laptop
3. **Close unnecessary background apps**
4. **Use the app's own recommendations**

## Development

### Adding New Features
1. **Backend modifications**: Edit `src/backend/ml_backend.py`
2. **Frontend changes**: Edit `src/frontend/app.py`
3. **New ML models**: Train and save in `models/` directory
4. **Additional data**: Add to `data/` directory

### Model Retraining
```bash
cd src/backend
python ml_backend.py
```

### Testing
```bash
# Test backend
cd src/backend
python -c "from ml_backend import *; print('Backend OK')"

# Test frontend (in browser)
streamlit run src/frontend/app.py
```

## Technical Specifications

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8-3.11
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for application + dependencies
- **CPU**: Any modern processor (optimized for efficiency)

### Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **psutil**: System and process utilities
- **joblib**: Model serialization
- **plotly**: Interactive visualizations

## Green AI Principles

### Energy Efficiency
- **Model Optimization**: Lightweight algorithms with minimal computational overhead
- **Resource Monitoring**: Real-time tracking of system resources
- **Smart Scheduling**: Intelligent task scheduling to minimize power usage
- **Efficient Data Processing**: Optimized data pipelines for reduced energy consumption

### Sustainability Features
- **Power Savings Calculator**: Quantifies environmental impact
- **Energy Tips**: Provides actionable sustainability recommendations  
- **Efficient Architecture**: Minimizes application footprint
- **Real-time Optimization**: Continuously optimizes for power efficiency

## Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Areas for Improvement
- Enhanced idle detection algorithms
- Additional system monitoring metrics
- Mobile application version
- Cloud-based analytics
- Integration with smart home systems

## License

This project is developed for educational purposes. Please ensure compliance with your institution's guidelines when using or modifying the code.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the GitHub issues (if applicable)
3. Create a new issue with detailed description

## Acknowledgments

- **Green AI Community**: For sustainable computing principles
- **Streamlit**: For the excellent web application framework
- **scikit-learn**: For machine learning capabilities
- **Open Source Community**: For the various libraries and tools used

---

**Smart Power Reminder** - Making computing more sustainable through intelligent power management! 🌱⚡
