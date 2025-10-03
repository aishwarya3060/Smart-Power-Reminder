import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import sys
from datetime import datetime, timedelta
import json

# Add backend path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.append(backend_path)

try:
    from ml_backend import IdleDetectionModel, SystemMonitor, PowerSavingsCalculator
except ImportError:
    st.error("Backend modules not found. Please ensure the backend is properly installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Smart Power Reminder - Green AI Desktop App",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #4682B4;
    margin-bottom: 0.5rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e1e5e9;
}
.idle-app {
    background-color: #ffebee;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid #f44336;
}
.active-app {
    background-color: #e8f5e8;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid #4caf50;
}
.energy-tip {
    background-color: #fff3e0;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff9800;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

class SmartPowerReminderApp:
    def __init__(self):
        self.model = IdleDetectionModel()
        self.monitor = SystemMonitor()
        self.calculator = PowerSavingsCalculator()
        self.load_model()

    def load_model(self):
        """Load the trained ML model."""
        # Use project root for robust path resolution
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, "models")
        success = self.model.load_model(model_dir)
        if not success:
            st.warning("⚠️ Pre-trained model not found. Please train the model first.")

    def simulate_running_apps(self, num_apps=8):
        """Simulate running applications with their resource usage."""
        app_names = [
            'Google Chrome', 'VS Code', 'Spotify', 'Microsoft Word', 
            'Excel', 'Steam', 'Discord', 'WhatsApp Desktop',
            'Adobe Photoshop', 'VLC Media Player', 'Zoom', 'Slack'
        ]

        apps = []
        for i in range(num_apps):
            app_name = np.random.choice(app_names)
            features = self.monitor.simulate_app_features(app_name)

            # Get prediction from ML model
            if self.model.model is not None:
                prediction_result = self.model.predict_idle_status(features)
                status = prediction_result['prediction']
                confidence = max(prediction_result['idle_probability'], 
                               prediction_result['active_probability'])
            else:
                # Fallback logic without ML model
                if (features['cpu_usage_percent'] < 10 and 
                    features['time_since_last_interaction_min'] > 15):
                    status = 'Idle'
                else:
                    status = 'Active'
                confidence = 0.85

            apps.append({
                'app_name': app_name,
                'status': status,
                'confidence': confidence,
                **features
            })

        return apps

    def create_dashboard(self):
        """Create the main dashboard."""
        st.markdown('<h1 class="main-header">🌱 Smart Power Reminder</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Green AI-Powered Desktop Energy Management</p>', unsafe_allow_html=True)

        # Sidebar for settings
        with st.sidebar:
            st.header("⚙️ Settings")

            idle_threshold = st.slider("Idle Time Threshold (minutes)", 5, 60, 15)
            auto_shutdown = st.checkbox("Enable Auto-shutdown", False)
            notification_enabled = st.checkbox("Enable Notifications", True)

            st.header("🎯 App Filters")
            excluded_apps = st.multiselect(
                "Exclude from monitoring:",
                ['System', 'Antivirus', 'Windows Explorer', 'Task Manager'],
                default=['System', 'Windows Explorer']
            )

            st.header("📊 Monitoring")
            refresh_interval = st.select_slider(
                "Refresh Rate (seconds):",
                options=[5, 10, 30, 60],
                value=10
            )

        # Auto-refresh mechanism
        placeholder = st.empty()

        with placeholder.container():
            # Get current app data
            apps_data = self.simulate_running_apps()
            df_apps = pd.DataFrame(apps_data)

            # Filter out excluded apps
            df_apps = df_apps[~df_apps['app_name'].isin(excluded_apps)]

            # Separate active and idle apps
            idle_apps = df_apps[df_apps['status'] == 'Idle']
            active_apps = df_apps[df_apps['status'] == 'Active']

            # Main metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="🟢 Active Apps",
                    value=len(active_apps),
                    delta=f"{len(active_apps)}/{len(df_apps)} running"
                )

            with col2:
                st.metric(
                    label="🔴 Idle Apps",
                    value=len(idle_apps),
                    delta="Ready for shutdown" if len(idle_apps) > 0 else "None detected"
                )

            with col3:
                total_power = df_apps['power_consumption_watts'].sum()
                idle_power = idle_apps['power_consumption_watts'].sum()
                st.metric(
                    label="⚡ Power Usage",
                    value=f"{total_power:.1f}W",
                    delta=f"-{idle_power:.1f}W potential savings"
                )

            with col4:
                avg_cpu = df_apps['cpu_usage_percent'].mean()
                st.metric(
                    label="🖥️ Avg CPU Usage",
                    value=f"{avg_cpu:.1f}%",
                    delta="System load"
                )

            # Power savings calculation
            if len(idle_apps) > 0:
                savings = self.calculator.estimate_power_saved(idle_apps['app_name'].tolist())

                st.markdown("---")
                st.markdown('<h2 class="sub-header">💰 Potential Energy Savings</h2>', unsafe_allow_html=True)

                savings_col1, savings_col2, savings_col3 = st.columns(3)

                with savings_col1:
                    st.metric("Daily Savings", f"{savings['daily_savings_kwh']:.2f} kWh")

                with savings_col2:
                    st.metric("Daily Cost Savings", f"${savings['daily_cost_savings']:.2f}")

                with savings_col3:
                    st.metric("Monthly Savings", f"${savings['monthly_cost_savings']:.2f}")

            # App Status Display
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<h3 class="sub-header">🔴 Idle Applications</h3>', unsafe_allow_html=True)

                if len(idle_apps) > 0:
                    for _, app in idle_apps.iterrows():
                        st.markdown(f"""
                        <div class="idle-app">
                            <strong>{app['app_name']}</strong><br>
                            CPU: {app['cpu_usage_percent']:.1f}% | 
                            RAM: {app['memory_usage_mb']:.0f}MB | 
                            Idle for: {app['time_since_last_interaction_min']:.1f} min<br>
                            <small>Confidence: {app['confidence']:.0%}</small>
                        </div>
                        """, unsafe_allow_html=True)

                        # Action buttons
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.button(f"🔔 Remind", key=f"remind_{app['app_name']}"):
                                st.success(f"Reminder sent for {app['app_name']}")
                        with col_b:
                            if st.button(f"😴 Sleep", key=f"sleep_{app['app_name']}"):
                                st.info(f"{app['app_name']} put to sleep mode")
                        with col_c:
                            if st.button(f"❌ Close", key=f"close_{app['app_name']}"):
                                st.warning(f"{app['app_name']} closed")
                else:
                    st.info("🎉 No idle applications detected!")

            with col2:
                st.markdown('<h3 class="sub-header">🟢 Active Applications</h3>', unsafe_allow_html=True)

                for _, app in active_apps.iterrows():
                    st.markdown(f"""
                    <div class="active-app">
                        <strong>{app['app_name']}</strong><br>
                        CPU: {app['cpu_usage_percent']:.1f}% | 
                        RAM: {app['memory_usage_mb']:.0f}MB | 
                        Power: {app['power_consumption_watts']:.1f}W<br>
                        <small>Status: Active ({app['confidence']:.0%} confidence)</small>
                    </div>
                    """, unsafe_allow_html=True)

            # Visualizations
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📊 System Analysis</h2>', unsafe_allow_html=True)

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                # Power consumption chart
                fig_power = px.bar(
                    df_apps, 
                    x='app_name', 
                    y='power_consumption_watts',
                    color='status',
                    title="Power Consumption by Application",
                    color_discrete_map={'Active': '#4CAF50', 'Idle': '#F44336'}
                )
                fig_power.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_power, use_container_width=True)

            with viz_col2:
                # CPU usage chart
                fig_cpu = px.scatter(
                    df_apps, 
                    x='cpu_usage_percent', 
                    y='memory_usage_mb',
                    size='power_consumption_watts',
                    color='status',
                    hover_data=['app_name'],
                    title="CPU vs Memory Usage",
                    color_discrete_map={'Active': '#4CAF50', 'Idle': '#F44336'}
                )
                st.plotly_chart(fig_cpu, use_container_width=True)

            # Usage history (simulated)
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📈 Usage History</h2>', unsafe_allow_html=True)

            # Generate sample historical data
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
            history_data = []

            for date in dates:
                history_data.append({
                    'date': date,
                    'total_apps': np.random.randint(8, 15),
                    'idle_apps': np.random.randint(1, 6),
                    'power_saved': np.random.uniform(20, 80),
                    'cost_saved': np.random.uniform(0.05, 0.25)
                })

            df_history = pd.DataFrame(history_data)

            history_col1, history_col2 = st.columns(2)

            with history_col1:
                fig_trend = px.line(
                    df_history, 
                    x='date', 
                    y=['total_apps', 'idle_apps'],
                    title="Application Usage Trend (7 Days)"
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            with history_col2:
                fig_savings = px.area(
                    df_history, 
                    x='date', 
                    y='cost_saved',
                    title="Daily Cost Savings ($)"
                )
                st.plotly_chart(fig_savings, use_container_width=True)

            # Energy saving tips
            st.markdown("---")
            st.markdown('<h2 class="sub-header">💡 Green AI Energy Tips</h2>', unsafe_allow_html=True)

            tips = self.calculator.get_energy_tips()
            selected_tips = np.random.choice(tips, 3, replace=False)

            for tip in selected_tips:
                st.markdown(f"""
                <div class="energy-tip">
                    💡 {tip}
                </div>
                """, unsafe_allow_html=True)

        # Auto-refresh
        time.sleep(refresh_interval)
        st.rerun()

def main():
    app = SmartPowerReminderApp()
    app.create_dashboard()

if __name__ == "__main__":
    main()