"""
Web Interface for Bertrand Q-Learning Batch Simulations

Research-focused Streamlit app for running simulations with configurable parameters,
viewing results, and downloading logs/CSV files.

Usage:
    streamlit run web_app.py
"""

import io
import time
from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st
import numpy as np

# Import configuration constants
import config

# Import simulation modules
from batch_simulator import run_single_simulation

# Page configuration
st.set_page_config(
    page_title="Bertrand Q-Learning Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better research UI (dark mode compatible)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4da6ff;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-bottom: 2rem;
    }
    /* Ensure all containers inherit dark theme */
    .stMetric, div[data-testid="stExpander"], .stTextArea textarea {
        background-color: transparent !important;
    }
    /* Fix text areas */
    .stTextArea textarea {
        color: inherit !important;
    }
    /* Fix dataframe styling */
    .stDataFrame {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    if "simulation_log" not in st.session_state:
        st.session_state.simulation_log = ""
    if "is_running" not in st.session_state:
        st.session_state.is_running = False


def run_simulation_with_capture(
    discount_factors: List[float],
    n_simulations: int,
    max_steps: int,
    price_convergence_count: int,
    learning_rate: float,
    progress_bar,
    status_text,
) -> tuple[pd.DataFrame, str]:
    """
    Run batch simulations while capturing output for display.
    
    Returns:
        Tuple of (results DataFrame, log string)
    """
    # Create a string buffer to capture logs
    log_buffer = io.StringIO()
    
    # Generate filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"simulation_results_{timestamp}.csv"
    log_file = f"simulation_log_{timestamp}.txt"
    
    all_results = []
    total_sims = len(discount_factors) * n_simulations
    sim_count = 0
    
    # Header
    header = f"""{'='*70}
BATCH SIMULATION CONFIGURATION
{'='*70}
Discount factors: {discount_factors}
Simulations per factor: {n_simulations}
Total simulations: {total_sims:,}
Max steps per sim: {max_steps:,}
Price convergence threshold: {price_convergence_count}
Learning rate: {learning_rate}
Output file: {output_file}
{'='*70}
"""
    log_buffer.write(header)
    
    batch_start_time = time.time()
    
    for gamma in discount_factors:
        log_buffer.write(f"\n{'='*70}\n")
        log_buffer.write(f"Running {n_simulations} simulations for γ = {gamma}\n")
        log_buffer.write(f"{'='*70}\n")
        
        for sim_idx in range(n_simulations):
            sim_count += 1
            seed = int(time.time() * 1000000) % (2**31)
            
            # Update progress
            progress = sim_count / total_sims
            progress_bar.progress(progress)
            status_text.text(f"Running simulation {sim_count}/{total_sims} (γ={gamma}, run {sim_idx+1}/{n_simulations})")
            
            log_buffer.write(f"\nSimulation {sim_count}/{total_sims} (γ={gamma}, run {sim_idx+1}/{n_simulations})\n")
            
            # Run simulation
            result = run_single_simulation(
                discount_factor=gamma,
                max_steps=max_steps,
                price_convergence_count=price_convergence_count,
                learning_rate=learning_rate,
                seed=seed,
                verbose=False,
            )
            
            result["simulation_id"] = sim_count
            result["gamma_index"] = sim_idx
            all_results.append(result)
            
            # Log result summary
            log_buffer.write(f"  Market 1: Converged={result['converged_m1']}, "
                           f"Price A={result['converged_price_1a']:.2f}, "
                           f"Price B={result['converged_price_1b']:.2f}\n")
            log_buffer.write(f"  Market 2: Converged={result['converged_m2']}, "
                           f"Price A={result['converged_price_2a']:.2f}, "
                           f"Price B={result['converged_price_2b']:.2f}\n")
            log_buffer.write(f"  Time: {result['time_to_converge']:.2f}s\n")
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save files
    df_results.to_csv(output_file, index=False)
    
    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time
    
    # Summary statistics
    summary = f"""
{'='*70}
BATCH SIMULATION COMPLETE
{'='*70}
Total simulations: {total_sims:,}
Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)
Average time per sim: {total_time/total_sims:.2f}s
Results saved to: {output_file}
{'='*70}

SUMMARY STATISTICS - MARKET 1
{'='*70}
"""
    log_buffer.write(summary)
    
    # Market 1 summary
    summary_m1 = df_results.groupby('discount_factor_a').agg({
        'converged_m1': ['sum', 'mean'],
        'convergence_step_m1': ['mean', 'std'],
        'converged_price_1a': ['mean', 'std'],
        'converged_price_1b': ['mean', 'std'],
    }).round(4)
    log_buffer.write(summary_m1.to_string())
    
    log_buffer.write(f"\n\nSUMMARY STATISTICS - MARKET 2\n{'='*70}\n")
    
    # Market 2 summary
    summary_m2 = df_results.groupby('discount_factor_a').agg({
        'converged_m2': ['sum', 'mean'],
        'convergence_step_m2': ['mean', 'std'],
        'converged_price_2a': ['mean', 'std'],
        'converged_price_2b': ['mean', 'std'],
    }).round(4)
    log_buffer.write(summary_m2.to_string())
    
    # Get log content
    log_content = log_buffer.getvalue()
    
    # Save log file
    with open(log_file, 'w') as f:
        f.write(log_content)
    
    return df_results, log_content, output_file, log_file


def main():
    """Main Streamlit app."""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📊 Bertrand Q-Learning Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Agent Price Competition Research Tool</p>', unsafe_allow_html=True)
    
    # Sidebar - Parameter Configuration
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        
        st.subheader("Discount Factors (γ)")
        
        # Use discount factors from config
        discount_factors_list = config.DISCOUNT_FACTORS
        
        st.caption(f"Fixed: {len(discount_factors_list)} values")
        st.code(", ".join([f"{g:.2f}" for g in discount_factors_list]), language=None)
        
        st.divider()
        
        st.subheader("Simulation Settings")
        
        n_simulations = st.number_input(
            "Simulations per γ",
            min_value=1,
            max_value=100,
            value=config.DEFAULT_N_SIMULATIONS,
            help="Number of independent runs for each discount factor"
        )
        
        max_steps = st.select_slider(
            "Max steps per simulation",
            options=[100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
            value=config.DEFAULT_MAX_STEPS,
            format_func=lambda x: f"{x:,}"
        )
        
        price_convergence_count = st.number_input(
            "Price convergence threshold",
            min_value=10,
            max_value=10000,
            value=config.DEFAULT_PRICE_CONVERGENCE_COUNT,
            help="Consecutive steps with same price to trigger convergence"
        )
        
        learning_rate = st.slider(
            "Learning rate (α)",
            min_value=0.01,
            max_value=1.0,
            value=config.LEARNING_RATE,
            step=0.01,
            format="%.2f",
            help="Q-learning update rate"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Run button
        run_button = st.button(
            "🚀 Run Batch Simulation",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running
        )
        
        if run_button:
            try:
                # Use the fixed discount factors
                gammas = discount_factors_list
                
                st.session_state.is_running = True
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run simulations
                with st.spinner("Running simulations..."):
                    df_results, log_content, csv_file, log_file = run_simulation_with_capture(
                        discount_factors=gammas,
                        n_simulations=n_simulations,
                        max_steps=max_steps,
                        price_convergence_count=price_convergence_count,
                        learning_rate=learning_rate,
                        progress_bar=progress_bar,
                        status_text=status_text,
                    )
                
                # Store results
                st.session_state.simulation_results = df_results
                st.session_state.simulation_log = log_content
                st.session_state.csv_file = csv_file
                st.session_state.log_file = log_file
                st.session_state.is_running = False
                
                progress_bar.progress(1.0)
                status_text.text("✅ Simulation complete!")
                st.success(f"Completed {len(df_results)} simulations!")
                
            except Exception as e:
                st.session_state.is_running = False
                st.error(f"Error: {str(e)}")
                raise e
    
    with col2:
        # Download buttons (only show if results exist)
        if st.session_state.simulation_results is not None:
            st.subheader("📥 Download Results")
            
            # CSV download
            csv_data = st.session_state.simulation_results.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=st.session_state.get("csv_file", "simulation_results.csv"),
                mime="text/csv",
                use_container_width=True,
            )
            
            # Log download
            st.download_button(
                label="📝 Download Log",
                data=st.session_state.simulation_log,
                file_name=st.session_state.get("log_file", "simulation_log.txt"),
                mime="text/plain",
                use_container_width=True,
            )
    
    # Results display
    if st.session_state.simulation_results is not None:
        st.divider()
        
        df = st.session_state.simulation_results
        
        # Summary metrics
        st.subheader("📈 Summary Metrics")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Simulations", len(df))
        with metric_cols[1]:
            st.metric("Market 1 Convergence", f"{df['converged_m1'].mean()*100:.1f}%")
        with metric_cols[2]:
            st.metric("Market 2 Convergence", f"{df['converged_m2'].mean()*100:.1f}%")
        with metric_cols[3]:
            st.metric("Avg Time/Sim", f"{df['time_to_converge'].mean():.1f}s")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results Table", "📈 Charts", "📋 Summary Stats", "📝 Full Log"])
        
        with tab1:
            st.dataframe(
                df,
                use_container_width=True,
                height=400,
            )
        
        with tab2:
            # Charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.subheader("Market 1: Converged Prices by γ")
                chart_data_m1 = df.groupby('discount_factor_a').agg({
                    'converged_price_1a': 'mean',
                    'converged_price_1b': 'mean',
                }).reset_index()
                st.line_chart(
                    chart_data_m1.set_index('discount_factor_a'),
                    use_container_width=True,
                )
            
            with chart_col2:
                st.subheader("Market 2: Converged Prices by γ")
                chart_data_m2 = df.groupby('discount_factor_a').agg({
                    'converged_price_2a': 'mean',
                    'converged_price_2b': 'mean',
                }).reset_index()
                st.line_chart(
                    chart_data_m2.set_index('discount_factor_a'),
                    use_container_width=True,
                )
            
            # Convergence steps chart
            st.subheader("Convergence Steps by Discount Factor")
            conv_data = df.groupby('discount_factor_a').agg({
                'convergence_step_m1': 'mean',
                'convergence_step_m2': 'mean',
            }).reset_index()
            st.bar_chart(
                conv_data.set_index('discount_factor_a'),
                use_container_width=True,
            )
        
        with tab3:
            st.subheader("Market 1 Statistics")
            summary_m1 = df.groupby('discount_factor_a').agg({
                'converged_m1': ['sum', 'mean'],
                'convergence_step_m1': ['mean', 'std'],
                'converged_price_1a': ['mean', 'std'],
                'converged_price_1b': ['mean', 'std'],
                'converged_q_value_1a': ['mean', 'std'],
                'converged_q_value_1b': ['mean', 'std'],
            }).round(4)
            st.dataframe(summary_m1, use_container_width=True)
            
            st.subheader("Market 2 Statistics")
            summary_m2 = df.groupby('discount_factor_a').agg({
                'converged_m2': ['sum', 'mean'],
                'convergence_step_m2': ['mean', 'std'],
                'converged_price_2a': ['mean', 'std'],
                'converged_price_2b': ['mean', 'std'],
                'converged_q_value_2a': ['mean', 'std'],
                'converged_q_value_2b': ['mean', 'std'],
            }).round(4)
            st.dataframe(summary_m2, use_container_width=True)
        
        with tab4:
            st.text_area(
                "Simulation Log",
                value=st.session_state.simulation_log,
                height=500,
            )
    
    # Footer
    st.divider()
    st.caption("Bertrand Q-Learning Multi-Agent Simulation | Research Tool")


if __name__ == "__main__":
    main()

