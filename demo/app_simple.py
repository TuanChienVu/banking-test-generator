#!/usr/bin/env python3
"""
Banking AI Test Generator - Streamlit Application
Luận văn Thạc sĩ: Generative AI for Testing of Banking System in Mobile Applications

Demo web interface đơn giản để test model.
"""

import streamlit as st
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "utils"))

# Import model interface
from simple_model_interface import SimpleModelInterface

# Page configuration
st.set_page_config(
    page_title="Banking AI Test Generator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling - Enhanced UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 0;
    }
    
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 40px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 10s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.8); opacity: 0.3; }
        50% { transform: scale(1.2); opacity: 0.1; }
        100% { transform: scale(0.8); opacity: 0.3; }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-bottom: 0.3rem;
    }
    
    /* Test output styling */
    .test-output {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        color: #e8e8e8;
        border: 2px solid #0f3460;
        border-radius: 15px;
        padding: 2rem;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 1rem;
        line-height: 1.8;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.3),
                    0 5px 20px rgba(15, 52, 96, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .test-output::before {
        content: '📋 GHERKIN FORMAT';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 0.75rem;
        color: #53e3a6;
        opacity: 0.5;
        font-weight: bold;
        letter-spacing: 1px;
    }
    
    /* Keyword highlighting */
    .test-output strong {
        color: #53e3a6;
        font-weight: 600;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e1e4e8;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s;
    }
    
    .stTextArea textarea:focus {
        border-color: #2a5298;
        box-shadow: 0 0 0 2px rgba(42, 82, 152, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e1e4e8;
    }
    
    /* Metrics cards */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #f0f0f3, #ffffff);
        border: 1px solid #e1e4e8;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #2a5298;
        font-weight: 700;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f7f9fc;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(145deg, #e3f2fd, #f3f4f6);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stDownloadButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background: white;
        border-radius: 8px;
        border: 2px solid #e1e4e8;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Animation for loading */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading {
        animation: shimmer 2s infinite;
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 1000px 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sample User Stories
SAMPLE_USER_STORIES = {
    "🔐 Login & Authentication": "As a banking customer, I want to login to the mobile app using biometric authentication",
    "💳 Money Transfer": "As a user, I want to transfer money between my accounts with real-time balance updates",
    "💰 Balance Inquiry": "As a customer, I want to check my account balance and recent transactions",
    "📱 Bill Payment": "As a user, I want to pay utility bills through the mobile app",
    "🔔 Notifications": "As a customer, I want to receive push notifications for all transactions",
    "💳 Card Management": "As a user, I want to temporarily block my debit card if lost"
}

@st.cache_resource
def load_model():
    """Load model with caching"""
    return SimpleModelInterface(preload_model=True)

def main():
    """Main application"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Banking AI Test Generator</h1>
        <p>Generative AI for Testing of Banking System in Mobile Applications</p>
        <p><strong>Luận văn Thạc sĩ - Vũ Tuấn Chiến</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    # Load model with status indicator
    with st.spinner("🔧 Loading AI model..."):
        try:
            model_interface = load_model()
            
            # Display model status
            model_info = model_interface.get_model_info()
            
            if "Model Loaded Successfully" in model_info or "My Trained Model" in model_info:
                st.success("✅ YOUR Trained AI Model Loaded Successfully!")
                st.info("🤖 Using YOUR fine-tuned model (222.9M parameters, trained on Kaggle with 8000+ banking test cases)")
            else:
                st.warning("⚠️ Model running in demo mode")
                st.info("💡 **Demo Mode**: Sử dụng templates có sẵn. Model thực sẽ load trong lần generate đầu tiên (có thể mất 1-2 phút).")
                
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return
    
    # Sidebar
    st.sidebar.header("📋 Sample User Stories")
    
    selected_sample = st.sidebar.selectbox(
        "Choose a sample:",
        ["None"] + list(SAMPLE_USER_STORIES.keys())
    )
    
    if selected_sample != "None":
        st.sidebar.markdown(f"**{selected_sample}:**")
        st.sidebar.markdown(f"*{SAMPLE_USER_STORIES[selected_sample]}*")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input User Story")
        
        # Text input
        if selected_sample != "None":
            default_text = SAMPLE_USER_STORIES[selected_sample]
        else:
            default_text = ""
            
        user_story = st.text_area(
            "Enter your user story:",
            value=default_text,
            height=100,
            placeholder="As a [user], I want to [action] so that [benefit]..."
        )
        
        # Generation options
        st.subheader("⚙️ Generation Options")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            test_type = st.selectbox(
                "Test Type:",
                ["functional", "security", "performance", "compliance"]
            )
        
        with col_opt2:
            max_length = st.slider(
                "Max Length:",
                min_value=50,
                max_value=200,
                value=150
            )
        
        # Generate button
        if st.button("🚀 Generate Test Case", type="primary", key="generate_main"):
            if not user_story.strip():
                st.error("❌ Please enter a user story")
            else:
                with st.spinner("🔄 Generating test case..."):
                    try:
                        result = model_interface.generate_test_case(
                            user_story=user_story,
                            test_type=test_type,
                            max_length=max_length
                        )
                        
                        if result and result.get('success'):
                            st.success("✅ Test case generated successfully!")
                            
                            # Display result
                            st.subheader("📋 Generated Test Case")
                            test_case = result.get('test_case', 'No test case generated')
                            
                            # Check if test case contains HTML tags or use metadata to determine format
                            if '<div' in test_case or '<span' in test_case:
                                # HTML formatted output
                                st.markdown(f"""
                                <div class="test-output">
                                {test_case}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Plain text with markdown formatting
                                # Convert markdown bold to HTML for better display
                                html_test = test_case.replace('**', '')
                                html_test = html_test.replace('Given', '<span style="color: #53e3a6; font-weight: bold;">Given</span>')
                                html_test = html_test.replace('When', '<span style="color: #53e3a6; font-weight: bold;">When</span>')
                                html_test = html_test.replace('Then', '<span style="color: #53e3a6; font-weight: bold;">Then</span>')
                                html_test = html_test.replace('And', '<span style="color: #53e3a6; font-weight: bold;">And</span>')
                                html_test = html_test.replace('Feature:', '<span style="color: #ffd700; font-weight: bold;">Feature:</span>')
                                html_test = html_test.replace('Scenario:', '<span style="color: #ffd700; font-weight: bold;">Scenario:</span>')
                                html_test = html_test.replace('\n', '<br/>')
                                
                                st.markdown(f"""
                                <div class="test-output">
                                {html_test}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Metadata
                            if result.get('metadata'):
                                metadata = result['metadata']
                                with col2:
                                    st.subheader("📊 Generation Info")
                                    st.metric("Test Type", metadata.get('test_type', 'N/A'))
                                    st.metric("Confidence", f"{metadata.get('confidence', 0):.2f}")
                                    st.metric("Generation Time", f"{metadata.get('generation_time', 0):.2f}s")
                            
                            # Download button
                            st.download_button(
                                label="💾 Download Test Case",
                                data=test_case,
                                file_name=f"test_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                            
                        else:
                            st.error("❌ Failed to generate test case")
                            error_msg = result.get('error') if result else 'Unknown error'
                            st.error(f"Error: {error_msg}")
                            
                    except Exception as e:
                        st.error(f"❌ Generation error: {e}")
    
    with col2:
        st.subheader("ℹ️ About")
        st.markdown("""
        **Banking Test Generator** sử dụng AI để tự động sinh test case cho hệ thống ngân hàng di động.
        
        **Tính năng:**
        - 🎯 Functional Testing
        - 🔒 Security Testing  
        - ⚡ Performance Testing
        - 📋 Compliance Testing
        
        **Cách sử dụng:**
        1. Nhập user story
        2. Chọn loại test
        3. Nhấn Generate
        4. Tải xuống kết quả
        """)

if __name__ == "__main__":
    main()