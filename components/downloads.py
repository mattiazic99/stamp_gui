import streamlit as st
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt

def create_download_button(fig, filename, button_text=None):
    """Create a download button for matplotlib figures"""
    if button_text is None:
        button_text = f"📥 Download {filename}"
    
    # Save figure to bytes
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    img_buffer.seek(0)
    
    # Create download button
    st.download_button(
        label=button_text,
        data=img_buffer.getvalue(),
        file_name=filename,
        mime='image/png'
    )

def create_csv_download(df, filename, button_text=None, index=True):
    """Create a download button for pandas DataFrames as CSV"""
    if button_text is None:
        button_text = f"📥 Download {filename}"
    
    csv_data = df.to_csv(index=index).encode('utf-8')
    
    st.download_button(
        label=button_text,
        data=csv_data,
        file_name=filename,
        mime='text/csv'
    )

def create_excel_download(df, filename, button_text=None, sheet_name='Data'):
    """Create a download button for pandas DataFrames as Excel"""
    if button_text is None:
        button_text = f"📥 Download {filename}"
    
    # Create Excel file in memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=True)
    excel_buffer.seek(0)
    
    st.download_button(
        label=button_text,
        data=excel_buffer.getvalue(),
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def create_multiple_csv_download(dataframes_dict, zip_filename, button_text=None):
    """Create a download button for multiple DataFrames as a ZIP file with CSVs"""
    if button_text is None:
        button_text = f"📥 Download {zip_filename}"
    
    import zipfile
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for name, df in dataframes_dict.items():
            csv_data = df.to_csv(index=True)
            zip_file.writestr(f"{name}.csv", csv_data)
    
    zip_buffer.seek(0)
    
    st.download_button(
        label=button_text,
        data=zip_buffer.getvalue(),
        file_name=zip_filename,
        mime='application/zip'
    )

def create_text_download(text_content, filename, button_text=None):
    """Create a download button for text content"""
    if button_text is None:
        button_text = f"📥 Download {filename}"
    
    st.download_button(
        label=button_text,
        data=text_content,
        file_name=filename,
        mime='text/plain'
    )

def create_json_download(data, filename, button_text=None):
    """Create a download button for JSON data"""
    if button_text is None:
        button_text = f"📥 Download {filename}"
    
    import json
    json_string = json.dumps(data, indent=2)
    
    st.download_button(
        label=button_text,
        data=json_string,
        file_name=filename,
        mime='application/json'
    )

def display_download_section(title="📥 Downloads"):
    """Display a styled download section header"""
    st.markdown(f"""
    <div class="download-section">
        <h4>{title}</h4>
    </div>
    """, unsafe_allow_html=True)