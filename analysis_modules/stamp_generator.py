'''import streamlit as st
import subprocess
import os
import time
import zipfile
from io import BytesIO
from components.downloads import create_download_button, display_download_section
from components.styling import apply_plot_style

def show():
    """STAMP Dataset Generator Page"""
    
    st.header("🛠️ STAMP Dataset Generator")
    st.markdown("Generate STAMP-compatible datasets from raw expression data for analysis.")
    
    # === Configuration ===
    DIR_NORMALIZED = "tpm_normalizzati"
    DIR_SETS = "sets_stamp"
    DIR_SETS_MAPPED = "sets_stamp_symboli"
    GENE_MAP_FILE = "all_genes.txt"
    SCRIPT_STEP2 = "step_2.py"
    SCRIPT_STEP3A = "step_3a_corretto.py"
    
    # === Session state ===
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "tissues" not in st.session_state:
        st.session_state.tissues = []
    
    # === Info Section ===
    st.markdown("""
    <div class="analysis-section">
        <h3>📋 About STAMP Generator</h3>
        <p>This tool processes raw gene expression data to generate STAMP-compatible files for age-based gene switching analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show process overview
    with st.expander("🔍 View Process Overview"):
        st.markdown("""
        ### 🔄 Generation Process:
        
        1. **📊 Data Normalization** - Normalize TPM values across age groups
        2. **🎯 Threshold Application** - Apply expression threshold to identify active genes
        3. **🧬 Gene Set Creation** - Generate age-specific gene sets per tissue
        4. **🏷️ Symbol Mapping** - Convert Ensembl IDs to gene symbols
        5. **📁 File Export** - Create downloadable STAMP-compatible files
        
        ### 📂 Output Files:
        - **`*_sets_stamp.txt`** - Raw gene sets with Ensembl IDs
        - **`*_sets_stamp_mapped.txt`** - Gene sets with human-readable symbols
        """)
    
    # === Prerequisites Check ===
    st.markdown("""
    <div class="analysis-section">
        <h3>🔍 Prerequisites Check</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Check directories and files
    checks = [
        (DIR_NORMALIZED, "Normalized data directory"),
        (GENE_MAP_FILE, "Gene mapping file"),
        (SCRIPT_STEP2, "Step 2 processing script"),
        (SCRIPT_STEP3A, "Step 3A processing script")
    ]
    
    all_checks_passed = True
    for path, description in checks:
        if os.path.exists(path):
            st.success(f"✅ {description}: `{path}` found")
        else:
            st.error(f"❌ {description}: `{path}` not found")
            all_checks_passed = False
    
    if not all_checks_passed:
        st.error("❌ Prerequisites not met. Please ensure all required files and directories are present.")
        st.info("💡 Make sure you have run the data preparation pipeline before using this generator.")
        return
    
    # === Tissue Selection ===
    st.markdown("""
    <div class="analysis-section">
        <h3>📂 Tissue Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get available tissues
    file_list = [f for f in os.listdir(DIR_NORMALIZED) if f.endswith("_normalized.csv")]
    all_tissues = sorted([f.replace("_normalized.csv", "") for f in file_list])
    
    if not all_tissues:
        st.error("❌ No normalized tissue files found. Please run the normalization step first.")
        return
    
    st.info(f"📊 Found {len(all_tissues)} normalized tissue datasets available for processing.")
    
    # Selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        choice = st.radio(
            "📌 Select tissues to process:",
            ["Single Tissue", "Multiple Tissues", "All Tissues"],
            help="Choose how many tissues to process in this run"
        )
        
        if choice == "Single Tissue":
            selected_tissues = [st.selectbox("🔬 Choose one tissue:", all_tissues)]
        elif choice == "Multiple Tissues":
            selected_tissues = st.multiselect(
                "🔬 Choose multiple tissues:",
                all_tissues,
                help="Select specific tissues to process"
            )
        else:
            selected_tissues = all_tissues
            st.info(f"🔄 All {len(all_tissues)} tissues will be processed.")
    
    with col2:
        st.markdown("### 📊 Selection Summary")
        if selected_tissues:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(selected_tissues)}</h3>
                <p>Tissues Selected</p>
            </div>
            """, unsafe_allow_html=True)
            
            if len(selected_tissues) <= 5:
                st.markdown("**Selected tissues:**")
                for tissue in selected_tissues:
                    st.markdown(f"• {tissue}")
            else:
                st.markdown(f"**Selected:** {selected_tissues[0]}, {selected_tissues[1]}, ... and {len(selected_tissues)-2} more")
    
    if not selected_tissues:
        st.warning("⚠️ No tissues selected. Please choose at least one tissue to process.")
        return
    
    # === Processing Parameters ===
    st.markdown("""
    <div class="analysis-section">
        <h3>⚙️ Processing Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider(
            "🎚️ Gene Expression Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Threshold to consider a gene as 'expressed' (0.0 = all genes, 1.0 = only highly expressed)"
        )
        
        st.markdown(f"""
        **Current threshold:** `{threshold}`
        
        - **Lower values** (0.1-0.3): Include more genes, detect subtle changes
        - **Medium values** (0.4-0.6): Balanced approach (recommended)
        - **Higher values** (0.7-1.0): Only highly expressed genes
        """)
    
    with col2:
        st.markdown("### 🎯 Expected Output")
        st.markdown(f"""
        **For each tissue, you'll get:**
        - 📁 Raw gene sets (Ensembl IDs)
        - 🏷️ Mapped gene sets (Gene symbols)
        - 📊 5 age groups per tissue (30-79 years)
        
        **Total files:** {len(selected_tissues) * 2} files
        """)
    
    # === Generation Process ===
    st.markdown("""
    <div class="analysis-section">
        <h3>🚀 File Generation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Generate STAMP Files", type="primary", use_container_width=True):
        st.session_state.tissues = selected_tissues
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("🔄 Generating STAMP files..."):
                # Step 1: Update threshold in script
                status_text.text("📝 Updating processing parameters...")
                progress_bar.progress(10)
                
                with open(SCRIPT_STEP3A, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                with open(SCRIPT_STEP3A, "w", encoding="utf-8") as f:
                    for line in lines:
                        if line.strip().startswith("THRESHOLD ="):
                            f.write(f"THRESHOLD = {threshold}  # threshold to consider a gene 'expressed'\n")
                        else:
                            f.write(line)
                
                # Step 2: Run processing scripts
                status_text.text("⚙️ Running data processing pipeline...")
                progress_bar.progress(30)
                
                subprocess.run(["python", SCRIPT_STEP2], check=True)
                progress_bar.progress(50)
                
                subprocess.run(["python", SCRIPT_STEP3A], check=True)
                progress_bar.progress(70)
                
                # Step 3: Gene mapping
                status_text.text("🧬 Mapping gene symbols...")
                
                def load_gene_map():
                    gene_map = {}
                    with open(GENE_MAP_FILE, encoding="utf-8") as f:
                        for line in f:
                            if "(" in line and ")" in line:
                                ensembl = line.split("(")[0].strip()
                                symbol = line.split("(")[1].replace(")", "").strip()
                                gene_map[ensembl] = symbol
                    return gene_map
                
                os.makedirs(DIR_SETS_MAPPED, exist_ok=True)
                gene_map = load_gene_map()
                
                # Process each tissue
                for i, tissue in enumerate(selected_tissues):
                    status_text.text(f"🔬 Processing tissue: {tissue} ({i+1}/{len(selected_tissues)})")
                    
                    input_path = os.path.join(DIR_SETS, f"{tissue}_sets_stamp.txt")
                    output_path = os.path.join(DIR_SETS_MAPPED, f"{tissue}_sets_stamp_mapped.txt")
                    
                    if os.path.exists(input_path):
                        with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
                            for line in fin:
                                mapped_genes = [gene_map.get(g, g) for g in line.strip().split()]
                                fout.write(" ".join(mapped_genes) + "\n")
                    
                    progress_bar.progress(70 + (i + 1) * 25 // len(selected_tissues))
                
                progress_bar.progress(100)
                status_text.text("✅ Generation completed successfully!")
                
                st.session_state.processed = True
                st.success("🎉 All STAMP files have been generated successfully!")
                
                # Show summary
                st.markdown("### 📊 Generation Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(selected_tissues)}</h3>
                        <p>Tissues Processed</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{threshold}</h3>
                        <p>Expression Threshold</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_files = len([f for f in os.listdir(DIR_SETS_MAPPED) if f.endswith("_sets_stamp_mapped.txt")])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{total_files}</h3>
                        <p>Files Generated</p>
                    </div>
                    """, unsafe_allow_html=True)
                
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error during processing: {e}")
            st.error("Please check that all required scripts are present and executable.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
    
    # === Download Section ===
    if st.session_state.processed:
        display_download_section("📥 Download Generated Files")
        
        # Collect available files
        download_options = []
        for tissue in st.session_state.tissues:
            path1 = os.path.join(DIR_SETS, f"{tissue}_sets_stamp.txt")
            path2 = os.path.join(DIR_SETS_MAPPED, f"{tissue}_sets_stamp_mapped.txt")
            
            if os.path.exists(path1):
                download_options.append((f"{tissue}_sets_stamp.txt", path1, "Raw (Ensembl IDs)"))
            if os.path.exists(path2):
                download_options.append((f"{tissue}_sets_stamp_mapped.txt", path2, "Mapped (Gene Symbols)"))
        
        if download_options:
            st.markdown("### 📁 Available Files")
            
            # Show file list with descriptions
            file_df_data = []
            for name, path, desc in download_options:
                file_size = os.path.getsize(path) / 1024  # KB
                file_df_data.append({
                    "File Name": name,
                    "Type": desc,
                    "Size (KB)": f"{file_size:.1f}",
                    "Tissue": name.split("_")[0]
                })
            
            import pandas as pd
            file_df = pd.DataFrame(file_df_data)
            st.dataframe(file_df, use_container_width=True)
            
            # File selection for download
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Select Files to Download")
                selected_files = st.multiselect(
                    "Choose files:",
                    [x[0] for x in download_options],
                    help="Select one or more files to download"
                )
            
            with col2:
                st.markdown("#### ⚡ Quick Selection")
                if st.button("📊 Select All Mapped Files", use_container_width=True):
                    selected_files = [x[0] for x in download_options if "mapped" in x[0]]
                    st.experimental_rerun()
                
                if st.button("🧬 Select All Raw Files", use_container_width=True):
                    selected_files = [x[0] for x in download_options if "mapped" not in x[0]]
                    st.experimental_rerun()
                
                if st.button("📁 Select All Files", use_container_width=True):
                    selected_files = [x[0] for x in download_options]
                    st.experimental_rerun()
            
            # Download buttons
            if selected_files:
                st.markdown("### ⬇️ Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Individual file downloads
                    st.markdown("**Individual Downloads:**")
                    for name, path, desc in download_options:
                        if name in selected_files:
                            with open(path, 'rb') as f:
                                st.download_button(
                                    label=f"📄 {name}",
                                    data=f.read(),
                                    file_name=name,
                                    mime='text/plain',
                                    key=f"download_{name}"
                                )
                
                with col2:
                    # ZIP download
                    st.markdown("**Bulk Download:**")
                    if st.button("📦 Create ZIP Archive", use_container_width=True):
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                            for name, path, desc in download_options:
                                if name in selected_files:
                                    zipf.write(path, arcname=name)
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            label="📥 Download ZIP Archive",
                            data=zip_buffer.getvalue(),
                            file_name="stamp_files_generated.zip",
                            mime='application/zip',
                            use_container_width=True
                        )
            else:
                st.info("👆 Select files above to enable download options.")
        
        # Reset option
        st.markdown("---")
        if st.button("🔄 Start New Generation", type="secondary", use_container_width=True):
            st.session_state.processed = False
            st.session_state.tissues = []
            st.experimental_rerun()
    
    # === Help Section ===
    with st.expander("❓ Need Help?"):
        st.markdown("""
        ### 🆘 Troubleshooting
        
        **Common Issues:**
        
        1. **Missing prerequisites**: Ensure all required scripts and data directories are present
        2. **Processing errors**: Check that Python scripts have proper permissions
        3. **No output files**: Verify that input data is properly formatted
        
        ### 📚 File Formats
        
        **Generated STAMP files contain:**
        - 5 lines (one per age group: 30-39, 40-49, 50-59, 60-69, 70-79)
        - Space-separated gene names/IDs per line
        - Compatible with all STAMP analysis modules
        
        ### 🔧 Parameters
        
        **Expression Threshold:**
        - Determines which genes are considered "active" in each age group
        - Higher values = more stringent filtering
        - Recommended: 0.4-0.6 for balanced analysis
        """)
'''



import streamlit as st
import subprocess
import os
import time
import zipfile
from io import BytesIO
from components.downloads import create_download_button, display_download_section
from components.styling import apply_plot_style

def show():
    """STAMP Dataset Generator Page"""
    
    st.header("🛠️ STAMP Dataset Generator")
    st.markdown("Generate STAMP-compatible datasets from raw expression data for analysis.")
    
    # === Configuration ===
    DIR_NORMALIZED = "tpm_normalizzati"
    DIR_SETS = "sets_stamp"
    DIR_SETS_MAPPED = "sets_stamp_symboli"
    GENE_MAP_FILE = "all_genes.txt"
    SCRIPT_STEP2 = "step_2.py"
    SCRIPT_STEP3A = "step_3a_corretto.py"
    
    # === Session state ===
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "tissues" not in st.session_state:
        st.session_state.tissues = []
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    
    # === Info Section ===
    st.markdown("""
    <div class="analysis-section">
        <h3>📋 About STAMP Generator</h3>
        <p>This tool processes raw gene expression data to generate STAMP-compatible files for age-based gene switching analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show process overview
    with st.expander("🔍 View Process Overview"):
        st.markdown("""
        ### 📄 Generation Process:
        
        1. **📊 Data Normalization** - Normalize TPM values across age groups
        2. **🎯 Threshold Application** - Apply expression threshold to identify active genes
        3. **🧬 Gene Set Creation** - Generate age-specific gene sets per tissue
        4. **🏷️ Symbol Mapping** - Convert Ensembl IDs to gene symbols
        5. **📁 File Export** - Create downloadable STAMP-compatible files
        
        ### 📂 Output Files:
        - **`*_sets_stamp.txt`** - Raw gene sets with Ensembl IDs
        - **`*_sets_stamp_mapped.txt`** - Gene sets with human-readable symbols
        """)
    
    # === Prerequisites Check ===
    st.markdown("""
    <div class="analysis-section">
        <h3>🔍 Prerequisites Check</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Check directories and files
    checks = [
        (DIR_NORMALIZED, "Normalized data directory"),
        (GENE_MAP_FILE, "Gene mapping file"),
        (SCRIPT_STEP2, "Step 2 processing script"),
        (SCRIPT_STEP3A, "Step 3A processing script")
    ]
    
    all_checks_passed = True
    for path, description in checks:
        if os.path.exists(path):
            st.success(f"✅ {description} found")
        else:
            st.error(f"❌ {description} not found")
            all_checks_passed = False
    
    if not all_checks_passed:
        st.error("❌ Prerequisites not met. Please ensure all required files and directories are present.")
        st.info("💡 Make sure you have run the data preparation pipeline before using this generator.")
        return
    
    # === Tissue Selection ===
    st.markdown("""
    <div class="analysis-section">
        <h3>📂 Tissue Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get available tissues
    file_list = [f for f in os.listdir(DIR_NORMALIZED) if f.endswith("_normalized.csv")]
    all_tissues = sorted([f.replace("_normalized.csv", "") for f in file_list])
    
    if not all_tissues:
        st.error("❌ No normalized tissue files found. Please run the normalization step first.")
        return
    
    st.info(f"📊 Found {len(all_tissues)} normalized tissue datasets available for processing.")
    
    # Selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        choice = st.radio(
            "📌 Select tissues to process:",
            ["Single Tissue", "Multiple Tissues", "All Tissues"],
            help="Choose how many tissues to process in this run"
        )
        
        if choice == "Single Tissue":
            selected_tissues = [st.selectbox("🔬 Choose one tissue:", all_tissues)]
        elif choice == "Multiple Tissues":
            selected_tissues = st.multiselect(
                "🔬 Choose multiple tissues:",
                all_tissues,
                help="Select specific tissues to process"
            )
        else:
            selected_tissues = all_tissues
            st.info(f"📄 All {len(all_tissues)} tissues will be processed.")
    
    with col2:
        st.markdown("### 📊 Selection Summary")
        if selected_tissues:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(selected_tissues)}</h3>
                <p>Tissues Selected</p>
            </div>
            """, unsafe_allow_html=True)
            
            if len(selected_tissues) <= 5:
                st.markdown("**Selected tissues:**")
                for tissue in selected_tissues:
                    st.markdown(f"• {tissue}")
            else:
                st.markdown(f"**Selected:** {selected_tissues[0]}, {selected_tissues[1]}, ... and {len(selected_tissues)-2} more")
    
    if not selected_tissues:
        st.warning("⚠️ No tissues selected. Please choose at least one tissue to process.")
        return
    
    # === Processing Parameters ===
    st.markdown("""
    <div class="analysis-section">
        <h3>⚙️ Processing Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider(
            "🎚️ Gene Expression Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Threshold to consider a gene as 'expressed' (0.0 = all genes, 1.0 = only highly expressed)"
        )
        
        st.markdown(f"""
        **Current threshold:** `{threshold}`
        
        - **Lower values** (0.1-0.3): Include more genes, detect subtle changes
        - **Medium values** (0.4-0.6): Balanced approach (recommended)
        - **Higher values** (0.7-1.0): Only highly expressed genes
        """)
    
    with col2:
        st.markdown("### 🎯 Expected Output")
        st.markdown(f"""
        **For each tissue, you'll get:**
        - 📁 Raw gene sets (Ensembl IDs)
        - 🏷️ Mapped gene sets (Gene symbols)
        - 📊 5 age groups per tissue (30-79 years)
        
        **Total files:** {len(selected_tissues) * 2} files
        """)
    
    # === Generation Process ===
    st.markdown("""
    <div class="analysis-section">
        <h3>🚀 File Generation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Generate STAMP Files", type="primary", use_container_width=True):
        st.session_state.tissues = selected_tissues
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("🔄 Generating STAMP files..."):
                # Step 1: Update threshold in script
                status_text.text("📝 Updating processing parameters...")
                progress_bar.progress(10)
                
                with open(SCRIPT_STEP3A, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                with open(SCRIPT_STEP3A, "w", encoding="utf-8") as f:
                    for line in lines:
                        if line.strip().startswith("THRESHOLD ="):
                            f.write(f"THRESHOLD = {threshold}  # threshold to consider a gene 'expressed'\n")
                        else:
                            f.write(line)
                
                # Step 2: Run processing scripts
                status_text.text("⚙️ Running data processing pipeline...")
                progress_bar.progress(30)
                
                subprocess.run(["python", SCRIPT_STEP2], check=True)
                progress_bar.progress(50)
                
                subprocess.run(["python", SCRIPT_STEP3A], check=True)
                progress_bar.progress(70)
                
                # Step 3: Gene mapping
                status_text.text("🧬 Mapping gene symbols...")
                
                def load_gene_map():
                    gene_map = {}
                    with open(GENE_MAP_FILE, encoding="utf-8") as f:
                        for line in f:
                            if "(" in line and ")" in line:
                                ensembl = line.split("(")[0].strip()
                                symbol = line.split("(")[1].replace(")", "").strip()
                                gene_map[ensembl] = symbol
                    return gene_map
                
                os.makedirs(DIR_SETS_MAPPED, exist_ok=True)
                gene_map = load_gene_map()
                
                # Process each tissue
                for i, tissue in enumerate(selected_tissues):
                    status_text.text(f"🔬 Processing tissue: {tissue} ({i+1}/{len(selected_tissues)})")
                    
                    input_path = os.path.join(DIR_SETS, f"{tissue}_sets_stamp.txt")
                    output_path = os.path.join(DIR_SETS_MAPPED, f"{tissue}_sets_stamp_mapped.txt")
                    
                    if os.path.exists(input_path):
                        with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
                            for line in fin:
                                mapped_genes = [gene_map.get(g, g) for g in line.strip().split()]
                                fout.write(" ".join(mapped_genes) + "\n")
                    
                    progress_bar.progress(70 + (i + 1) * 25 // len(selected_tissues))
                
                progress_bar.progress(100)
                status_text.text("✅ Generation completed successfully!")
                
                st.session_state.processed = True
                st.success("🎉 All STAMP files have been generated successfully!")
                
                # Show summary
                st.markdown("### 📊 Generation Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(selected_tissues)}</h3>
                        <p>Tissues Processed</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{threshold}</h3>
                        <p>Expression Threshold</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_files = len([f for f in os.listdir(DIR_SETS_MAPPED) if f.endswith("_sets_stamp_mapped.txt")])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{total_files}</h3>
                        <p>Files Generated</p>
                    </div>
                    """, unsafe_allow_html=True)
                
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error during processing: {e}")
            st.error("Please check that all required scripts are present and executable.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
    
    # === Download Section ===
    if st.session_state.processed:
        display_download_section("📥 Download Generated Files")
        
        # Collect available files
        download_options = []
        for tissue in st.session_state.tissues:
            path1 = os.path.join(DIR_SETS, f"{tissue}_sets_stamp.txt")
            path2 = os.path.join(DIR_SETS_MAPPED, f"{tissue}_sets_stamp_mapped.txt")
            
            if os.path.exists(path1):
                download_options.append((f"{tissue}_sets_stamp.txt", path1, "Raw (Ensembl IDs)"))
            if os.path.exists(path2):
                download_options.append((f"{tissue}_sets_stamp_mapped.txt", path2, "Mapped (Gene Symbols)"))
        
        if download_options:
            st.markdown("### 📁 Available Files")
            
            # Show file list with descriptions
            file_df_data = []
            for name, path, desc in download_options:
                file_size = os.path.getsize(path) / 1024  # KB
                file_df_data.append({
                    "File Name": name,
                    "Type": desc,
                    "Size (KB)": f"{file_size:.1f}",
                    "Tissue": name.split("_")[0]
                })
            
            import pandas as pd
            file_df = pd.DataFrame(file_df_data)
            st.dataframe(file_df, use_container_width=True)
            
            # File selection for download
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Select Files to Download")
                selected_files = st.multiselect(
                    "Choose files:",
                    [x[0] for x in download_options],
                    default=st.session_state.selected_files,
                    help="Select one or more files to download",
                    key="file_selector"
                )
                
                # Update session state
                st.session_state.selected_files = selected_files
            
            with col2:
                st.markdown("#### ⚡ Quick Selection")
                if st.button("📊 Select All Mapped Files", use_container_width=True):
                    st.session_state.selected_files = [x[0] for x in download_options if "mapped" in x[0]]
                    st.rerun()
                
                if st.button("🧬 Select All Raw Files", use_container_width=True):
                    st.session_state.selected_files = [x[0] for x in download_options if "mapped" not in x[0]]
                    st.rerun()
                
                if st.button("📁 Select All Files", use_container_width=True):
                    st.session_state.selected_files = [x[0] for x in download_options]
                    st.rerun()
            
            # Download buttons
            if selected_files:
                st.markdown("### ⬇️ Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Individual file downloads
                    st.markdown("**Individual Downloads:**")
                    for name, path, desc in download_options:
                        if name in selected_files:
                            with open(path, 'rb') as f:
                                st.download_button(
                                    label=f"📄 {name}",
                                    data=f.read(),
                                    file_name=name,
                                    mime='text/plain',
                                    key=f"download_{name}"
                                )
                
                with col2:
                    # ZIP download
                    st.markdown("**Bulk Download:**")
                    if st.button("📦 Create ZIP Archive", use_container_width=True):
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                            for name, path, desc in download_options:
                                if name in selected_files:
                                    zipf.write(path, arcname=name)
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            label="📥 Download ZIP Archive",
                            data=zip_buffer.getvalue(),
                            file_name="stamp_files_generated.zip",
                            mime='application/zip',
                            use_container_width=True,
                            key="download_zip"
                        )
            else:
                st.info("👆 Select files above to enable download options.")
        
        # Reset option
        st.markdown("---")
        if st.button("🔄 Start New Generation", type="secondary", use_container_width=True):
            st.session_state.processed = False
            st.session_state.tissues = []
            st.session_state.selected_files = []
            st.rerun()
    
    # === Help Section ===
    with st.expander("❓ Need Help?"):
        st.markdown("""
        ### 🆘 Troubleshooting
        
        **Common Issues:**
        
        1. **Missing prerequisites**: Ensure all required scripts and data directories are present
        2. **Processing errors**: Check that Python scripts have proper permissions
        3. **No output files**: Verify that input data is properly formatted
        
        ### 📚 File Formats
        
        **Generated STAMP files contain:**
        - 5 lines (one per age group: 30-39, 40-49, 50-59, 60-69, 70-79)
        - Space-separated gene names/IDs per line
        - Compatible with all STAMP analysis modules
        
        ### 🔧 Parameters
        
        **Expression Threshold:**
        - Determines which genes are considered "active" in each age group
        - Higher values = more stringent filtering
        - Recommended: 0.4-0.6 for balanced analysis
        """)