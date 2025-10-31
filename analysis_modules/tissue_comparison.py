import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.parsing import parse_stamp_file, extract_tissue_name  # Import della funzione di pulizia
from components.downloads import create_download_button, create_csv_download, display_download_section
from components.styling import apply_plot_style, format_heatmap, create_styled_figure

def show():
    """Tissue Comparison Analysis Page"""
    
    st.header("🔄 Tissue Comparison Analysis")
    st.markdown("Compare gene expression patterns between two specific tissues across age groups.")
    
    age_groups = ["30–39", "40–49", "50–59", "60–69", "70–79"]
    
    # File upload section
    st.markdown("""
    <div class="analysis-section">
        <h3>📂 Upload Two Tissues for Comparison</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader(
            "📊 First Tissue", 
            type=["txt"], 
            key="tissue1_comp",
            help="Upload first tissue gene switching data"
        )
    with col2:
        file2 = st.file_uploader(
            "📊 Second Tissue", 
            type=["txt"], 
            key="tissue2_comp",
            help="Upload second tissue gene switching data"
        )
    
    if not (file1 and file2):
        st.info("👆 Please upload both tissue files to start the comparison.")
        return
    
    # Parse files
    fasce1, counts1, df1 = parse_stamp_file(file1, age_groups)
    fasce2, counts2, df2 = parse_stamp_file(file2, age_groups)
    
    # USA LA FUNZIONE DI PULIZIA PER I NOMI
    tissue1_name = extract_tissue_name(file1.name)
    tissue2_name = extract_tissue_name(file2.name)
    
    # Age group selection
    st.markdown("### 🎯 Age Group Selection")
    selected_ages = st.multiselect(
        "Select age groups for comparison:",
        age_groups, 
        default=age_groups,
        help="Choose which age groups to include in the comparison"
    )
    
    if not selected_ages:
        st.warning("⚠️ Please select at least one age group.")
        return
    
    # Filter data
    df1_filtered = df1[df1["Age"].isin(selected_ages)]
    df2_filtered = df2[df2["Age"].isin(selected_ages)]
    
    counts1_filtered = [
        df1_filtered[df1_filtered["Age"] == age].shape[0]
        for age in age_groups if age in selected_ages
    ]
    counts2_filtered = [
        df2_filtered[df2_filtered["Age"] == age].shape[0]
        for age in age_groups if age in selected_ages
    ]
    
    # === COMPARISON METRICS ===
    st.markdown("""
    <div class="analysis-section">
        <h2>📊 Comparison Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate overall statistics
    total_genes_1 = len(set(df1["Gene"]))
    total_genes_2 = len(set(df2["Gene"]))
    common_genes = len(set(df1["Gene"]) & set(df2["Gene"]))
    unique_genes = len(set(df1["Gene"]) | set(df2["Gene"]))
    jaccard_sim = common_genes / unique_genes if unique_genes > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_genes_1}</h3>
            <p>{tissue1_name}<br>Total Genes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_genes_2}</h3>
            <p>{tissue2_name}<br>Total Genes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{common_genes}</h3>
            <p>Shared<br>Genes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{jaccard_sim:.2%}</h3>
            <p>Jaccard<br>Similarity</p>
        </div>
        """, unsafe_allow_html=True)
    
    # === SIDE-BY-SIDE COMPARISON CHART ===
    st.markdown("### 📊 Gene Count Comparison by Age Group")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    apply_plot_style()
    
    x = np.arange(len(selected_ages))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts1_filtered, width,
                  label=tissue1_name, color="#3498db", alpha=0.8, 
                  edgecolor='#2980b9', linewidth=1.5)
    bars2 = ax.bar(x + width/2, counts2_filtered, width,
                  label=tissue2_name, color="#e74c3c", alpha=0.8, 
                  edgecolor='#c0392b', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
    
    ax.set_xticks(x)
    ax.set_xticklabels(selected_ages)
    ax.set_title(f"Gene Expression Comparison: {tissue1_name} vs {tissue2_name}", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Switching Genes", fontsize=12, fontweight='bold')
    ax.set_xlabel("Age Group", fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    create_download_button(fig, f"comparison_{tissue1_name}_vs_{tissue2_name}.png")
    
    # === SHARED VS EXCLUSIVE ANALYSIS ===
    st.markdown("""
    <div class="analysis-section">
        <h2>🤝 Shared vs Exclusive Gene Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate for whole lifespan
    genes_1_all = set(df1["Gene"])
    genes_2_all = set(df2["Gene"])
    shared_genes_all = sorted(genes_1_all & genes_2_all)
    exclusive_1_all = sorted(genes_1_all - genes_2_all)
    exclusive_2_all = sorted(genes_2_all - genes_1_all)
    
    # Pie chart of gene distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    apply_plot_style()
    
    # Pie chart
    sizes = [len(shared_genes_all), len(exclusive_1_all), len(exclusive_2_all)]
    labels = [f'Shared\n({len(shared_genes_all)})', 
              f'Exclusive to\n{tissue1_name}\n({len(exclusive_1_all)})', 
              f'Exclusive to\n{tissue2_name}\n({len(exclusive_2_all)})']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, colors=colors)
    ax1.set_title(f"Gene Distribution: {tissue1_name} vs {tissue2_name}", 
                 fontsize=14, fontweight='bold')
    
    # Bar chart comparison
    categories = ['Shared', f'{tissue1_name}\nExclusive', f'{tissue2_name}\nExclusive']
    values = sizes
    bars = ax2.bar(categories, values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    ax2.set_title("Gene Count by Category", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Number of Genes", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    create_download_button(fig, f"shared_exclusive_{tissue1_name}_vs_{tissue2_name}.png")
    
    # === AGE-SPECIFIC ANALYSIS ===
    st.markdown("### 📅 Age-Specific Shared Genes Analysis")
    
    # Calculate shared genes for each age group
    age_shared_data = []
    for age in age_groups:
        genes_1_age = set(df1[df1["Age"] == age]["Gene"])
        genes_2_age = set(df2[df2["Age"] == age]["Gene"])
        shared_age = len(genes_1_age & genes_2_age)
        total_1_age = len(genes_1_age)
        total_2_age = len(genes_2_age)
        union_age = len(genes_1_age | genes_2_age)
        jaccard_age = shared_age / union_age if union_age > 0 else 0
        
        age_shared_data.append({
            'Age Group': age,
            f'{tissue1_name} Genes': total_1_age,
            f'{tissue2_name} Genes': total_2_age,
            'Shared Genes': shared_age,
            'Jaccard Similarity': jaccard_age
        })
    
    df_age_analysis = pd.DataFrame(age_shared_data)
    
    # Display table
    st.dataframe(df_age_analysis, use_container_width=True)
    
    # Line chart of Jaccard similarity across ages
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_plot_style()
    
    ax.plot(df_age_analysis['Age Group'], df_age_analysis['Jaccard Similarity'], 
           marker='o', linewidth=3, markersize=8, color='#9b59b6')
    ax.fill_between(df_age_analysis['Age Group'], df_age_analysis['Jaccard Similarity'], 
                   alpha=0.3, color='#9b59b6')
    
    ax.set_title(f"Jaccard Similarity Across Age Groups: {tissue1_name} vs {tissue2_name}", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Jaccard Similarity", fontsize=12, fontweight='bold')
    ax.set_xlabel("Age Group", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(df_age_analysis['Jaccard Similarity']) * 1.1)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    create_download_button(fig, f"jaccard_similarity_{tissue1_name}_vs_{tissue2_name}.png")
    
    # === DOWNLOAD SECTION ===
    display_download_section("📥 Download Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Data Tables**")
        create_csv_download(df_age_analysis, f"age_analysis_{tissue1_name}_vs_{tissue2_name}.csv", 
                           "⬇️ Age Analysis CSV")
        
        # Combined dataset
        df1_labeled = df1.copy()
        df1_labeled['Tissue'] = tissue1_name
        df2_labeled = df2.copy()
        df2_labeled['Tissue'] = tissue2_name
        combined_df = pd.concat([df1_labeled, df2_labeled], ignore_index=True)
        create_csv_download(combined_df, f"combined_{tissue1_name}_{tissue2_name}.csv", 
                           "⬇️ Combined Dataset CSV")
    
    with col2:
        st.markdown("**🤝 Shared Genes**")
        shared_df = pd.DataFrame(shared_genes_all, columns=['Gene'])
        create_csv_download(shared_df, f"shared_genes_{tissue1_name}_{tissue2_name}.csv", 
                           "⬇️ Shared Genes CSV")
        
        # Show preview
        if len(shared_genes_all) > 0:
            st.text(f"Preview: {', '.join(shared_genes_all[:5])}{'...' if len(shared_genes_all) > 5 else ''}")
    
    with col3:
        st.markdown("**🧬 Exclusive Genes**")
        exclusive_1_df = pd.DataFrame(exclusive_1_all, columns=['Gene'])
        create_csv_download(exclusive_1_df, f"exclusive_{tissue1_name}.csv", 
                           f"⬇️ {tissue1_name} Exclusive CSV")
        
        exclusive_2_df = pd.DataFrame(exclusive_2_all, columns=['Gene'])
        create_csv_download(exclusive_2_df, f"exclusive_{tissue2_name}.csv", 
                           f"⬇️ {tissue2_name} Exclusive CSV")
    
    # === DETAILED GENE LISTS ===
    st.markdown("""
    <div class="analysis-section">
        <h2>📋 Detailed Gene Lists</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🤝 Shared Genes", f"🧬 {tissue1_name} Exclusive", f"🧬 {tissue2_name} Exclusive"])
    
    with tab1:
        st.markdown(f"**{len(shared_genes_all)} genes shared between both tissues**")
        if shared_genes_all:
            st.text(", ".join(shared_genes_all))
    
    with tab2:
        st.markdown(f"**{len(exclusive_1_all)} genes exclusive to {tissue1_name}**")
        if exclusive_1_all:
            st.text(", ".join(exclusive_1_all))
    
    with tab3:
        st.markdown(f"**{len(exclusive_2_all)} genes exclusive to {tissue2_name}**")
        if exclusive_2_all:
            st.text(", ".join(exclusive_2_all))