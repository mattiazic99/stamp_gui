import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.analysis import compute_common_genes_matrix, compute_percent_overlap_matrix
from utils.parsing import parse_multiple_stamp_files, extract_tissue_name  # AGGIUNTO IMPORT
from components.downloads import create_download_button, create_csv_download, display_download_section
from components.styling import apply_plot_style, format_heatmap, create_styled_figure

def show():
    """Gene Sharing Analysis Page"""
    
    st.header("🤝 Gene Sharing Analysis")
    st.markdown("Comprehensive analysis of shared and exclusive genes between tissues and age groups.")
    
    age_groups = ["30–39", "40–49", "50–59", "60–69", "70–79"]
    
    # File upload section
    st.markdown("""
    <div class="analysis-section">
        <h3>📂 Upload Multiple Tissue Files</h3>
        <p>Upload tissue files to analyze gene sharing patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📂 Upload STAMP .txt files", 
        type=["txt"],
        accept_multiple_files=True, 
        key="gene_sharing_files",
        help="Upload multiple tissue gene switching files"
    )
    
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("👆 Please upload at least 2 tissue files for gene sharing analysis.")
        
        # Show analysis features
        st.markdown("### 🔍 Gene Sharing Analysis Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📊 Sharing Metrics:**
            - Absolute shared gene counts
            - Percentage overlap analysis
            - Jaccard similarity indices
            - Exclusive gene identification
            """)
        with col2:
            st.markdown("""
            **🔎 Analysis Options:**
            - Whole lifespan sharing
            - Age-specific sharing
            - Pairwise comparisons
            - Multi-tissue intersections
            """)
        return
    
    st.success(f"✅ {len(uploaded_files)} tissue files loaded successfully!")
    
    # Parse all files USANDO LA FUNZIONE CHE GIÀ PULISCE I NOMI
    result = parse_multiple_stamp_files(uploaded_files, age_groups)
    data_parsed = result['data']
    summary = result['summary']
    
    # Converti in formato compatibile con le funzioni di analisi
    data = {tissue: data_parsed[tissue]['gene_sets'] for tissue in data_parsed.keys()}
    tissues = list(data.keys())  # Nomi già puliti
    
    # Analysis options
    st.markdown("### ⚙️ Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_scope = st.selectbox("📊 Analysis Scope", 
                                    ["Whole Lifespan", "Age-Specific", "Both"],
                                    help="Choose the temporal scope for analysis")
        sharing_metric = st.selectbox("📈 Sharing Metric", 
                                    ["Absolute Count", "Percentage", "Both"],
                                    help="Choose how to measure gene sharing")
    
    with col2:
        comparison_type = st.selectbox("🔍 Comparison Type", 
                                     ["All Pairwise", "Selected Pairs", "Multi-way"],
                                     help="Choose comparison methodology")
        if analysis_scope in ["Age-Specific", "Both"]:
            selected_age = st.selectbox("🎯 Age Group", age_groups, index=2)
    
    # === WHOLE LIFESPAN ANALYSIS ===
    if analysis_scope in ["Whole Lifespan", "Both"]:
        st.markdown("""
        <div class="analysis-section">
            <h2>🔗 Whole Lifespan Gene Sharing</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate lifespan gene sets for each tissue
        lifespan_genes = {}
        for tissue in tissues:
            lifespan_genes[tissue] = set.union(*data[tissue])
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_unique = len(set.union(*lifespan_genes.values()))
        avg_genes_per_tissue = np.mean([len(genes) for genes in lifespan_genes.values()])
        max_genes = max([len(genes) for genes in lifespan_genes.values()])
        min_genes = min([len(genes) for genes in lifespan_genes.values()])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_unique}</h3>
                <p>Total Unique<br>Genes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_genes_per_tissue:.0f}</h3>
                <p>Avg Genes<br>per Tissue</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{max_genes}</h3>
                <p>Max Genes<br>(One Tissue)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{min_genes}</h3>
                <p>Min Genes<br>(One Tissue)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Absolute shared genes matrix
        if sharing_metric in ["Absolute Count", "Both"]:
            st.markdown("### 🔢 Absolute Shared Gene Counts")
            
            matrix_common, tissues_sorted = compute_common_genes_matrix(data)
            
            if matrix_common.size > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                apply_plot_style()
                
                mask = np.triu(np.ones_like(matrix_common, dtype=bool), k=1)
                heatmap = sns.heatmap(
                    matrix_common,
                    annot=True,
                    fmt="d",
                    xticklabels=tissues_sorted,
                    yticklabels=tissues_sorted,
                    cmap="Purples",
                    ax=ax,
                    mask=mask,
                    square=True,
                    cbar_kws={'label': 'Shared Gene Count'},
                    linewidths=0.5
                )
                
                format_heatmap(ax, "Shared Genes Between Tissues (Absolute Counts)", 
                              "Tissue", "Tissue", "Shared Gene Count")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                create_download_button(fig, "shared_genes_absolute_lifespan.png")
        
        # Percentage overlap matrix
        if sharing_metric in ["Percentage", "Both"]:
            st.markdown("### 📊 Percentage Overlap Analysis")
            
            matrix_percent, tissues_sorted = compute_percent_overlap_matrix(data)
            
            if matrix_percent.size > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                apply_plot_style()
                
                mask = np.triu(np.ones_like(matrix_percent, dtype=bool), k=1)
                heatmap = sns.heatmap(
                    matrix_percent,
                    annot=True,
                    fmt=".1f",
                    xticklabels=tissues_sorted,
                    yticklabels=tissues_sorted,
                    cmap="crest",
                    ax=ax,
                    mask=mask,
                    square=True,
                    cbar_kws={'label': 'Overlap Percentage (%)'},
                    linewidths=0.5
                )
                
                format_heatmap(ax, "Gene Overlap Percentage Between Tissues", 
                              "Tissue", "Tissue", "Overlap Percentage (%)")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                create_download_button(fig, "overlap_percentage_lifespan.png")
        
        # Detailed pairwise sharing table
        st.markdown("### 📋 Detailed Pairwise Sharing Statistics")
        
        pairwise_data = []
        for i, tissue1 in enumerate(tissues_sorted):
            for j, tissue2 in enumerate(tissues_sorted):
                if i < j:  # Only upper triangle
                    genes1 = lifespan_genes[tissue1]
                    genes2 = lifespan_genes[tissue2]
                    
                    shared = len(genes1 & genes2)
                    exclusive1 = len(genes1 - genes2)
                    exclusive2 = len(genes2 - genes1)
                    total_union = len(genes1 | genes2)
                    jaccard = shared / total_union if total_union > 0 else 0
                    
                    pairwise_data.append({
                        'Tissue 1': tissue1,
                        'Tissue 2': tissue2,
                        'Shared Genes': shared,
                        f'{tissue1} Exclusive': exclusive1,
                        f'{tissue2} Exclusive': exclusive2,
                        'Jaccard Similarity': f"{jaccard:.3f}",
                        'Total Union': total_union
                    })
        
        df_pairwise = pd.DataFrame(pairwise_data)
        st.dataframe(df_pairwise, use_container_width=True)
        
        # Top sharing pairs
        st.markdown("### 🏆 Top Gene Sharing Pairs")
        
        df_sorted = df_pairwise.sort_values('Shared Genes', ascending=False)
        top_pairs = df_sorted.head(5)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        apply_plot_style()
        
        pair_labels = [f"{row['Tissue 1']}\nvs\n{row['Tissue 2']}" 
                      for _, row in top_pairs.iterrows()]
        shared_counts = top_pairs['Shared Genes'].values
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_pairs)))
        bars = ax.bar(range(len(top_pairs)), shared_counts, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(range(len(top_pairs)))
        ax.set_xticklabels(pair_labels, fontsize=10)
        ax.set_title("Top 5 Tissue Pairs by Shared Genes", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Number of Shared Genes", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        create_download_button(fig, "top_sharing_pairs_lifespan.png")
    
    # === AGE-SPECIFIC ANALYSIS ===
    if analysis_scope in ["Age-Specific", "Both"]:
        st.markdown(f"""
        <div class="analysis-section">
            <h2>📅 Age-Specific Gene Sharing (Age {selected_age})</h2>
        </div>
        """, unsafe_allow_html=True)
        
        selected_idx = age_groups.index(selected_age)
        
        # Age-specific statistics
        age_genes = {tissue: data[tissue][selected_idx] for tissue in tissues}
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_age_unique = len(set.union(*age_genes.values()))
        avg_age_genes = np.mean([len(genes) for genes in age_genes.values()])
        max_age_genes = max([len(genes) for genes in age_genes.values()])
        min_age_genes = min([len(genes) for genes in age_genes.values()])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_age_unique}</h3>
                <p>Unique Genes<br>in Age {selected_age}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_age_genes:.0f}</h3>
                <p>Avg Genes<br>per Tissue</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{max_age_genes}</h3>
                <p>Max Genes<br>(One Tissue)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{min_age_genes}</h3>
                <p>Min Genes<br>(One Tissue)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Age-specific shared genes matrix
        if sharing_metric in ["Absolute Count", "Both"]:
            st.markdown(f"### 🔢 Shared Gene Counts (Age {selected_age})")
            
            matrix_age_common, tissues_sorted = compute_common_genes_matrix(data, idx=selected_idx)
            
            if matrix_age_common.size > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                apply_plot_style()
                
                mask = np.triu(np.ones_like(matrix_age_common, dtype=bool), k=1)
                heatmap = sns.heatmap(
                    matrix_age_common,
                    annot=True,
                    fmt="d",
                    xticklabels=tissues_sorted,
                    yticklabels=tissues_sorted,
                    cmap="mako",
                    ax=ax,
                    mask=mask,
                    square=True,
                    cbar_kws={'label': 'Shared Gene Count'},
                    linewidths=0.5
                )
                
                format_heatmap(ax, f"Shared Genes - Age Group {selected_age}", 
                              "Tissue", "Tissue", "Shared Gene Count")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                create_download_button(fig, f"shared_genes_age_{selected_age.replace('–','_')}.png")
        
        # Age-specific percentage overlap
        if sharing_metric in ["Percentage", "Both"]:
            st.markdown(f"### 📊 Overlap Percentage (Age {selected_age})")
            
            matrix_age_percent, tissues_sorted = compute_percent_overlap_matrix(data, idx=selected_idx)
            
            if matrix_age_percent.size > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                apply_plot_style()
                
                mask = np.triu(np.ones_like(matrix_age_percent, dtype=bool), k=1)
                heatmap = sns.heatmap(
                    matrix_age_percent,
                    annot=True,
                    fmt=".1f",
                    xticklabels=tissues_sorted,
                    yticklabels=tissues_sorted,
                    cmap="vlag",
                    ax=ax,
                    mask=mask,
                    square=True,
                    cbar_kws={'label': 'Overlap Percentage (%)'},
                    linewidths=0.5
                )
                
                format_heatmap(ax, f"Gene Overlap Percentage - Age {selected_age}", 
                              "Tissue", "Tissue", "Overlap Percentage (%)")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                create_download_button(fig, f"overlap_percentage_age_{selected_age.replace('–','_')}.png")
    
    # === PAIRWISE DETAILED COMPARISON ===
    if comparison_type in ["Selected Pairs", "All Pairwise"]:
        st.markdown("""
        <div class="analysis-section">
            <h2>🔍 Detailed Pairwise Comparison</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if comparison_type == "Selected Pairs":
            col1, col2 = st.columns(2)
            with col1:
                tissue1 = st.selectbox("🧪 Select First Tissue", tissues, key="pair_t1")
            with col2:
                tissue2 = st.selectbox("🧪 Select Second Tissue", 
                                      [t for t in tissues if t != tissue1], key="pair_t2")
        else:
            # Show top 3 most similar pairs
            if 'df_pairwise' in locals():
                top_3_pairs = df_sorted.head(3)
                st.markdown("### 🏆 Top 3 Most Similar Tissue Pairs")
                
                for idx, (_, row) in enumerate(top_3_pairs.iterrows()):
                    tissue1, tissue2 = row['Tissue 1'], row['Tissue 2']
                    
                    with st.expander(f"#{idx+1}: {tissue1} vs {tissue2} ({row['Shared Genes']} shared genes)"):
                        show_pairwise_analysis(tissue1, tissue2, data, age_groups, 
                                             analysis_scope, selected_age if 'selected_age' in locals() else None)
        
        if comparison_type == "Selected Pairs":
            show_pairwise_analysis(tissue1, tissue2, data, age_groups, 
                                 analysis_scope, selected_age if 'selected_age' in locals() else None)
    
    # === MULTI-WAY INTERSECTIONS ===
    if comparison_type == "Multi-way" and len(tissues) >= 3:
        st.markdown("""
        <div class="analysis-section">
            <h2>🔄 Multi-way Gene Intersections</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Allow user to select tissues for intersection
        selected_tissues = st.multiselect(
            "🎯 Select tissues for intersection analysis:",
            tissues,
            default=tissues[:3] if len(tissues) >= 3 else tissues,
            help="Choose 3+ tissues to analyze their gene intersections"
        )
        
        if len(selected_tissues) >= 3:
            show_multiway_analysis(selected_tissues, data, age_groups, 
                                 analysis_scope, selected_age if 'selected_age' in locals() else None)
    
    # === DOWNLOAD SECTION ===
    display_download_section("📥 Download Gene Sharing Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Sharing Matrices**")
        if 'matrix_common' in locals() and matrix_common.size > 0:
            df_common = pd.DataFrame(matrix_common, index=tissues_sorted, columns=tissues_sorted)
            create_csv_download(df_common, "shared_genes_matrix_lifespan.csv", 
                               "⬇️ Shared Genes Matrix CSV")
        
        if 'matrix_percent' in locals() and matrix_percent.size > 0:
            df_percent = pd.DataFrame(matrix_percent, index=tissues_sorted, columns=tissues_sorted)
            create_csv_download(df_percent, "overlap_percentage_matrix_lifespan.csv", 
                               "⬇️ Overlap Percentage CSV")
    
    with col2:
        st.markdown("**📋 Detailed Tables**")
        if 'df_pairwise' in locals():
            create_csv_download(df_pairwise, "pairwise_sharing_statistics.csv", 
                               "⬇️ Pairwise Statistics CSV")
    
    with col3:
        st.markdown("**🧬 Gene Lists**")
        # Create comprehensive gene sharing summary
        summary_data = []
        for tissue in tissues:
            genes_lifespan = set.union(*data[tissue])
            for gene in genes_lifespan:
                # Count in how many tissues this gene appears
                tissues_with_gene = [t for t in tissues if gene in set.union(*data[t])]
                summary_data.append({
                    'Gene': gene,
                    'Primary_Tissue': tissue,
                    'Total_Tissues': len(tissues_with_gene),
                    'Tissues_List': ', '.join(tissues_with_gene)
                })
        
        if summary_data:
            df_gene_summary = pd.DataFrame(summary_data)
            df_gene_summary = df_gene_summary.drop_duplicates('Gene')
            create_csv_download(df_gene_summary, "gene_sharing_summary.csv", 
                               "⬇️ Gene Sharing Summary CSV")


def show_pairwise_analysis(tissue1, tissue2, data, age_groups, analysis_scope, selected_age):
    """Show detailed pairwise analysis between two tissues"""
    
    # Whole lifespan comparison
    if analysis_scope in ["Whole Lifespan", "Both"]:
        genes1_all = set.union(*data[tissue1])
        genes2_all = set.union(*data[tissue2])
        
        shared_all = genes1_all & genes2_all
        exclusive1_all = genes1_all - genes2_all
        exclusive2_all = genes2_all - genes1_all
        
        st.markdown(f"#### 🔗 Whole Lifespan: {tissue1} vs {tissue2}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(shared_all)}</h3>
                <p>Shared Genes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(exclusive1_all)}</h3>
                <p>{tissue1} Exclusive</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(exclusive2_all)}</h3>
                <p>{tissue2} Exclusive</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Pie chart
        if len(shared_all) + len(exclusive1_all) + len(exclusive2_all) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            apply_plot_style()
            
            sizes = [len(shared_all), len(exclusive1_all), len(exclusive2_all)]
            labels = [f'Shared\n({len(shared_all)})', 
                     f'{tissue1} Exclusive\n({len(exclusive1_all)})', 
                     f'{tissue2} Exclusive\n({len(exclusive2_all)})']
            colors = ['#2ecc71', '#3498db', '#e74c3c']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                            startangle=90, colors=colors)
            ax.set_title(f"Gene Distribution: {tissue1} vs {tissue2}", 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            create_download_button(fig, f"pie_chart_{tissue1}_vs_{tissue2}_lifespan.png")
    
    # Age-specific comparison
    if analysis_scope in ["Age-Specific", "Both"] and selected_age:
        selected_idx = age_groups.index(selected_age)
        
        genes1_age = data[tissue1][selected_idx]
        genes2_age = data[tissue2][selected_idx]
        
        shared_age = genes1_age & genes2_age
        exclusive1_age = genes1_age - genes2_age
        exclusive2_age = genes2_age - genes1_age
        
        st.markdown(f"#### 📅 Age {selected_age}: {tissue1} vs {tissue2}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(shared_age)}</h3>
                <p>Shared Genes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(exclusive1_age)}</h3>
                <p>{tissue1} Exclusive</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(exclusive2_age)}</h3>
                <p>{tissue2} Exclusive</p>
            </div>
            """, unsafe_allow_html=True)


def show_multiway_analysis(selected_tissues, data, age_groups, analysis_scope, selected_age):
    """Show multi-way intersection analysis"""
    
    st.markdown(f"### 🔄 Intersection Analysis: {', '.join(selected_tissues)}")
    
    if analysis_scope in ["Whole Lifespan", "Both"]:
        # Calculate intersections for whole lifespan
        tissue_gene_sets = [set.union(*data[tissue]) for tissue in selected_tissues]
        
        # Core intersection (genes in ALL tissues)
        core_intersection = set.intersection(*tissue_gene_sets)
        
        # Union (genes in ANY tissue)
        total_union = set.union(*tissue_gene_sets)
        
        st.markdown(f"#### 🔗 Whole Lifespan Multi-way Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(core_intersection)}</h3>
                <p>Core Genes<br>(All Tissues)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(total_union)}</h3>
                <p>Total Unique<br>Genes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            overlap_ratio = len(core_intersection) / len(total_union) if len(total_union) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{overlap_ratio:.2%}</h3>
                <p>Core Overlap<br>Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        if len(core_intersection) > 0:
            st.markdown("**🎯 Core Genes (present in all selected tissues):**")
            st.text(", ".join(sorted(core_intersection)[:50]) + ("..." if len(core_intersection) > 50 else ""))
            
            # Download core genes
            core_df = pd.DataFrame(sorted(core_intersection), columns=['Gene'])
            create_csv_download(core_df, f"core_genes_{'_'.join(selected_tissues[:3])}.csv", 
                               "⬇️ Download Core Genes CSV")