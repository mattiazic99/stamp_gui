import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from utils.analysis import compute_jaccard_matrix, compute_linkage, compute_common_genes_matrix, compute_percent_overlap_matrix
from utils.parsing import parse_multiple_stamp_files, extract_tissue_name  # AGGIUNTO IMPORT
from components.downloads import create_download_button, create_csv_download, display_download_section
from components.styling import apply_plot_style, format_heatmap, create_styled_figure

def show():
    """Multi-Tissue Analysis Page"""
    
    st.header("🧬 Multi-Tissue Analysis")
    st.markdown("Comprehensive similarity analysis and hierarchical clustering across multiple tissues.")
    
    age_groups = ["30–39", "40–49", "50–59", "60–69", "70–79"]
    
    # File upload section
    st.markdown("""
    <div class="analysis-section">
        <h3>📂 Upload Multiple Tissue Files</h3>
        <p>Upload at least 3 tissue files for comprehensive multi-tissue analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📂 Upload STAMP .txt files", 
        type=["txt"],
        accept_multiple_files=True, 
        key="multi_tissue_files",
        help="Upload multiple tissue gene switching files for comparison"
    )
    
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("👆 Please upload at least 2 tissue files for multi-tissue analysis.")
        
        # Show example
        st.markdown("### 📋 Analysis Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🔍 Similarity Analysis:**
            - Jaccard similarity matrices
            - Age-averaged comparisons
            - Lifetime gene overlap
            - Correlation heatmaps
            """)
        with col2:
            st.markdown("""
            **🌳 Clustering Analysis:**
            - Hierarchical clustering
            - Dendrogram visualization
            - Distance matrices
            - Tissue grouping
            """)
        return
    
    st.success(f"✅ {len(uploaded_files)} tissue files loaded successfully!")
    
    # Parse all files USANDO LA FUNZIONE DI PARSING CHE GIÀ PULISCE I NOMI
    result = parse_multiple_stamp_files(uploaded_files, age_groups)
    data = result['data']
    summary = result['summary']
    
    # Estrai i nomi puliti dei tessuti
    tissues = list(data.keys())  # Questi sono già puliti dalla funzione parse_multiple_stamp_files
    
    # Analysis options
    st.markdown("### ⚙️ Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        show_dendrograms = st.checkbox("🌳 Show dendrograms", value=True, 
                                      help="Display hierarchical clustering dendrograms")
        similarity_metric = st.selectbox("📊 Similarity Metric", 
                                       ["Jaccard Index", "Overlap Coefficient"], 
                                       help="Choose similarity calculation method")
    
    with col2:
        clustering_method = st.selectbox("🔗 Clustering Method", 
                                       ["average", "complete", "single", "ward"], 
                                       help="Hierarchical clustering linkage method")
        color_palette = st.selectbox("🎨 Color Palette", 
                                   ["coolwarm", "viridis", "RdYlBu", "plasma"],
                                   help="Choose color scheme for heatmaps")
    
    # === TISSUE OVERVIEW ===
    st.markdown("""
    <div class="analysis-section">
        <h2>📊 Tissue Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate basic statistics usando i nomi puliti
    tissue_stats = []
    for tissue in tissues:
        gene_sets = data[tissue]['gene_sets']
        total_genes = len(set.union(*gene_sets)) if gene_sets else 0
        avg_genes_per_age = np.mean([len(age_set) for age_set in gene_sets]) if gene_sets else 0
        max_genes_age = max([len(age_set) for age_set in gene_sets]) if gene_sets else 0
        min_genes_age = min([len(age_set) for age_set in gene_sets]) if gene_sets else 0
        
        tissue_stats.append({
            'Tissue': tissue,  # Nome già pulito
            'Total Unique Genes': total_genes,
            'Avg Genes/Age': f"{avg_genes_per_age:.1f}",
            'Max Genes (Age)': max_genes_age,
            'Min Genes (Age)': min_genes_age
        })
    
    df_stats = pd.DataFrame(tissue_stats)
    st.dataframe(df_stats, use_container_width=True)
    
    # Converti per compatibilità con le funzioni di analisi
    data_for_analysis = {tissue: data[tissue]['gene_sets'] for tissue in tissues}
    
    # Bar chart of total genes per tissue
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_plot_style()
    
    total_genes = [int(row['Total Unique Genes']) for row in tissue_stats]
    bars = ax.bar(tissues, total_genes, color=plt.cm.Set3(np.linspace(0, 1, len(tissues))), 
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Total Unique Genes per Tissue", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Unique Genes", fontsize=12, fontweight='bold')
    ax.set_xlabel("Tissue", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    create_download_button(fig, "tissue_overview.png")
    
    # === SIMILARITY MATRICES ===
    st.markdown("""
    <div class="analysis-section">
        <h2>📊 Similarity Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Compute similarity matrices
    matrix_age, tissues_sorted = compute_jaccard_matrix(data_for_analysis, mode="age")
    matrix_life, _ = compute_jaccard_matrix(data_for_analysis, mode="life")
    
    # Age-averaged similarity heatmap
    st.markdown("### 📊 Average Similarity Across Age Groups")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    apply_plot_style()
    
    mask = np.triu(np.ones_like(matrix_age, dtype=bool), k=1)
    heatmap = sns.heatmap(
        matrix_age, 
        annot=True, 
        fmt=".3f",
        xticklabels=tissues_sorted, 
        yticklabels=tissues_sorted,
        cmap=color_palette, 
        ax=ax,
        mask=mask,
        square=True,
        cbar_kws={'label': 'Jaccard Similarity'},
        linewidths=0.5
    )
    
    format_heatmap(ax, "Tissue Similarity (Averaged Over Age Groups)", 
                  "Tissue", "Tissue", "Jaccard Similarity")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    create_download_button(fig, "similarity_age_averaged.png")
    
    # Lifetime similarity heatmap
    st.markdown("### 📊 Lifetime Similarity Matrix")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    apply_plot_style()
    
    mask = np.triu(np.ones_like(matrix_life, dtype=bool), k=1)
    heatmap = sns.heatmap(
        matrix_life, 
        annot=True, 
        fmt=".3f",
        xticklabels=tissues_sorted, 
        yticklabels=tissues_sorted,
        cmap="YlGnBu", 
        ax=ax,
        mask=mask,
        square=True,
        cbar_kws={'label': 'Jaccard Similarity'},
        linewidths=0.5
    )
    
    format_heatmap(ax, "Tissue Similarity (Whole Lifetime)", 
                  "Tissue", "Tissue", "Jaccard Similarity")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    create_download_button(fig, "similarity_lifetime.png")
    
    # === HIERARCHICAL CLUSTERING ===
    if show_dendrograms:
        st.markdown("""
        <div class="analysis-section">
            <h2>🌳 Hierarchical Clustering</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Compute linkages
        linkage_age = compute_linkage(matrix_age, method=clustering_method)
        linkage_life = compute_linkage(matrix_life, method=clustering_method)
        
        if linkage_age is not None:
            st.markdown("### 🌿 Dendrogram - Age-Based Similarity")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            apply_plot_style()
            
            dendrogram(linkage_age, labels=tissues_sorted, ax=ax, leaf_rotation=45)
            ax.set_title(f"Hierarchical Clustering (Age-Based, {clustering_method.title()} Linkage)", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel("Distance (1 - Similarity)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Tissue", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            create_download_button(fig, "dendrogram_age_based.png")
        
        if linkage_life is not None:
            st.markdown("### 🌿 Dendrogram - Lifetime-Based Similarity")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            apply_plot_style()
            
            dendrogram(linkage_life, labels=tissues_sorted, ax=ax, leaf_rotation=45)
            ax.set_title(f"Hierarchical Clustering (Lifetime-Based, {clustering_method.title()} Linkage)", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel("Distance (1 - Similarity)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Tissue", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            create_download_button(fig, "dendrogram_lifetime_based.png")
    
    # === SHARED GENES ANALYSIS ===
    st.markdown("""
    <div class="analysis-section">
        <h2>🤝 Shared Genes Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Absolute shared genes matrix
    st.markdown("### 🔗 Absolute Shared Gene Counts")
    
    matrix_common, _ = compute_common_genes_matrix(data_for_analysis)
    
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
        
        create_download_button(fig, "shared_genes_absolute.png")
    
    # Percentage overlap matrix
    st.markdown("### 📈 Percentage Overlap Analysis")
    
    matrix_percent, _ = compute_percent_overlap_matrix(data_for_analysis)
    
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
        
        create_download_button(fig, "overlap_percentage.png")
    
    # === TISSUE RANKING ===
    st.markdown("""
    <div class="analysis-section">
        <h2>🏆 Tissue Similarity Ranking</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if len(tissues) >= 3:
        ref_tissue = st.selectbox("🔍 Choose reference tissue for ranking:", 
                                 tissues_sorted, key="ranking_ref_multi")
        
        st.markdown(f"### 📋 Similarity Ranking relative to **{ref_tissue}**")
        
        # Get reference tissue index
        ref_idx = tissues_sorted.index(ref_tissue)
        similarities = []
        
        for i, tissue in enumerate(tissues_sorted):
            if tissue != ref_tissue:
                similarity = matrix_life[ref_idx, i] * 100  # Convert to percentage
                similarities.append((tissue, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create ranking dataframe
        ranking_df = pd.DataFrame(similarities, columns=['Tissue', 'Similarity (%)'])
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df[['Rank', 'Tissue', 'Similarity (%)']]
        
        # Display ranking table
        st.dataframe(ranking_df, use_container_width=True)
        
        # Create ranking bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        apply_plot_style()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(similarities)))
        bars = ax.barh(range(len(similarities)), [s[1] for s in similarities], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(similarities)))
        ax.set_yticklabels([s[0] for s in similarities])
        ax.set_xlabel("Similarity (%)", fontsize=12, fontweight='bold')
        ax.set_title(f"Tissue Similarity Ranking (Reference: {ref_tissue})", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        create_download_button(fig, f"ranking_{ref_tissue}.png")
        create_csv_download(ranking_df, f"similarity_ranking_{ref_tissue}.csv", 
                           "⬇️ Download Ranking CSV")
    
    # === DOWNLOAD SECTION ===
    display_download_section("📥 Download Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Similarity Matrices**")
        
        # Age-averaged matrix
        df_age_sim = pd.DataFrame(matrix_age, index=tissues_sorted, columns=tissues_sorted)
        create_csv_download(df_age_sim, "similarity_matrix_age_averaged.csv", 
                           "⬇️ Age-Averaged Matrix CSV")
        
        # Lifetime matrix
        df_life_sim = pd.DataFrame(matrix_life, index=tissues_sorted, columns=tissues_sorted)
        create_csv_download(df_life_sim, "similarity_matrix_lifetime.csv", 
                           "⬇️ Lifetime Matrix CSV")
    
    with col2:
        st.markdown("**🤝 Shared Gene Matrices**")
        
        if matrix_common.size > 0:
            df_common = pd.DataFrame(matrix_common, index=tissues_sorted, columns=tissues_sorted)
            create_csv_download(df_common, "shared_genes_matrix.csv", 
                               "⬇️ Shared Genes Matrix CSV")
        
        if matrix_percent.size > 0:
            df_percent = pd.DataFrame(matrix_percent, index=tissues_sorted, columns=tissues_sorted)
            create_csv_download(df_percent, "overlap_percentage_matrix.csv", 
                               "⬇️ Overlap Percentage CSV")
    
    with col3:
        st.markdown("**📈 Summary Statistics**")
        create_csv_download(df_stats, "tissue_statistics_summary.csv", 
                           "⬇️ Tissue Statistics CSV")
        
        # Create comprehensive summary
        summary_data = {
            'Analysis Type': ['Multi-Tissue Analysis'],
            'Number of Tissues': [len(tissues)],
            'Tissues': [', '.join(tissues_sorted)],
            'Similarity Metric': [similarity_metric],
            'Clustering Method': [clustering_method],
            'Total Unique Genes': [summary.get('total_unique_genes', 0)],
            'Average Similarity': [f"{np.mean(matrix_life[np.triu_indices_from(matrix_life, k=1)]):.3f}"],
            'Max Similarity': [f"{np.max(matrix_life[np.triu_indices_from(matrix_life, k=1)]):.3f}"],
            'Min Similarity': [f"{np.min(matrix_life[np.triu_indices_from(matrix_life, k=1)]):.3f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        create_csv_download(summary_df, "analysis_summary.csv", 
                           "⬇️ Analysis Summary CSV")
    
    # === ADVANCED ANALYSIS ===
    if st.checkbox("🔬 Show Advanced Analysis", value=False):
        st.markdown("""
        <div class="analysis-section">
            <h2>🔬 Advanced Multi-Tissue Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Correlation analysis between tissue gene counts
        st.markdown("### 📊 Gene Count Correlations Across Age Groups")
        
        # Create matrix of gene counts per tissue per age
        count_matrix = np.zeros((len(tissues), len(age_groups)))
        for i, tissue in enumerate(tissues):
            gene_sets = data[tissue]['gene_sets']
            for j, age_idx in enumerate(range(5)):
                if age_idx < len(gene_sets):
                    count_matrix[i, j] = len(gene_sets[age_idx])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(count_matrix)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        apply_plot_style()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        heatmap = sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            xticklabels=tissues,
            yticklabels=tissues,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            mask=mask,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'},
            linewidths=0.5
        )
        
        format_heatmap(ax, "Gene Count Correlation Between Tissues", 
                      "Tissue", "Tissue", "Correlation")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        create_download_button(fig, "gene_count_correlation.png")
        
        # Age group diversity analysis
        st.markdown("### 📅 Age Group Diversity Analysis")
        
        # Calculate Shannon diversity for each tissue across age groups
        diversities = []
        for tissue in tissues:
            gene_sets = data[tissue]['gene_sets']
            counts = [len(gene_sets[i]) for i in range(len(gene_sets))]
            total = sum(counts)
            if total > 0:
                proportions = [c/total for c in counts if c > 0]
                shannon = -sum(p * np.log(p) for p in proportions)
                diversities.append(shannon)
            else:
                diversities.append(0)
        
        diversity_df = pd.DataFrame({
            'Tissue': tissues,
            'Shannon Diversity': diversities
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        apply_plot_style()
        
        bars = ax.bar(diversity_df['Tissue'], diversity_df['Shannon Diversity'], 
                     color=plt.cm.plasma(np.linspace(0, 1, len(tissues))), 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_title("Age Group Diversity by Tissue (Shannon Index)", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Shannon Diversity Index", fontsize=12, fontweight='bold')
        ax.set_xlabel("Tissue", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        create_download_button(fig, "age_diversity_analysis.png")
        create_csv_download(diversity_df, "shannon_diversity.csv", 
                           "⬇️ Download Diversity Data CSV")