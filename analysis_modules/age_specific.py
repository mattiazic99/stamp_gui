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
    """Age-Specific Analysis Page"""
    
    st.header("📅 Age-Specific Analysis")
    st.markdown("Analyze gene expression patterns for specific age groups across multiple tissues.")
    
    age_groups = ["30–39", "40–49", "50–59", "60–69", "70–79"]
    
    # File upload section
    st.markdown("""
    <div class="analysis-section">
        <h3>📂 Upload Multiple Tissue Files</h3>
        <p>Upload multiple tissue files to analyze age-specific patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📂 Upload STAMP .txt files", 
        type=["txt"],
        accept_multiple_files=True, 
        key="age_specific_files",
        help="Upload multiple tissue gene switching files"
    )
    
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("👆 Please upload at least 2 tissue files for age-specific analysis.")
        
        # Show age group information
        st.markdown("### 📋 Age Group Information")
        age_info_df = pd.DataFrame({
            'Age Group': age_groups,
            'Age Range': ['30-39 years', '40-49 years', '50-59 years', '60-69 years', '70-79 years'],
            'Life Stage': ['Early Adult', 'Middle Adult', 'Late Middle Age', 'Early Senior', 'Senior'],
            'Description': [
                'Peak physical performance period',
                'Career establishment phase',
                'Pre-retirement transition',
                'Early retirement phase',
                'Advanced aging period'
            ]
        })
        st.dataframe(age_info_df, use_container_width=True)
        return
    
    st.success(f"✅ {len(uploaded_files)} tissue files loaded successfully!")
    
    # Parse all files USANDO LA FUNZIONE CHE GIÀ PULISCE I NOMI
    result = parse_multiple_stamp_files(uploaded_files, age_groups)
    data_parsed = result['data']
    summary = result['summary']
    
    # Converti in formato compatibile con le funzioni di analisi
    data = {tissue: data_parsed[tissue]['gene_sets'] for tissue in data_parsed.keys()}
    tissues = list(data.keys())  # Nomi già puliti
    
    # Age group selection
    st.markdown("### 🎯 Select Age Group for Analysis")
    selected_age = st.selectbox(
        "Choose age group:",
        age_groups,
        index=2,  # Default to middle age (50-59)
        help="Select the specific age group to analyze across all tissues"
    )
    
    selected_idx = age_groups.index(selected_age)
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        show_dendrograms = st.checkbox("🌳 Show dendrogram", value=True)
        color_palette = st.selectbox("🎨 Color Palette", 
                                   ["rocket", "mako", "flare", "crest", "viridis"])
    with col2:
        clustering_method = st.selectbox("🔗 Clustering Method", 
                                       ["average", "complete", "single", "ward"])
        analysis_type = st.selectbox("📊 Analysis Focus", 
                                   ["Similarity", "Gene Counts", "Both"])
    
    # === AGE-SPECIFIC OVERVIEW ===
    st.markdown(f"""
    <div class="analysis-section">
        <h2>📊 Age Group {selected_age} - Tissue Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate statistics for selected age group
    age_stats = []
    for tissue in tissues:
        genes_in_age = data[tissue][selected_idx]
        total_genes_tissue = len(set.union(*data[tissue]))
        percentage_in_age = (len(genes_in_age) / total_genes_tissue * 100) if total_genes_tissue > 0 else 0
        
        age_stats.append({
            'Tissue': tissue,  # Nome già pulito
            'Genes in Age Group': len(genes_in_age),
            'Total Tissue Genes': total_genes_tissue,
            'Percentage in Age': f"{percentage_in_age:.1f}%"
        })
    
    df_age_stats = pd.DataFrame(age_stats)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_age_genes = sum([len(data[t][selected_idx]) for t in tissues])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_age_genes}</h3>
            <p>Total Genes<br>in Age Group</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_age_genes = len(set.union(*[data[t][selected_idx] for t in tissues]))
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unique_age_genes}</h3>
            <p>Unique Genes<br>in Age Group</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_genes = np.mean([len(data[t][selected_idx]) for t in tissues])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_genes:.1f}</h3>
            <p>Average Genes<br>per Tissue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_genes = max([len(data[t][selected_idx]) for t in tissues])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{max_genes}</h3>
            <p>Max Genes<br>in One Tissue</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display detailed statistics table
    st.dataframe(df_age_stats, use_container_width=True)
    
    # Bar chart of gene counts per tissue for selected age
    if analysis_type in ["Gene Counts", "Both"]:
        st.markdown(f"### 📊 Gene Counts by Tissue (Age Group {selected_age})")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        apply_plot_style()
        
        gene_counts = [len(data[tissue][selected_idx]) for tissue in tissues]
        colors = plt.cm.Set3(np.linspace(0, 1, len(tissues)))
        bars = ax.bar(tissues, gene_counts, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f"Gene Expression by Tissue - Age Group {selected_age}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel("Number of Switching Genes", fontsize=12, fontweight='bold')
        ax.set_xlabel("Tissue", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        create_download_button(fig, f"gene_counts_age_{selected_age.replace('–','_')}.png")
    
    # === SIMILARITY ANALYSIS FOR SELECTED AGE ===
    if analysis_type in ["Similarity", "Both"]:
        st.markdown(f"""
        <div class="analysis-section">
            <h2>🔍 Similarity Analysis - Age Group {selected_age}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Compute similarity matrix for selected age
        age_data = {tissue: [data[tissue][selected_idx]] for tissue in tissues}
        matrix_age_specific, tissues_sorted = compute_jaccard_matrix(age_data, mode="life")
        
        # Similarity heatmap
        st.markdown(f"### 📊 Tissue Similarity Matrix (Age {selected_age})")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        apply_plot_style()
        
        mask = np.triu(np.ones_like(matrix_age_specific, dtype=bool), k=1)
        heatmap = sns.heatmap(
            matrix_age_specific,
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
        
        format_heatmap(ax, f"Tissue Similarity - Age Group {selected_age}", 
                      "Tissue", "Tissue", "Jaccard Similarity")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        create_download_button(fig, f"similarity_age_{selected_age.replace('–','_')}.png")
        
        # Hierarchical clustering for selected age
        if show_dendrograms:
            st.markdown(f"### 🌳 Hierarchical Clustering (Age {selected_age})")
            
            linkage_age_specific = compute_linkage(matrix_age_specific, method=clustering_method)
            
            if linkage_age_specific is not None:
                fig, ax = plt.subplots(figsize=(12, 6))
                apply_plot_style()
                
                dendrogram(linkage_age_specific, labels=tissues_sorted, ax=ax, leaf_rotation=45)
                ax.set_title(f"Tissue Clustering - Age Group {selected_age} ({clustering_method.title()} Linkage)", 
                            fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel("Distance (1 - Similarity)", fontsize=12, fontweight='bold')
                ax.set_xlabel("Tissue", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                create_download_button(fig, f"dendrogram_age_{selected_age.replace('–','_')}.png")
        
        # Tissue ranking for selected age
        st.markdown(f"### 🏆 Tissue Similarity Ranking (Age {selected_age})")
        
        if len(tissues) >= 3:
            ref_tissue = st.selectbox("🔍 Choose reference tissue:", 
                                     tissues_sorted, key="age_ranking_ref")
            
            ref_idx = tissues_sorted.index(ref_tissue)
            similarities = []
            
            for i, tissue in enumerate(tissues_sorted):
                if tissue != ref_tissue:
                    similarity = matrix_age_specific[ref_idx, i] * 100
                    similarities.append((tissue, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Create ranking chart
            fig, ax = plt.subplots(figsize=(10, 6))
            apply_plot_style()
            
            tissues_rank = [s[0] for s in similarities]
            similarities_rank = [s[1] for s in similarities]
            colors = plt.cm.viridis(np.linspace(0, 1, len(similarities)))
            
            bars = ax.barh(range(len(similarities)), similarities_rank, 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            ax.set_yticks(range(len(similarities)))
            ax.set_yticklabels(tissues_rank)
            ax.set_xlabel("Similarity (%)", fontsize=12, fontweight='bold')
            ax.set_title(f"Similarity Ranking - Age {selected_age} (Ref: {ref_tissue})", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{width:.1f}%', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            create_download_button(fig, f"ranking_age_{selected_age.replace('–','_')}_{ref_tissue}.png")
    
    # === AGE COMPARISON ACROSS TISSUES ===
    st.markdown("""
    <div class="analysis-section">
        <h2>📈 Age Group Comparison</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Compare selected age with other age groups
    st.markdown("### 📊 Gene Count Across All Age Groups")
    
    # Create matrix of gene counts per age per tissue
    age_comparison_data = []
    for tissue in tissues:
        for i, age in enumerate(age_groups):
            gene_count = len(data[tissue][i])
            is_selected = (age == selected_age)
            age_comparison_data.append({
                'Tissue': tissue,  # Nome già pulito
                'Age Group': age,
                'Gene Count': gene_count,
                'Selected': is_selected
            })
    
    df_age_comparison = pd.DataFrame(age_comparison_data)
    
    # Heatmap of gene counts across all ages and tissues
    pivot_data = df_age_comparison.pivot(index='Tissue', columns='Age Group', values='Gene Count')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    apply_plot_style()
    
    heatmap = sns.heatmap(
        pivot_data,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={'label': 'Gene Count'},
        linewidths=0.5
    )
    
    # Highlight selected age group
    selected_col = age_groups.index(selected_age)
    ax.add_patch(plt.Rectangle((selected_col, 0), 1, len(tissues), 
                              fill=False, edgecolor='red', linewidth=3))
    
    format_heatmap(ax, f"Gene Counts Across Age Groups (Highlighted: {selected_age})", 
                  "Age Group", "Tissue", "Gene Count")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    create_download_button(fig, f"age_comparison_heatmap_{selected_age.replace('–','_')}.png")
    
    # === DOWNLOAD SECTION ===
    display_download_section("📥 Download Age-Specific Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Data Tables**")
        create_csv_download(df_age_stats, f"age_stats_{selected_age.replace('–','_')}.csv", 
                           "⬇️ Age Statistics CSV")
        create_csv_download(pivot_data, f"age_comparison_matrix_{selected_age.replace('–','_')}.csv", 
                           "⬇️ Age Comparison Matrix CSV")
    
    with col2:
        st.markdown("**🔍 Similarity Data**")
        if 'matrix_age_specific' in locals():
            df_similarity = pd.DataFrame(matrix_age_specific, 
                                       index=tissues_sorted, columns=tissues_sorted)
            create_csv_download(df_similarity, f"similarity_matrix_{selected_age.replace('–','_')}.csv", 
                               "⬇️ Similarity Matrix CSV")
    
    with col3:
        st.markdown("**🧬 Gene Lists**")
        # Create comprehensive gene list for selected age
        all_age_genes = []
        for tissue in tissues:
            tissue_genes = data[tissue][selected_idx]
            for gene in tissue_genes:
                all_age_genes.append({'Gene': gene, 'Tissue': tissue, 'Age Group': selected_age})
        
        if all_age_genes:
            df_all_genes = pd.DataFrame(all_age_genes)
            create_csv_download(df_all_genes, f"all_genes_{selected_age.replace('–','_')}.csv", 
                               "⬇️ All Genes CSV")