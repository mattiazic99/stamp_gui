import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.parsing import parse_multiple_stamp_files  # USO PARSING CENTRALIZZATO
from components.downloads import create_download_button, create_csv_download, display_download_section
from components.styling import apply_plot_style, format_heatmap, create_styled_figure

def show():
    """Single Gene Analysis Page"""
    
    st.header("🔍 Single Gene Analysis")
    st.markdown("Track individual gene expression patterns across tissues and age groups.")
    
    age_groups = ["30–39", "40–49", "50–59", "60–69", "70–79"]
    
    # File upload section
    st.markdown("""
    <div class="analysis-section">
        <h3>📂 Upload Multiple Tissue Files</h3>
        <p>Upload tissue files to track gene expression patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📂 Upload STAMP .txt files", 
        type=["txt"],
        accept_multiple_files=True, 
        key="single_gene_files",
        help="Upload multiple tissue gene switching files"
    )
    
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("👆 Please upload at least 2 tissue files for single gene analysis.")
        
        # Show example analysis
        st.markdown("### 🔍 Single Gene Analysis Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Gene Tracking:**
            - Gene presence across tissues
            - Age-specific expression patterns
            - Tissue-specific appearance
            - Expression timeline analysis
            """)
        with col2:
            st.markdown("""
            **📊 Visualization Options:**
            - Presence heatmaps
            - Timeline plots
            - Tissue distribution charts
            - Age pattern analysis
            """)
        
        # Show example genes
        st.markdown("### 💡 Example Gene Searches")
        example_genes = ["APOE", "TP53", "BRCA1", "EGFR", "MYC", "PTEN", "RB1", "VHL"]
        st.info(f"Common genes to search: {', '.join(example_genes)}")
        return
    
    st.success(f"✅ {len(uploaded_files)} tissue files loaded successfully!")
    
    # Parse all files USANDO LA FUNZIONE CENTRALIZZATA
    result = parse_multiple_stamp_files(uploaded_files, age_groups)
    data_parsed = result['data']
    summary = result['summary']
    
    # Estrai tessuti e converti formato per compatibilità
    tissues = list(data_parsed.keys())  # Nomi già puliti!
    data = {}
    for tissue in tissues:
        data[tissue] = data_parsed[tissue]['gene_sets']
    
    # Create comprehensive gene list
    all_genes = set()
    for tissue in tissues:
        for age_set in data[tissue]:
            all_genes.update(age_set)
    
    all_genes = sorted(list(all_genes))
    
    # Gene search section
    st.markdown("""
    <div class="analysis-section">
        <h2>🔬 Gene Search and Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gene input methods
        search_method = st.radio(
            "🔍 Gene Search Method:",
            ["Manual Input", "Select from List", "Multiple Genes"],
            help="Choose how to specify genes for analysis"
        )
        
        if search_method == "Manual Input":
            gene_input = st.text_input(
                "🧬 Enter gene name (e.g., APOE):",
                placeholder="Type gene name...",
                help="Enter a single gene name to analyze"
            ).strip().upper()
            genes_to_analyze = [gene_input] if gene_input else []
            
        elif search_method == "Select from List":
            gene_input = st.selectbox(
                "🧬 Select gene from available list:",
                [""] + all_genes,
                help="Choose from genes present in your datasets"
            )
            genes_to_analyze = [gene_input] if gene_input else []
            
        else:  # Multiple Genes
            genes_input = st.text_area(
                "🧬 Enter multiple gene names (one per line):",
                placeholder="APOE\nTP53\nBRCA1",
                help="Enter multiple gene names, one per line"
            )
            genes_to_analyze = [g.strip().upper() for g in genes_input.split('\n') if g.strip()]
    
    with col2:
        st.markdown("### 📊 Dataset Overview")
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(all_genes)}</h3>
            <p>Total Unique Genes</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(tissues)}</h3>
            <p>Tissues Available</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(age_groups)}</h3>
            <p>Age Groups</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis options
    if genes_to_analyze and genes_to_analyze != ['']:
        st.markdown("### ⚙️ Analysis Options")
        
        col1, col2 = st.columns(2)
        with col1:
            show_heatmap = st.checkbox("🔥 Show presence heatmap", value=True)
            show_timeline = st.checkbox("📈 Show timeline analysis", value=True)
        with col2:
            show_statistics = st.checkbox("📊 Show detailed statistics", value=True)
            show_tissue_distribution = st.checkbox("🧪 Show tissue distribution", value=True)
        
        # Filter genes that exist in the dataset
        valid_genes = [gene for gene in genes_to_analyze if gene in all_genes]
        invalid_genes = [gene for gene in genes_to_analyze if gene not in all_genes]
        
        if invalid_genes:
            st.warning(f"⚠️ Genes not found in dataset: {', '.join(invalid_genes)}")
        
        if not valid_genes:
            st.error("❌ No valid genes found in the dataset. Please check gene names.")
            return
        
        st.success(f"✅ Analyzing {len(valid_genes)} gene(s): {', '.join(valid_genes)}")
        
        # === MAIN ANALYSIS ===
        for gene_idx, gene in enumerate(valid_genes):
            if len(valid_genes) > 1:
                st.markdown(f"""
                <div class="analysis-section">
                    <h2>🧬 Analysis for Gene: {gene}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Create presence matrix for this gene
            presence_matrix = np.zeros((len(tissues), len(age_groups)), dtype=int)
            presence_text = pd.DataFrame(index=tissues, columns=age_groups)
            
            for i, tissue in enumerate(tissues):
                for j, age_idx in enumerate(range(len(age_groups))):
                    is_present = gene in data[tissue][age_idx]
                    presence_matrix[i, j] = 1 if is_present else 0
                    presence_text.iloc[i, j] = "✅" if is_present else "❌"
            
            # Gene statistics
            total_appearances = np.sum(presence_matrix)
            tissues_with_gene = np.sum(np.any(presence_matrix, axis=1))
            ages_with_gene = np.sum(np.any(presence_matrix, axis=0))
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_appearances}</h3>
                    <p>Total<br>Appearances</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{tissues_with_gene}</h3>
                    <p>Tissues with<br>Gene</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{ages_with_gene}</h3>
                    <p>Age Groups<br>with Gene</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                expression_rate = (total_appearances / (len(tissues) * len(age_groups))) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{expression_rate:.1f}%</h3>
                    <p>Expression<br>Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Presence heatmap
            if show_heatmap:
                st.markdown(f"### 🔥 Presence Heatmap - {gene}")
                
                fig, ax = plt.subplots(figsize=(10, max(6, len(tissues) * 0.4)))
                apply_plot_style()
                
                heatmap = sns.heatmap(
                    presence_matrix,
                    annot=presence_text.values,
                    fmt="",
                    xticklabels=age_groups,
                    yticklabels=tissues,  # Ora i nomi sono puliti!
                    cmap="RdYlGn",
                    cbar_kws={'label': 'Gene Presence (0=Absent, 1=Present)'},
                    linewidths=0.5,
                    ax=ax
                )
                
                format_heatmap(ax, f"Gene Presence Pattern: {gene}", 
                              "Age Group", "Tissue", "Presence")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                create_download_button(fig, f"heatmap_{gene}_presence.png")
            
            # Timeline analysis
            if show_timeline:
                st.markdown(f"### 📈 Timeline Analysis - {gene}")
                
                # Count appearances per age group
                age_counts = np.sum(presence_matrix, axis=0)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                apply_plot_style()
                
                # Line plot of gene expression across ages
                ax1.plot(age_groups, age_counts, marker='o', linewidth=3, 
                        markersize=8, color='#e74c3c')
                ax1.fill_between(age_groups, age_counts, alpha=0.3, color='#e74c3c')
                ax1.set_title(f"Gene Expression Timeline: {gene}", 
                             fontsize=14, fontweight='bold')
                ax1.set_ylabel("Number of Tissues", fontsize=12, fontweight='bold')
                ax1.set_xlabel("Age Group", fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, len(tissues))
                
                # Add value labels
                for i, count in enumerate(age_counts):
                    ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', 
                           fontweight='bold')
                
                # Bar plot of tissue distribution
                tissue_counts = np.sum(presence_matrix, axis=1)
                colors = plt.cm.viridis(np.linspace(0, 1, len(tissues)))
                bars = ax2.barh(tissues, tissue_counts, color=colors, alpha=0.8,  # Nomi puliti!
                               edgecolor='black', linewidth=1)
                
                ax2.set_title(f"Tissue Distribution: {gene}", 
                             fontsize=14, fontweight='bold')
                ax2.set_xlabel("Number of Age Groups", fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')
                ax2.set_xlim(0, len(age_groups))
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                           f'{int(width)}', ha='left', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                create_download_button(fig, f"timeline_{gene}_analysis.png")
            
            # Detailed statistics table
            if show_statistics:
                st.markdown(f"### 📊 Detailed Statistics - {gene}")
                
                # Create detailed stats
                stats_data = []
                for tissue in tissues:
                    tissue_appearances = np.sum(presence_matrix[tissues.index(tissue), :])
                    age_list = [age_groups[j] for j in range(len(age_groups)) 
                              if presence_matrix[tissues.index(tissue), j] == 1]
                    
                    stats_data.append({
                        'Tissue': tissue,  # Nome pulito!
                        'Appearances': tissue_appearances,
                        'Expression Rate': f"{(tissue_appearances/len(age_groups)*100):.1f}%",
                        'Age Groups': ', '.join(age_list) if age_list else 'None'
                    })
                
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True)
                
                # Age group analysis
                st.markdown("#### 📅 Age Group Analysis")
                age_stats_data = []
                for i, age in enumerate(age_groups):
                    age_appearances = np.sum(presence_matrix[:, i])
                    tissue_list = [tissues[j] for j in range(len(tissues)) 
                                 if presence_matrix[j, i] == 1]
                    
                    age_stats_data.append({
                        'Age Group': age,
                        'Tissues with Gene': age_appearances,
                        'Expression Rate': f"{(age_appearances/len(tissues)*100):.1f}%",
                        'Tissues': ', '.join(tissue_list) if tissue_list else 'None'  # Nomi puliti!
                    })
                
                df_age_stats = pd.DataFrame(age_stats_data)
                st.dataframe(df_age_stats, use_container_width=True)
            
            # Tissue distribution pie chart
            if show_tissue_distribution and total_appearances > 0:
                st.markdown(f"### 🧪 Tissue Expression Distribution - {gene}")
                
                # Get tissues that express the gene
                expressing_tissues = [(tissues[i], np.sum(presence_matrix[i, :])) 
                                    for i in range(len(tissues)) 
                                    if np.sum(presence_matrix[i, :]) > 0]
                
                if expressing_tissues:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    apply_plot_style()
                    
                    # Pie chart
                    tissue_names = [t[0] for t in expressing_tissues]  # Nomi puliti!
                    counts = [t[1] for t in expressing_tissues]
                    colors = plt.cm.Set3(np.linspace(0, 1, len(expressing_tissues)))
                    
                    wedges, texts, autotexts = ax1.pie(counts, labels=tissue_names, 
                                                      autopct='%1.1f%%', colors=colors,
                                                      startangle=90)
                    ax1.set_title(f"Expression Distribution by Tissue: {gene}", 
                                 fontsize=14, fontweight='bold')
                    
                    # Bar chart
                    bars = ax2.bar(tissue_names, counts, color=colors, alpha=0.8,  # Nomi puliti!
                                  edgecolor='black', linewidth=1.2)
                    ax2.set_title(f"Expression Frequency by Tissue: {gene}", 
                                 fontsize=14, fontweight='bold')
                    ax2.set_ylabel("Number of Age Groups", fontsize=12, fontweight='bold')
                    ax2.set_xlabel("Tissue", fontsize=12, fontweight='bold')
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    create_download_button(fig, f"distribution_{gene}_tissues.png")
            
            # Download section for this gene
            if len(valid_genes) > 1:
                st.markdown(f"#### 📥 Downloads for {gene}")
            
            col1, col2 = st.columns(2)
            with col1:
                # Presence matrix CSV
                create_csv_download(presence_text, f"presence_matrix_{gene}.csv", 
                                   f"⬇️ {gene} Presence Matrix CSV")
            with col2:
                # Statistics CSV
                if 'df_stats' in locals():
                    create_csv_download(df_stats, f"statistics_{gene}.csv", 
                                       f"⬇️ {gene} Statistics CSV")
            
            # Add separator between genes
            if gene_idx < len(valid_genes) - 1:
                st.markdown("---")
        
        # === MULTI-GENE COMPARISON ===
        if len(valid_genes) > 1:
            st.markdown("""
            <div class="analysis-section">
                <h2>🔄 Multi-Gene Comparison</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Create comparison matrix
            comparison_data = []
            for gene in valid_genes:
                gene_presence = np.zeros((len(tissues), len(age_groups)), dtype=int)
                for i, tissue in enumerate(tissues):
                    for j, age_idx in enumerate(range(len(age_groups))):
                        gene_presence[i, j] = 1 if gene in data[tissue][age_idx] else 0
                
                total_appearances = np.sum(gene_presence)
                tissues_with_gene = np.sum(np.any(gene_presence, axis=1))
                ages_with_gene = np.sum(np.any(gene_presence, axis=0))
                expression_rate = (total_appearances / (len(tissues) * len(age_groups))) * 100
                
                comparison_data.append({
                    'Gene': gene,
                    'Total Appearances': total_appearances,
                    'Tissues with Gene': tissues_with_gene,
                    'Age Groups with Gene': ages_with_gene,
                    'Expression Rate (%)': f"{expression_rate:.1f}%"
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Comparison bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            apply_plot_style()
            
            x = np.arange(len(valid_genes))
            width = 0.25
            
            appearances = [int(row['Total Appearances']) for _, row in df_comparison.iterrows()]
            tissues_count = [int(row['Tissues with Gene']) for _, row in df_comparison.iterrows()]
            ages_count = [int(row['Age Groups with Gene']) for _, row in df_comparison.iterrows()]
            
            bars1 = ax.bar(x - width, appearances, width, label='Total Appearances', 
                          alpha=0.8, color='#3498db')
            bars2 = ax.bar(x, tissues_count, width, label='Tissues with Gene', 
                          alpha=0.8, color='#e74c3c')
            bars3 = ax.bar(x + width, ages_count, width, label='Age Groups with Gene', 
                          alpha=0.8, color='#2ecc71')
            
            ax.set_xlabel('Genes', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Multi-Gene Expression Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(valid_genes)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            create_download_button(fig, "multi_gene_comparison.png")
            
            # Download multi-gene comparison
            create_csv_download(df_comparison, "multi_gene_comparison.csv", 
                               "⬇️ Multi-Gene Comparison CSV")
        
        # === GLOBAL DOWNLOAD SECTION ===
        if len(valid_genes) > 0:
            display_download_section("📥 Download All Results")
            
            # Create comprehensive summary
            summary_data = {
                'Analysis_Type': ['Single Gene Analysis'],
                'Genes_Analyzed': [', '.join(valid_genes)],
                'Number_of_Genes': [len(valid_genes)],
                'Number_of_Tissues': [len(tissues)],
                'Number_of_Age_Groups': [len(age_groups)],
                'Total_Data_Points': [len(tissues) * len(age_groups) * len(valid_genes)]
            }
            summary_df = pd.DataFrame(summary_data)
            create_csv_download(summary_df, "gene_analysis_summary.csv", 
                               "⬇️ Analysis Summary CSV")
    
    else:
        st.info("👆 Please enter at least one gene name to start the analysis.")
        
        # Show some statistics about available genes
        if all_genes:
            st.markdown("### 📊 Available Genes Overview")
            
            # Show random sample of genes
            sample_size = min(20, len(all_genes))
            sample_genes = np.random.choice(all_genes, sample_size, replace=False)
            
            st.markdown(f"**Sample of available genes ({sample_size} of {len(all_genes)}):**")
            st.text(", ".join(sorted(sample_genes)))
            
            # Show most common genes (genes that appear in most tissue-age combinations)
            gene_frequency = {}
            for gene in all_genes:
                count = 0
                for tissue in tissues:
                    for age_set in data[tissue]:
                        if gene in age_set:
                            count += 1
                gene_frequency[gene] = count
            
            # Top 10 most frequent genes
            top_genes = sorted(gene_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_genes:
                st.markdown("### 🏆 Most Frequently Expressed Genes")
                freq_df = pd.DataFrame(top_genes, columns=['Gene', 'Frequency'])
                freq_df['Expression Rate (%)'] = (freq_df['Frequency'] / (len(tissues) * len(age_groups)) * 100).round(1)
                st.dataframe(freq_df, use_container_width=True)