import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plots import plot_pie_common_vs_exclusive
from utils.parsing import parse_multiple_stamp_files  # USO PARSING CENTRALIZZATO
from components.downloads import create_download_button, create_csv_download, display_download_section, create_multiple_csv_download
from components.styling import apply_plot_style, format_heatmap, create_styled_figure

def show():
    """Group Comparison Analysis Page"""
    
    st.header("👥 Group Comparison Analysis")
    st.markdown("Compare gene expression patterns between custom tissue groups.")
    
    age_groups = ["30–39", "40–49", "50–59", "60–69", "70–79"]
    
    # File upload section
    st.markdown("""
    <div class="analysis-section">
        <h3>📂 Upload Multiple Tissue Files</h3>
        <p>Upload tissue files to create and compare custom groups</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📂 Upload STAMP .txt files", 
        type=["txt"],
        accept_multiple_files=True, 
        key="group_comparison_files",
        help="Upload multiple tissue gene switching files for group comparison"
    )
    
    if not uploaded_files or len(uploaded_files) < 4:
        st.info("👆 Please upload at least 4 tissue files to enable group comparison analysis.")
        
        # Show group comparison features
        st.markdown("### 👥 Group Comparison Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Group Analysis:**
            - Custom tissue grouping
            - Inter-group comparisons
            - Shared vs exclusive genes
            - Statistical significance testing
            """)
        with col2:
            st.markdown("""
            **📊 Comparison Metrics:**
            - Jaccard similarity
            - Gene overlap percentages
            - Group-specific patterns
            - Multi-group intersections
            """)
        
        # Show example groupings
        st.markdown("### 💡 Example Group Comparisons")
        examples = [
            "**Organ Systems**: Heart, Liver, Kidney vs Brain, Muscle, Lung",
            "**Metabolic vs Structural**: Liver, Pancreas vs Bone, Cartilage", 
            "**Central vs Peripheral**: Brain, Spinal Cord vs Skin, Muscle",
            "**High vs Low Metabolism**: Heart, Brain, Liver vs Bone, Skin"
        ]
        for example in examples:
            st.markdown(f"- {example}")
        
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
    
    # Group creation section
    st.markdown("""
    <div class="analysis-section">
        <h2>👥 Create Tissue Groups</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 Select at least 2 tissues for each group to enable comparison.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔵 Group 1")
        group1 = st.multiselect(
            "Select tissues for Group 1:",
            tissues,  # Nomi puliti!
            key="group1_tissues",
            help="Choose tissues for the first group"
        )
        
        if group1:
            group1_name = st.text_input(
                "Group 1 Name:",
                value="Group 1",
                key="group1_name",
                help="Enter a descriptive name for Group 1"
            )
    
    with col2:
        st.markdown("#### 🔴 Group 2")
        available_for_group2 = [t for t in tissues if t not in group1]
        group2 = st.multiselect(
            "Select tissues for Group 2:",
            available_for_group2,  # Nomi puliti!
            key="group2_tissues",
            help="Choose tissues for the second group"
        )
        
        if group2:
            group2_name = st.text_input(
                "Group 2 Name:",
                value="Group 2",
                key="group2_name",
                help="Enter a descriptive name for Group 2"
            )
    
    # Validation
    if len(group1) < 2 or len(group2) < 2:
        st.warning("⚠️ Please select at least 2 tissues for each group to enable comparison.")
        return
    
    # Analysis scope selection
    st.markdown("### ⚙️ Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_scope = st.selectbox(
            "📊 Analysis Scope:",
            ["Full Lifespan", "Specific Age Group", "All Age Groups Separately"],
            help="Choose the temporal scope for group comparison"
        )
        
        if analysis_scope == "Specific Age Group":
            selected_age = st.selectbox("🎯 Select Age Group:", age_groups, index=2)
    
    with col2:
        comparison_metrics = st.multiselect(
            "📈 Comparison Metrics:",
            ["Jaccard Similarity", "Shared Gene Count", "Overlap Percentage", "Statistical Tests"],
            default=["Jaccard Similarity", "Shared Gene Count"],
            help="Choose which metrics to calculate"
        )
    
    # Group overview
    st.markdown("""
    <div class="analysis-section">
        <h2>📊 Group Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 🔵 {group1_name}")
        st.write(f"**Tissues:** {', '.join(group1)}")  # Nomi puliti!
        
        # Calculate group1 statistics
        group1_genes = set()
        if analysis_scope == "Full Lifespan":
            group1_genes = set.union(*[set.union(*data[tissue]) for tissue in group1])
        elif analysis_scope == "Specific Age Group":
            selected_idx = age_groups.index(selected_age)
            group1_genes = set.union(*[data[tissue][selected_idx] for tissue in group1])
        else:
            # Per l'overview quando si scelgono tutti i gruppi d'età separati,
            # usiamo l'unione su tutto l'arco di vita per evitare variabili non inizializzate.
            group1_genes = set.union(*[set.union(*data[tissue]) for tissue in group1])
        
        st.write(f"**Total Unique Genes:** {len(group1_genes)}")
        st.write(f"**Average Genes per Tissue:** {len(group1_genes) / len(group1):.1f}")
    
    with col2:
        st.markdown(f"#### 🔴 {group2_name}")
        st.write(f"**Tissues:** {', '.join(group2)}")  # Nomi puliti!
        
        # Calculate group2 statistics
        group2_genes = set()
        if analysis_scope == "Full Lifespan":
            group2_genes = set.union(*[set.union(*data[tissue]) for tissue in group2])
        elif analysis_scope == "Specific Age Group":
            group2_genes = set.union(*[data[tissue][selected_idx] for tissue in group2])
        else:
            group2_genes = set.union(*[set.union(*data[tissue]) for tissue in group2])
        
        st.write(f"**Total Unique Genes:** {len(group2_genes)}")
        st.write(f"**Average Genes per Tissue:** {len(group2_genes) / len(group2):.1f}")
    
    # === MAIN COMPARISON ANALYSIS ===
    if analysis_scope in ["Full Lifespan", "Specific Age Group"]:
        perform_group_comparison(
            group1, group2, group1_name, group2_name, 
            group1_genes, group2_genes, data, 
            comparison_metrics, analysis_scope,
            selected_age if analysis_scope == "Specific Age Group" else None,
            age_groups  # <-- passiamo age_groups per evitare NameError
        )
    
    elif analysis_scope == "All Age Groups Separately":
        st.markdown("""
        <div class="analysis-section">
            <h2>📅 Age-by-Age Group Comparison</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Perform comparison for each age group
        age_comparison_results = []
        
        for age_idx, age in enumerate(age_groups):
            st.markdown(f"### 📊 Age Group: {age}")
            
            group1_age_genes = set.union(*[data[tissue][age_idx] for tissue in group1])
            group2_age_genes = set.union(*[data[tissue][age_idx] for tissue in group2])
            
            # Calculate metrics
            shared_genes = group1_age_genes & group2_age_genes
            exclusive1 = group1_age_genes - group2_age_genes
            exclusive2 = group2_age_genes - group1_age_genes
            union_genes = group1_age_genes | group2_age_genes
            
            jaccard_sim = len(shared_genes) / len(union_genes) if len(union_genes) > 0 else 0
            
            age_comparison_results.append({
                'Age Group': age,
                f'{group1_name} Genes': len(group1_age_genes),
                f'{group2_name} Genes': len(group2_age_genes),
                'Shared Genes': len(shared_genes),
                'Jaccard Similarity': jaccard_sim,
                f'{group1_name} Exclusive': len(exclusive1),
                f'{group2_name} Exclusive': len(exclusive2)
            })
            
            # Display age-specific metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(shared_genes)}</h3>
                    <p>Shared Genes</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{jaccard_sim:.3f}</h3>
                    <p>Jaccard Similarity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(exclusive1)}</h3>
                    <p>{group1_name} Exclusive</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(exclusive2)}</h3>
                    <p>{group2_name} Exclusive</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create summary table
        df_age_comparison = pd.DataFrame(age_comparison_results)
        
        st.markdown("### 📋 Age-by-Age Comparison Summary")
        st.dataframe(df_age_comparison, use_container_width=True)
        
        # Plot age-wise trends
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        apply_plot_style()
        
        ages = df_age_comparison['Age Group']
        
        # Jaccard similarity trend
        jaccard_values = df_age_comparison['Jaccard Similarity']
        ax1.plot(ages, jaccard_values, marker='o', linewidth=3, markersize=8, color='#9b59b6')
        ax1.fill_between(ages, jaccard_values, alpha=0.3, color='#9b59b6')
        ax1.set_title('Jaccard Similarity Across Age Groups', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Jaccard Similarity', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Shared genes trend
        shared_values = df_age_comparison['Shared Genes']
        ax2.bar(ages, shared_values, color='#2ecc71', alpha=0.8, edgecolor='black')
        ax2.set_title('Shared Genes Across Age Groups', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Shared Genes', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Group sizes comparison
        group1_values = df_age_comparison[f'{group1_name} Genes']
        group2_values = df_age_comparison[f'{group2_name} Genes']
        
        x = np.arange(len(ages))
        width = 0.35
        
        ax3.bar(x - width/2, group1_values, width, label=group1_name, 
               color='#3498db', alpha=0.8)
        ax3.bar(x + width/2, group2_values, width, label=group2_name, 
               color='#e74c3c', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(ages)
        ax3.set_title('Group Gene Counts by Age', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Genes', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Exclusive genes comparison
        excl1_values = df_age_comparison[f'{group1_name} Exclusive']
        excl2_values = df_age_comparison[f'{group2_name} Exclusive']
        
        ax4.bar(x - width/2, excl1_values, width, label=f'{group1_name} Exclusive', 
               color='#f39c12', alpha=0.8)
        ax4.bar(x + width/2, excl2_values, width, label=f'{group2_name} Exclusive', 
               color='#e67e22', alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(ages)
        ax4.set_title('Exclusive Genes by Age', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Exclusive Genes', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        create_download_button(fig, f"age_comparison_{group1_name}_vs_{group2_name}.png")
        create_csv_download(df_age_comparison, f"age_comparison_{group1_name}_vs_{group2_name}.csv", 
                           "⬇️ Download Age Comparison CSV")
    
    # === DOWNLOAD SECTION ===
    display_download_section("📥 Download Group Comparison Results")
    
    # Create comprehensive download package
    download_data = {}
    
    # Group compositions
    group_composition = pd.DataFrame({
        'Group': [group1_name] * len(group1) + [group2_name] * len(group2),
        'Tissue': group1 + group2  # Nomi puliti!
    })
    download_data['group_composition'] = group_composition
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Group 1 Name', 'Group 2 Name', 'Group 1 Tissues', 'Group 2 Tissues',
                  'Analysis Scope', 'Total Comparisons'],
        'Value': [group1_name, group2_name, ', '.join(group1), ', '.join(group2),  # Nomi puliti!
                 analysis_scope, '1' if analysis_scope != "All Age Groups Separately" else str(len(age_groups))]
    })
    download_data['summary_statistics'] = summary_stats
    
    if 'df_age_comparison' in locals():
        download_data['age_by_age_comparison'] = df_age_comparison
    
    # Create ZIP download
    create_multiple_csv_download(
        download_data,
        f"group_comparison_{group1_name}_vs_{group2_name}.zip",
        "⬇️ Download Complete Analysis Package (ZIP)"
    )


def perform_group_comparison(group1, group2, group1_name, group2_name, 
                           group1_genes, group2_genes, data, 
                           comparison_metrics, analysis_scope, selected_age=None,
                           age_groups=None):
    """Perform detailed comparison between two tissue groups"""
    
    st.markdown(f"""
    <div class="analysis-section">
        <h2>🔄 {group1_name} vs {group2_name} Comparison</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate comparison metrics
    shared_genes = group1_genes & group2_genes
    exclusive1 = group1_genes - group2_genes
    exclusive2 = group2_genes - group1_genes
    union_genes = group1_genes | group2_genes
    
    # Basic metrics
    jaccard_similarity = len(shared_genes) / len(union_genes) if len(union_genes) > 0 else 0
    overlap_percentage = len(shared_genes) / min(len(group1_genes), len(group2_genes)) * 100 if min(len(group1_genes), len(group2_genes)) > 0 else 0
    
    # Display main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(shared_genes)}</h3>
            <p>Shared Genes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{jaccard_similarity:.3f}</h3>
            <p>Jaccard<br>Similarity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{overlap_percentage:.1f}%</h3>
            <p>Overlap<br>Percentage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(exclusive1)}</h3>
            <p>{group1_name}<br>Exclusive</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(exclusive2)}</h3>
            <p>{group2_name}<br>Exclusive</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization section
    st.markdown("### 📊 Visual Comparison")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    apply_plot_style()
    
    # 1. Pie chart of gene distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [len(shared_genes), len(exclusive1), len(exclusive2)]
    labels = [f'Shared\n({len(shared_genes)})', 
              f'{group1_name}\nExclusive\n({len(exclusive1)})', 
              f'{group2_name}\nExclusive\n({len(exclusive2)})']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          startangle=90, colors=colors)
        ax1.set_title(f"Gene Distribution\n{group1_name} vs {group2_name}", 
                     fontsize=12, fontweight='bold')
    
    # 2. Bar chart comparison
    ax2 = fig.add_subplot(gs[0, 1])
    categories = [f'{group1_name}\nTotal', f'{group2_name}\nTotal', 'Shared', 'Union']
    values = [len(group1_genes), len(group2_genes), len(shared_genes), len(union_genes)]
    bars = ax2.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'], 
                   alpha=0.8, edgecolor='black')
    ax2.set_title('Gene Count Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Genes')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Venn diagram representation (simplified)
    ax3 = fig.add_subplot(gs[0, 2])
    # Create a simple representation of overlap
    overlap_data = pd.DataFrame({
        'Category': ['Shared', f'{group1_name} Only', f'{group2_name} Only'],
        'Count': [len(shared_genes), len(exclusive1), len(exclusive2)]
    })
    bars = ax3.barh(overlap_data['Category'], overlap_data['Count'], 
                    color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
    ax3.set_title('Gene Overlap Analysis', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Genes')
    
    # 4. Individual tissue contributions (Group 1)
    ax4 = fig.add_subplot(gs[1, :2])
    tissue_contributions1 = []
    for tissue in group1:
        if analysis_scope == "Full Lifespan":
            tissue_genes = set.union(*data[tissue])
        else:  # Specific age group
            selected_idx = age_groups.index(selected_age)
            tissue_genes = data[tissue][selected_idx]
        
        contribution = len(tissue_genes & shared_genes)
        total_genes = len(tissue_genes)
        tissue_contributions1.append({
            'Tissue': tissue,
            'Shared Contribution': contribution,
            'Total Genes': total_genes,
            'Contribution Rate': contribution / total_genes * 100 if total_genes > 0 else 0
        })
    
    df_contrib1 = pd.DataFrame(tissue_contributions1)
    
    x = np.arange(len(group1))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, df_contrib1['Shared Contribution'], width,
                   label='Shared Genes', color='#2ecc71', alpha=0.8)
    bars2 = ax4.bar(x + width/2, df_contrib1['Total Genes'], width,
                   label='Total Genes', color='#3498db', alpha=0.8)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(group1, rotation=45, ha='right')  # Nomi puliti!
    ax4.set_title(f'{group1_name} - Individual Tissue Contributions', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Genes')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Individual tissue contributions (Group 2)
    ax5 = fig.add_subplot(gs[2, :2])
    tissue_contributions2 = []
    for tissue in group2:
        if analysis_scope == "Full Lifespan":
            tissue_genes = set.union(*data[tissue])
        else:  # Specific age group
            selected_idx = age_groups.index(selected_age)
            tissue_genes = data[tissue][selected_idx]
        
        contribution = len(tissue_genes & shared_genes)
        total_genes = len(tissue_genes)
        tissue_contributions2.append({
            'Tissue': tissue,
            'Shared Contribution': contribution,
            'Total Genes': total_genes,
            'Contribution Rate': contribution / total_genes * 100 if total_genes > 0 else 0
        })
    
    df_contrib2 = pd.DataFrame(tissue_contributions2)
    
    x = np.arange(len(group2))
    
    bars1 = ax5.bar(x - width/2, df_contrib2['Shared Contribution'], width,
                   label='Shared Genes', color='#2ecc71', alpha=0.8)
    bars2 = ax5.bar(x + width/2, df_contrib2['Total Genes'], width,
                   label='Total Genes', color='#e74c3c', alpha=0.8)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(group2, rotation=45, ha='right')  # Nomi puliti!
    ax5.set_title(f'{group2_name} - Individual Tissue Contributions', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Number of Genes')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Similarity heatmap between groups
    ax6 = fig.add_subplot(gs[1:, 2])
    
    # Create similarity matrix between all tissues in both groups
    all_group_tissues = group1 + group2
    similarity_matrix = np.zeros((len(all_group_tissues), len(all_group_tissues)))
    
    for i, tissue1 in enumerate(all_group_tissues):
        for j, tissue2 in enumerate(all_group_tissues):
            if analysis_scope == "Full Lifespan":
                genes1 = set.union(*data[tissue1])
                genes2 = set.union(*data[tissue2])
            else:
                selected_idx = age_groups.index(selected_age)
                genes1 = data[tissue1][selected_idx]
                genes2 = data[tissue2][selected_idx]
            
            intersection = len(genes1 & genes2)
            union = len(genes1 | genes2)
            similarity_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Create group labels for coloring
    group_labels = [group1_name] * len(group1) + [group2_name] * len(group2)
    
    heatmap = sns.heatmap(similarity_matrix, 
                         annot=True, fmt='.2f',
                         xticklabels=all_group_tissues,  # Nomi puliti!
                         yticklabels=all_group_tissues,  # Nomi puliti!
                         cmap='coolwarm',
                         ax=ax6,
                         cbar_kws={'label': 'Jaccard Similarity'})
    
    ax6.set_title('Inter-tissue Similarity\n(Within and Between Groups)', 
                 fontsize=12, fontweight='bold')
    
    # Add group separators
    group1_end = len(group1)
    ax6.axhline(y=group1_end, color='black', linewidth=2)
    ax6.axvline(x=group1_end, color='black', linewidth=2)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    create_download_button(fig, f"comprehensive_comparison_{group1_name}_vs_{group2_name}.png")
    
    # Detailed gene lists
    st.markdown("### 📋 Detailed Gene Lists")
    
    tab1, tab2, tab3 = st.tabs([
        f"🤝 Shared Genes ({len(shared_genes)})",
        f"🔵 {group1_name} Exclusive ({len(exclusive1)})",
        f"🔴 {group2_name} Exclusive ({len(exclusive2)})"
    ])
    
    with tab1:
        if shared_genes:
            st.markdown(f"**{len(shared_genes)} genes shared between both groups:**")
            shared_list = sorted(list(shared_genes))
            st.text(", ".join(shared_list))
            
            # Download shared genes
            shared_df = pd.DataFrame(shared_list, columns=['Gene'])
            create_csv_download(shared_df, f"shared_genes_{group1_name}_vs_{group2_name}.csv", 
                               "⬇️ Download Shared Genes CSV")
        else:
            st.info("No shared genes found between the groups.")
    
    with tab2:
        if exclusive1:
            st.markdown(f"**{len(exclusive1)} genes exclusive to {group1_name}:**")
            exclusive1_list = sorted(list(exclusive1))
            st.text(", ".join(exclusive1_list))
            
            # Download exclusive genes
            excl1_df = pd.DataFrame(exclusive1_list, columns=['Gene'])
            create_csv_download(excl1_df, f"exclusive_{group1_name}.csv", 
                               f"⬇️ Download {group1_name} Exclusive CSV")
        else:
            st.info(f"No genes exclusive to {group1_name}.")
    
    with tab3:
        if exclusive2:
            st.markdown(f"**{len(exclusive2)} genes exclusive to {group2_name}:**")
            exclusive2_list = sorted(list(exclusive2))
            st.text(", ".join(exclusive2_list))
            
            # Download exclusive genes
            excl2_df = pd.DataFrame(exclusive2_list, columns=['Gene'])
            create_csv_download(excl2_df, f"exclusive_{group2_name}.csv", 
                               f"⬇️ Download {group2_name} Exclusive CSV")
        else:
            st.info(f"No genes exclusive to {group2_name}.")
    
    # Statistical analysis (if requested)
    if "Statistical Tests" in comparison_metrics:
        st.markdown("""
        <div class="analysis-section">
            <h3>📊 Statistical Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Hypergeometric test for enrichment
        from scipy.stats import hypergeom
        
        # Total gene universe
        all_genes_universe = set()
        for tissue in list(data.keys()):
            if analysis_scope == "Full Lifespan":
                all_genes_universe.update(set.union(*data[tissue]))
            else:
                selected_idx = age_groups.index(selected_age)
                all_genes_universe.update(data[tissue][selected_idx])
        
        total_genes = len(all_genes_universe)
        
        # Hypergeometric test
        overlap_observed = len(shared_genes)
        group1_size = len(group1_genes)
        group2_size = len(group2_genes)
        
        # P-value for observing this much overlap by chance
        p_value = 1 - hypergeom.cdf(overlap_observed - 1, total_genes, group1_size, group2_size)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{p_value:.2e}</h3>
                <p>P-value<br>(Hypergeometric)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{significance}</h3>
                <p>Statistical<br>Significance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            expected_overlap = (group1_size * group2_size) / total_genes
            fold_enrichment = overlap_observed / expected_overlap if expected_overlap > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{fold_enrichment:.2f}</h3>
                <p>Fold<br>Enrichment</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretation
        if p_value < 0.05:
            st.success(f"✅ The overlap between {group1_name} and {group2_name} is statistically significant (p < 0.05)")
        else:
            st.info(f"ℹ️ The overlap between {group1_name} and {group2_name} is not statistically significant (p ≥ 0.05)")
    
    # Summary statistics table
    st.markdown("### 📈 Summary Statistics")
    
    summary_data = {
        'Metric': [
            f'{group1_name} Total Genes',
            f'{group2_name} Total Genes',
            'Shared Genes',
            'Union Genes',
            'Jaccard Similarity',
            'Overlap Percentage',
            f'{group1_name} Exclusive',
            f'{group2_name} Exclusive'
        ],
        'Value': [
            len(group1_genes),
            len(group2_genes),
            len(shared_genes),
            len(union_genes),
            f"{jaccard_similarity:.3f}",
            f"{overlap_percentage:.1f}%",
            len(exclusive1),
            len(exclusive2)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    create_csv_download(summary_df, f"summary_statistics_{group1_name}_vs_{group2_name}.csv", 
                       "⬇️ Download Summary Statistics CSV")
