import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Optional, Union

def extract_tissue_name(filename: str) -> str:
    """
    Extract clean tissue name from filename - PULIZIA DIRETTA E SEMPLICE
    
    Args:
        filename: Original filename (e.g., "Brain - Hypothalamus_sets_mapped.txt")
        
    Returns:
        Clean tissue name (e.g., "Brain - Hypothalamus")
    """
    # Inizia con il nome del file
    clean_name = filename
    
    # Rimuovi TUTTE le estensioni
    extensions_to_remove = ['.txt', '.csv', '.tsv', '.TXT', '.CSV', '.TSV']
    for ext in extensions_to_remove:
        clean_name = clean_name.replace(ext, '')
    
    # Rimuovi TUTTI i suffissi indesiderati (case insensitive)
    suffixes_to_remove = [
        '_sets_mapped', '_sets_Mapped', '_SETS_MAPPED', '_SETS_mapped',
        '_sets', '_SETS', '_Sets',
        '_mapped', '_MAPPED', '_Mapped', 
        '_data', '_DATA', '_Data',
        '_switching', '_SWITCHING', '_Switching',
        '_stamp', '_STAMP', '_Stamp',
        '_genes', '_GENES', '_Genes',
        '_text', '_TEXT', '_Text',
        '_file', '_FILE', '_File',
        'sets_mapped', 'sets_Mapped', 'SETS_MAPPED', 'SETS_mapped',
        'sets', 'SETS', 'Sets',
        'mapped', 'MAPPED', 'Mapped',
        'text', 'TEXT', 'Text'
    ]
    
    for suffix in suffixes_to_remove:
        clean_name = clean_name.replace(suffix, '')
    
    # Pulisci underscores consecutivi e sostituisci con spazi
    clean_name = clean_name.replace('__', '_').replace('___', '_')
    clean_name = clean_name.replace('_', ' ')
    
    # Rimuovi spazi multipli e trim
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()
    
    # Se il nome è vuoto o troppo corto, usa un fallback
    if not clean_name or len(clean_name) < 2:
        # Estrai solo la parte principale del filename originale
        base_name = filename.split('.')[0]  # Rimuovi estensione
        base_name = base_name.replace('_', ' ')
        return base_name if base_name else "Unknown Tissue"
    
    return clean_name

def parse_stamp_file(uploaded_file, age_groups: List[str]) -> Tuple[List[str], List[int], pd.DataFrame]:
    """
    Parse a STAMP-formatted text file and return structured data
    
    Args:
        uploaded_file: Streamlit uploaded file object or None
        age_groups: List of age group labels
        
    Returns:
        tuple: (used_age_groups, gene_counts_per_age, flattened_dataframe)
    """
    if not uploaded_file:
        return [], [], pd.DataFrame(columns=["Age", "Gene"])

    try:
        # Read file content
        content = uploaded_file.read().decode("utf-8").strip().splitlines()
        
        # Initialize outputs
        counts = []
        used_groups = []
        df_data = []
        
        # Process each line (age group)
        for idx, line in enumerate(content):
            if idx < len(age_groups):
                age_group = age_groups[idx]
                
                # Clean and split genes
                genes = clean_gene_names(line.strip().split())
                
                counts.append(len(genes))
                used_groups.append(age_group)
                
                # Add to dataframe data
                for gene in genes:
                    df_data.append({"Age": age_group, "Gene": gene})
        
        # Fill remaining age groups with zero count if missing
        while len(counts) < len(age_groups):
            counts.append(0)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        return used_groups, counts, df
        
    except Exception as e:
        print(f"Error parsing file {uploaded_file.name}: {e}")
        return [], [], pd.DataFrame(columns=["Age", "Gene"])

def clean_gene_names(gene_list: List[str]) -> List[str]:
    """
    Clean and standardize gene names
    
    Args:
        gene_list: List of raw gene names
        
    Returns:
        List of cleaned gene names
    """
    cleaned_genes = []
    
    for gene in gene_list:
        if gene:  # Skip empty strings
            # Remove common prefixes/suffixes and clean
            cleaned = gene.strip().upper()
            
            # Remove common non-gene characters
            cleaned = re.sub(r'[^\w\-\.]', '', cleaned)
            
            # Skip if too short (likely not a real gene name)
            if len(cleaned) >= 2:
                cleaned_genes.append(cleaned)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_genes = []
    for gene in cleaned_genes:
        if gene not in seen:
            seen.add(gene)
            unique_genes.append(gene)
    
    return unique_genes

def validate_stamp_format(file_content: str, expected_age_groups: int = 5) -> Dict[str, Union[bool, str, List[str]]]:
    """
    Validate STAMP file format and provide feedback
    
    Args:
        file_content: Raw file content string
        expected_age_groups: Expected number of age groups (lines)
        
    Returns:
        Dictionary with validation results
    """
    lines = file_content.strip().splitlines()
    
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'line_count': len(lines),
        'gene_counts': [],
        'total_genes': 0,
        'unique_genes': 0
    }
    
    # Check line count
    if len(lines) != expected_age_groups:
        validation_result['warnings'].append(
            f"Expected {expected_age_groups} lines (age groups), found {len(lines)}"
        )
    
    all_genes = set()
    
    # Validate each line
    for i, line in enumerate(lines):
        line_genes = clean_gene_names(line.strip().split())
        gene_count = len(line_genes)
        
        validation_result['gene_counts'].append(gene_count)
        validation_result['total_genes'] += gene_count
        all_genes.update(line_genes)
        
        # Check for empty lines
        if gene_count == 0:
            validation_result['warnings'].append(f"Line {i+1} is empty (no genes found)")
        
        # Check for very long lines (potential formatting issue)
        if gene_count > 1000:
            validation_result['warnings'].append(
                f"Line {i+1} has {gene_count} genes (unusually high, check formatting)"
            )
    
    validation_result['unique_genes'] = len(all_genes)
    
    # Overall validation
    if validation_result['total_genes'] == 0:
        validation_result['is_valid'] = False
        validation_result['errors'].append("No genes found in file")
    
    return validation_result

def parse_multiple_stamp_files(uploaded_files, age_groups: List[str]) -> Dict[str, Dict]:
    """
    Parse multiple STAMP files and return structured data with CLEAN tissue names
    
    Args:
        uploaded_files: List of Streamlit uploaded file objects
        age_groups: List of age group labels
        
    Returns:
        Dictionary with tissue data and metadata
    """
    parsed_data = {}
    parsing_summary = {
        'total_files': len(uploaded_files) if uploaded_files else 0,
        'successful_parses': 0,
        'failed_parses': 0,
        'total_unique_genes': 0,
        'tissue_names': []
    }
    
    if not uploaded_files:
        return {'data': {}, 'summary': parsing_summary}
    
    all_genes = set()
    
    for file_obj in uploaded_files:
        try:
            # Extract CLEAN tissue name from filename
            original_filename = file_obj.name
            clean_tissue_name = extract_tissue_name(original_filename)
            
            # Ensure unique tissue names if duplicates exist
            final_tissue_name = clean_tissue_name
            counter = 1
            while final_tissue_name in parsed_data:
                final_tissue_name = f"{clean_tissue_name} ({counter})"
                counter += 1
            
            # Parse file
            used_groups, counts, df = parse_stamp_file(file_obj, age_groups)
            
            if not df.empty:
                # Convert to gene sets format
                gene_sets = []
                for age in age_groups:
                    age_genes = set(df[df["Age"] == age]["Gene"].tolist())
                    gene_sets.append(age_genes)
                    all_genes.update(age_genes)
                
                # Store with CLEAN name as key
                parsed_data[final_tissue_name] = {
                    'gene_sets': gene_sets,
                    'dataframe': df,
                    'counts': counts,
                    'total_genes': len(set(df["Gene"])),
                    'original_filename': original_filename,
                    'clean_name': final_tissue_name
                }
                
                parsing_summary['successful_parses'] += 1
                parsing_summary['tissue_names'].append(final_tissue_name)
                
            else:
                parsing_summary['failed_parses'] += 1
                
        except Exception as e:
            print(f"Error parsing {file_obj.name}: {e}")
            parsing_summary['failed_parses'] += 1
    
    parsing_summary['total_unique_genes'] = len(all_genes)
    
    return {'data': parsed_data, 'summary': parsing_summary}

def export_to_stamp_format(data: Dict[str, List[set]], age_groups: List[str]) -> str:
    """
    Export gene data back to STAMP format
    
    Args:
        data: Dictionary with tissue names and gene sets
        age_groups: List of age group labels
        
    Returns:
        STAMP-formatted string
    """
    output_lines = []
    
    for tissue, gene_sets in data.items():
        output_lines.append(f"# {tissue}")
        
        for i, age_group in enumerate(age_groups):
            if i < len(gene_sets):
                genes = sorted(list(gene_sets[i]))
                line = ' '.join(genes) if genes else ''
            else:
                line = ''
            output_lines.append(line)
        
        output_lines.append('')  # Empty line between tissues
    
    return '\n'.join(output_lines)

def create_gene_summary_table(data: Dict[str, List[set]], age_groups: List[str]) -> pd.DataFrame:
    """
    Create a summary table of gene counts across tissues and age groups
    
    Args:
        data: Dictionary with CLEAN tissue names and gene sets
        age_groups: List of age group labels
        
    Returns:
        DataFrame with gene count summary
    """
    summary_data = []
    
    for tissue, gene_sets in data.items():
        row = {'Tissue': tissue}
        
        total_unique = set()
        for i, age_group in enumerate(age_groups):
            if i < len(gene_sets):
                gene_count = len(gene_sets[i])
                total_unique.update(gene_sets[i])
            else:
                gene_count = 0
            
            row[f'{age_group}_Count'] = gene_count
        
        row['Total_Unique'] = len(total_unique)
        row['Average_Per_Age'] = sum(row[f'{age}_Count'] for age in age_groups) / len(age_groups)
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def detect_file_format(file_content: str) -> Dict[str, Union[str, bool, List[str]]]:
    """
    Detect and analyze file format characteristics
    
    Args:
        file_content: Raw file content string
        
    Returns:
        Dictionary with format detection results
    """
    lines = file_content.strip().splitlines()
    
    detection_result = {
        'format_type': 'unknown',
        'is_stamp_compatible': False,
        'line_count': len(lines),
        'delimiter': 'space',
        'has_headers': False,
        'sample_genes': [],
        'confidence': 0.0
    }
    
    if not lines:
        return detection_result
    
    # Analyze first few lines
    sample_lines = lines[:5]
    
    # Check for common delimiters
    delimiters = [' ', '\t', ',', ';']
    delimiter_scores = {}
    
    for delimiter in delimiters:
        scores = []
        for line in sample_lines:
            if line.strip():
                parts = line.split(delimiter)
                # Score based on number of parts and if they look like gene names
                scores.append(len([p for p in parts if p.strip() and is_gene_like(p.strip())]))
        
        delimiter_scores[delimiter] = sum(scores) / len(scores) if scores else 0
    
    # Choose best delimiter
    best_delimiter = max(delimiter_scores.items(), key=lambda x: x[1])
    detection_result['delimiter'] = 'space' if best_delimiter[0] == ' ' else best_delimiter[0]
    
    # Check if STAMP compatible
    non_empty_lines = [line for line in lines if line.strip()]
    if 3 <= len(non_empty_lines) <= 7:  # Reasonable number of age groups
        detection_result['is_stamp_compatible'] = True
        detection_result['format_type'] = 'stamp'
        detection_result['confidence'] = 0.8
    
    # Extract sample genes
    for line in sample_lines[:3]:
        if line.strip():
            parts = line.split(best_delimiter[0])
            gene_like_parts = [p.strip() for p in parts if p.strip() and is_gene_like(p.strip())]
            detection_result['sample_genes'].extend(gene_like_parts[:3])
    
    detection_result['sample_genes'] = detection_result['sample_genes'][:10]  # Limit sample size
    
    return detection_result

def is_gene_like(text: str) -> bool:
    """
    Check if a text string looks like a gene name
    
    Args:
        text: String to check
        
    Returns:
        Boolean indicating if text looks like a gene name
    """
    if not text or len(text) < 2:
        return False
    
    # Basic patterns for gene names
    # Most gene names are 2-20 characters, alphanumeric with some special chars
    if not re.match(r'^[A-Za-z0-9\-\.\_]+$', text):
        return False
    
    # Exclude obvious non-genes
    non_gene_patterns = [
        r'^\d+$',  # Pure numbers
        r'^[a-z]+$',  # All lowercase (less common for genes)
        r'.*\.txt$',  # File extensions
        r'.*\.csv$',
    ]
    
    for pattern in non_gene_patterns:
        if re.match(pattern, text.lower()):
            return False
    
    return True

def merge_duplicate_genes(gene_sets: List[set]) -> List[set]:
    """
    Merge potential duplicate genes with slight name variations
    
    Args:
        gene_sets: List of gene sets to clean
        
    Returns:
        List of cleaned gene sets
    """
    # This is a placeholder for more sophisticated gene name normalization
    # In practice, you might want to use external gene name databases
    
    cleaned_sets = []
    
    for gene_set in gene_sets:
        cleaned_set = set()
        for gene in gene_set:
            # Basic cleaning
            normalized_gene = gene.upper().strip()
            cleaned_set.add(normalized_gene)
        cleaned_sets.append(cleaned_set)
    
    return cleaned_sets

def batch_clean_tissue_names(filenames: List[str]) -> Dict[str, str]:
    """
    Pulisce una lista di nomi file in batch
    
    Args:
        filenames: Lista di nomi file da pulire
        
    Returns:
        Dizionario {filename_originale: nome_pulito}
    """
    return {filename: extract_tissue_name(filename) for filename in filenames}

def create_tissue_summary() -> Dict:
    """
    Crea un summary delle mappature disponibili
    
    Returns:
        Dizionario con statistiche delle mappature
    """
    return {
        "cleaning_method": "Direct string replacement",
        "removes_extensions": [".txt", ".csv", ".tsv"],
        "removes_suffixes": ["_sets_mapped", "_sets", "_mapped", "_data", "_switching", "_stamp", "_genes"],
        "output_format": "Clean tissue names for display"
    }