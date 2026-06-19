"""
FiRA Judgement Aggregation Starter Code
Dosen: Zico Pratama Putra
Kelompok: Rizqia - Tungkot - Intan
"""

import pandas as pd
import numpy as np

def load_raw_judgements(file_path: str) -> pd.DataFrame:
    """Load raw FiRA judgements"""
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded {len(df)} raw judgements")
    return df

def simple_majority_vote(group):
    """Baseline: Simple majority vote"""
    votes = group['judgement'].value_counts()
    return votes.idxmax()

def calculate_annotator_reliability(df: pd.DataFrame):
    """Menghitung reputasi annotator berdasarkan deviasi dari nilai rata-rata kelompok"""
    mean_scores = df.groupby(['query_id', 'doc_id'])['judgement'].transform('mean')
    df['deviation'] = np.abs(df['judgement'] - mean_scores)
    annotator_dev = df.groupby('annotator_id')['deviation'].mean()
    weights = 1.0 / (annotator_dev + 0.1)  # epsilon agar tidak zero-division
    return weights

def advanced_aggregation(group, annotator_weights):
    """
    [LENGKAP] Strategi agregasi tingkat lanjut menggunakan 
    pembobotan confidence & reliabilitas annotator (Weighted Voting)
    """
    scores = group['judgement'].astype(float).values
    confidences = group['confidence'].astype(float).values
    ann_ids = group['annotator_id'].values
    
    # Gabungkan bobot internal (confidence) dengan reputasi penilai
    g_weights = confidences * [annotator_weights.get(uid, 1.0) for uid in ann_ids]
    
    if g_weights.sum() > 0:
        final_score = np.average(scores, weights=g_weights)
    else:
        final_score = np.median(scores)
    return int(round(final_score))

def aggregate_judgements(df: pd.DataFrame, method='advanced') -> pd.DataFrame:
    """Main aggregation function"""
    annotator_weights = calculate_annotator_reliability(df)
    grouped = df.groupby(['query_id', 'doc_id'])
    
    aggregated = []
    for (qid, did), group in grouped:
        if method == 'majority':
            score = simple_majority_vote(group)
        else:
            score = advanced_aggregation(group, annotator_weights)
            
        aggregated.append({
            'query_id': int(qid),
            'doc_id': str(did),
            'score': int(score),
            'num_judgements': len(group),
            'std_score': float(np.std(group['judgement']))
        })
    
    result_df = pd.DataFrame(aggregated)
    print(f"Aggregated into {len(result_df)} unique query-doc pairs")
    return result_df

def save_qrels(aggregated_df: pd.DataFrame, output_path: str):
    """Save in TREC qrels format"""
    with open(output_path, 'w') as f:
        for _, row in aggregated_df.iterrows():
            f.write(f"{row['query_id']} 0 {row['doc_id']} {row['score']}\n")
    print(f"Qrels saved to {output_path}")

# ====================== MAIN ======================
if __name__ == "__main__":
    df = load_raw_judgements("data/fira_raw_judgements.tsv")
    agg_df = aggregate_judgements(df, method='advanced')
    save_qrels(agg_df, "data/fira_aggregated.qrels")
    
    print("\n=== Aggregation Statistics ===")
    print(agg_df['score'].value_counts().sort_index())