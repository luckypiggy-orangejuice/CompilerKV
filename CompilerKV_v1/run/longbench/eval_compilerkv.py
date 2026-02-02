import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--method', type=str, default='compilerkv')
    args = parser.parse_args()
    
    dataset_list = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
        "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
        "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
    ]
    
    print("="*60)
    print(f"Evaluating method: {args.method}")
    print("="*60)
    
    all_scores = {}
    for dataset in dataset_list:
        eval_file = os.path.join(args.results_dir, dataset, f"{args.method}.json")
        if not os.path.exists(eval_file):
            print(f"{dataset:25s}: [not found]")
            continue
        
        predictions, answers, lengths = [], [], []
        all_classes = None
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    all_classes = data.get("all_classes")
                    if "length" in data:
                        lengths.append(data["length"])
                except:
                    pass
        
        if len(predictions) == 0:
            print(f"{dataset:25s}: [no valid predictions]")
            continue
            
        score = scorer(dataset, predictions, answers, all_classes)
        all_scores[dataset] = score
        print(f"{dataset:25s}: {score:.2f} ({len(predictions)} samples)")
        
        # Save metrics
        output_dir = os.path.dirname(eval_file)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump({dataset: score}, f, ensure_ascii=False, indent=4)
    
    print("="*60)
    if all_scores:
        print(f"Average Score: {np.mean(list(all_scores.values())):.2f}")
    print("="*60)
    
    # Save results CSV
    import csv
    with open(os.path.join(args.results_dir, "results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(["dataset", args.method])
        for ds, sc in all_scores.items():
            writer.writerow([ds, sc])
        writer.writerow(["average", round(np.mean(list(all_scores.values())), 2)])
    print(f"Results saved to {args.results_dir}/results.csv")
