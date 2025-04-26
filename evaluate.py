from typing import Dict, Tuple, List
from src.inference import Inferencer
from src.AlignScore import AlignScore
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.metrics import matthews_corrcoef
import torch
from datasets import load_dataset
from pathlib import Path
import json


DATASET = {
    'xnli': {'source':'huggingface', 'dataset':'MilyaShams/xnli_ru_en_10k', 'split':'ru','text_a':'premise','text_b':'hypothesis','label':'label','task':'nli'},
    'ru_sts': {'source':'huggingface', 'dataset':'ai-forever/ru-stsbenchmark-sts', 'split':'test','text_a':'sentence1','text_b':'sentence2','label':'score','task':'regression'},
    'rufact':{'source':'huggingface', 'dataset':'akozlova/RuFacts', 'split':'validation','text_a':'evidence','text_b':'claim','label':'label','task':'binary'},
}

def evaluate_binary(inferencer: Inferencer, dataset,threshold=0.5):
    score = inferencer.inference(premise=dataset['text_a'], hypo=dataset['text_b'])[1]
    score = (score > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(dataset['labels'], score, average='binary')
    roc_auc = roc_auc_score(dataset['labels'], score)
    mcc = matthews_corrcoef(dataset['labels'], score)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'matthews_corrcoef': mcc,
    }

def evaluate_nli(inferencer: Inferencer, dataset):
    score = inferencer.inference(premise=dataset['text_a'], hypo=dataset['text_b'])[2][:,0]
    precision, recall, f1, _ = precision_recall_fscore_support(dataset['labels'], score, average='micro')
    accuracy = accuracy_score(dataset['labels'], score)
    mcc = matthews_corrcoef(dataset['labels'], score)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'matthews_corrcoef': mcc,
    }

def evaluate_regression(inferencer: Inferencer, dataset):
    score = inferencer.inference(premise=dataset['text_a'], hypo=dataset['text_b'])[0]
    mse = mean_squared_error(dataset['labels'], score)
    r2 = r2_score(dataset['labels'], score)
    return {
        'mse': mse,
        'r2': r2,
    }

TASK_MAP = {
        'binary': evaluate_binary,
        'nli': evaluate_nli,
        'regression': evaluate_regression,
    }

class EvaluationDatasetLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_config = DATASET[dataset_name]
        self.dataset = self.load_dataset()

    def load_dataset(self):
        if self.dataset_config['source'] == 'huggingface':
            dataset = load_dataset(self.dataset_config['dataset'], self.dataset_config['split'])
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not in the list of supported datasources. May be you need to implement it or change the datasource.")
        return self.process_dataset(dataset)
    
    def process_dataset(self, dataset):
        output = []
        for example in dataset:
            output.append({
                'text_a': example[self.dataset_config['text_a']],
                'text_b': [example[self.dataset_config['text_b']]],
                'text_c': [],
                'labels': example[self.dataset_config['label']],
            })
        return output
    
    def save_dataset(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.dataset, f)

    def load_from_file(self, path: Path):
        with open(path, 'r') as f:
            self.dataset = json.load(f)

    def evaluate(self, model: Inferencer):
        return TASK_MAP[self.dataset_config['task']](model, self.dataset)



if __name__ == '__main__':
    args = {}
    args['model_path'] = 'checkpoints/rufact_roberta_large_v2.pth'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['batch_size'] = 16
    args['max_length'] = 512
    args['threshold'] = 0.5

    model = Inferencer(model_path=args['model_path'],
                              device=args['device'],
                              batch_size=args['batch_size'],
                              max_length=args['max_length'],
                              evaluation_mode='bin_sp',
                              verbose=False)
    
    dataset_loader = EvaluationDatasetLoader(dataset_name='rufact')
    dataset_loader.load_from_file(Path('datasets/rufact.json'))
    results = dataset_loader.evaluate(model)
    print(results)
