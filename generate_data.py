from tqdm.autonotebook import tqdm
import os
import json
import random
import re
import pandas as pd
import torch
import transformers
from datasets import load_dataset
import logging
from logging import error, info, debug, warning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from random import sample


DATASET_HUGGINGFACE = {
    'xnli': ['MilyaShams/xnli_ru_en_10k', 'ru'],
    'ru_sts': ['ai-forever/ru-stsbenchmark-sts', 'train'],
    'rufact':['akozlova/RuFacts', 'train'],
}

DATASET_CONFIG = {
    'xnli': {'task': 'nli', 'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'huggingface': True},
    'ru_sts': {'task': 'sts', 'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'score', 'huggingface': True},
    'rufact': {'task': 'paraphrase', 'text_a': 'evidence', 'text_b': 'claim', 'label': 'label', 'huggingface':True},
}

class QA2D():
    def __init__(self, batch_size=32, device='cuda', verbose=True) -> None:
        from transformers import BartTokenizer, BartForConditionalGeneration
        self.tokenizer = BartTokenizer.from_pretrained("MarkS/bart-base-qa2d")
        self.model = BartForConditionalGeneration.from_pretrained("MarkS/bart-base-qa2d").to(device)
        self.batch_size = batch_size
        self.device=device
        self.verbose = verbose

    def generate(self, questions: list, answers: list):
        assert len(questions) == len(answers)
        qa_list = []
        for q, a in zip(questions, answers):
            qa_list.append(f"question: {q} answer: {a}")
        output = []
        for qa_pairs in tqdm(
            self.chunks(qa_list, self.batch_size),
            desc="QA to Declarative",
            total=int(len(qa_list)/self.batch_size),
            disable=(not self.verbose)
        ):
            input_text = qa_pairs
            input_token = self.tokenizer(
                input_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            dec_sents = self.model.generate(
                input_token.input_ids, max_length=512)
            result = self.tokenizer.batch_decode(
                dec_sents, skip_special_tokens=True)
            output.extend(result)

        return output

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]



class QAnswering():
    """
    To answer not-answerable questions
    """

    def __init__(self, batch_size=32, device='cuda') -> None:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        self.tokenizer = T5Tokenizer.from_pretrained(
            "valhalla/t5-base-qa-qg-hl")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "valhalla/t5-base-qa-qg-hl").to(device)
        self.batch_size = batch_size
        self.device = device

    def generate(self, questions: list, contexts: list):
        assert len(questions) == len(contexts)
        answers = []
        for qs, cs in tqdm(zip(self.chunks(questions, self.batch_size), self.chunks(contexts, self.batch_size)), desc="Generating Answers for not answerable", total=int(len(questions)/self.batch_size)):
            qc_pairs = []
            assert len(qs) == len(cs)
            for one_q, one_c in zip(qs, cs):
                qc_pairs.append(f"""question: {one_q} context: {one_c}""")
            input_ids = self.tokenizer(
                qc_pairs, padding=True, truncation=True, return_tensors='pt').to(self.device).input_ids
            outputs = self.model.generate(input_ids, max_length=512)
            answers.extend(self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True))

        return answers

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]



class MLMGeneratorWithPairedData():
    def __init__(self, corpra: list, device='cuda', batch_size=8, mask_percent=0.25) -> None:
        self.device = device
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.model = transformers.DistilBertForMaskedLM.from_pretrained(
            "distilbert-base-uncased").to(self.device)
        self.mask_percent = mask_percent
        self.batch_size = batch_size

        self.dataset = corpra  # text needs to be noised

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def generate(self):
        sents_output = []
        for examples in tqdm(self.chunks(self.dataset, self.batch_size), total=int(len(self.dataset)/self.batch_size), desc="MLM Generating"):
            sents_to_be_noised = [each for each in examples]
            sents_noised = self.mlm_infiller(sents_to_be_noised)

            sents_output.extend(sents_noised)

        return sents_output

    def mlm_infiller(self, batch):
        """
        input a batch of sentences, list
        """
        masked_batch = []
        masked_batch_ids = []
        for each_sent in batch:
            sent_tokens = self.tokenizer.tokenize(each_sent)
            sent_token_ids = self.tokenizer(each_sent)['input_ids']
            mask_list = sample(list(range(len(sent_tokens))), int(
                self.mask_percent * len(sent_tokens)))
            sent_tokens = [
                each if i not in mask_list else self.tokenizer.mask_token for i, each in enumerate(sent_tokens)]
            masked_batch_ids.append(
                [each if i-1 not in mask_list else self.tokenizer.mask_token_id for i, each in enumerate(sent_token_ids)])
            masked_batch.append(' '.join(sent_tokens))

        inputs = self.tokenizer(
            masked_batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits

        infill_tokens = []
        for i in range(len(masked_batch)):
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[
                i].nonzero(as_tuple=True)[0]

            predicted_token_id = logits[i, mask_token_index].argmax(axis=-1)
            infill_tokens.append(predicted_token_id)

        infilled_sent = []
        for masked_sent_ids, infill_token in zip(masked_batch_ids, infill_tokens):
            for infill_one_token in infill_token:
                for i, each_id in enumerate(masked_sent_ids):
                    if each_id == self.tokenizer.mask_token_id:
                        masked_sent_ids[i] = infill_one_token
                        break
            infilled_sent.append(self.tokenizer.decode(
                masked_sent_ids, skip_special_tokens=True))

        return infilled_sent


 
class ExtractiveSummarizationGenerator():
    def __init__(self) -> None:
        pass

    def generate(self, texts):
        '''
        texts: list of string
        '''
        from summa.summarizer import summarize

        summaries = []
        for text in tqdm(texts, desc="Extracting Summary"):
            for prop in range(1, 20):
                summ = summarize(text, ratio=prop/20.)
                if len(summ) > 0:
                    break
            summaries.append(summ)

        return summaries
    


class DataGenerator():
    def __init__(self, dataset_names) -> None:
        self.dataset_names = dataset_names
        self.datasets = dict()
        self.t5_qa = None
        self.t5_tokenizer = None

        self.load_dataset_from_huggingface()

    def load_dataset_from_huggingface(self):
        for each_dataset in self.dataset_names:
            if DATASET_CONFIG[each_dataset].get('huggingface'):
                self.datasets[each_dataset] = load_dataset(
                    *DATASET_HUGGINGFACE[each_dataset][:-1],trust_remote_code=True)[DATASET_HUGGINGFACE[each_dataset][-1]]
            elif DATASET_CONFIG[each_dataset].get('using_hf_api'):
                self.datasets[each_dataset] = load_dataset(
                    *DATASET_HUGGINGFACE[each_dataset][:-1], data_dir=DATASET_CONFIG[each_dataset]['data_dir'],trust_remote_code=True)[DATASET_HUGGINGFACE[each_dataset][-1]]
            elif DATASET_CONFIG[each_dataset].get('using_pandas'):
                if DATASET_CONFIG[each_dataset]['data_path'].split('.')[-1] == 'tsv':
                    self.datasets[each_dataset] = pd.read_csv(
                        DATASET_CONFIG[each_dataset]['data_path'], sep='\t')
                elif DATASET_CONFIG[each_dataset]['data_path'].split('.')[-1] == 'csv':
                    self.datasets[each_dataset] = pd.read_csv(
                        DATASET_CONFIG[each_dataset]['data_path'])
            elif DATASET_CONFIG[each_dataset].get('using_json'):
                self.datasets[each_dataset] = []
                if DATASET_CONFIG[each_dataset].get('raw_json'):
                    with open(DATASET_CONFIG[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                        self.datasets[each_dataset] = json.load(f)
                else:
                    try:
                        json_file = json.load(
                            open(DATASET_CONFIG[each_dataset]['data_path'], 'r', encoding='utf8'))
                        for example in json_file:
                            self.datasets[each_dataset].append(example)
                    except:
                        with open(DATASET_CONFIG[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                            for example in f:
                                self.datasets[each_dataset].append(
                                    json.loads(example))
            else:
                error('unable to locate raw dataset...')

    def process_stsb(self):
        output = []
        for example in tqdm(self.datasets['stsb'], desc=f'Constructing stsb'):
            text_a = example[DATASET_CONFIG['stsb']['text_a']]
            text_b = [example[DATASET_CONFIG['stsb']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['stsb']['label']] / 5.0

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def init_qa_t5(self):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        if self.t5_qa is None:
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                "t5-base", model_max_length=800)
            self.t5_qa = T5ForConditionalGeneration.from_pretrained("t5-base")
            self.t5_qa.to('cuda:1')
            self.t5_qa.eval()

    @staticmethod
    def mask_answer(context, answers):
        answers = sorted(answers, key=len, reverse=True)
        for answer in answers:
            pattern = f'(?<![\w\\-\u2013]){re.escape(answer)}(?![\w\\-\u2013])'
            context = re.sub(pattern, '', context, flags=re.IGNORECASE)
        return context

    def generate_fake_answer(self, context, question, answers):
        self.init_qa_t5()

        context_no_answer = self.mask_answer(context, answers)

        input_ids = self.t5_tokenizer(
            f'question: {question} context: {context_no_answer}',
            return_tensors="pt",
            truncation='only_first'
        ).input_ids.to(self.t5_qa.device)

        outputs = self.t5_qa.generate(
            input_ids,
            max_new_tokens=40,
            remove_invalid_values=True
        )

        return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def negative_sample_qa(self, samples, negative_sample_no_ans_only=True):
        outputs = []
        for context, question, answers in samples:
            if answers:
                outputs.append({
                    'text_a': context,
                    'text_b': [question],
                    'text_c': answers,
                    'label': 1
                })
            if not answers or not negative_sample_no_ans_only:
                fake_answer = self.generate_fake_answer(
                    context, question, answers)
                outputs.append({
                    'text_a': context,
                    'text_b': [question],
                    'text_c': [fake_answer],
                    'label': 0
                })

        return outputs

    def process_xnli(self):
        output = []
        for example in tqdm(self.datasets['xnli'], desc=f'Constructing xnli'):
            text_a = example[DATASET_CONFIG['xnli']['text_a']]
            text_b = [example[DATASET_CONFIG['xnli']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['xnli']['label']]
            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_rufact(self):
        output = []
        for example in tqdm(self.datasets['rufact'], desc=f'Constructing rufact'):
            text_a = example[DATASET_CONFIG['rufact']['text_a']]
            text_b = [example[DATASET_CONFIG['rufact']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['rufact']['label']]
            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_ru_sts(self):
        output = []
        for example in tqdm(self.datasets['ru_sts'], desc=f'Constructing ru_sts'):
            text_a = example[DATASET_CONFIG['ru_sts']['text_a']]
            text_b = [example[DATASET_CONFIG['ru_sts']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['ru_sts']['label']]
            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })
        return output

    def generate(self):
        if not os.path.exists('./data/training'):
            os.makedirs('./data/training')
        for each_dataset in self.datasets:
            with open(f'./data/training/{each_dataset}.json', 'w', encoding='utf8') as outfile:
                outfile.write("")
        for each_dataset in self.datasets:
            outputs = eval(f'self.process_{each_dataset}()')

            for each_output in outputs:
                dict_write_to_file = {
                    'task': DATASET_CONFIG[each_dataset]['task'],
                    'text_a': each_output['text_a'],  # string
                    # list of positive examples
                    'text_b': each_output['text_b'],
                    # list of negative examples
                    'text_c': each_output['text_c'],
                    # original label, if -1 only has positive pairs and negative pairs
                    'orig_label': each_output['label']
                }
                with open(f'./data/training/{each_dataset}.json', 'a', encoding='utf8') as outfile:
                    json.dump(dict_write_to_file, outfile, ensure_ascii=False)
                    outfile.write('\n')



if __name__ == "__main__":
    random.seed(42)
    gen = DataGenerator(list(DATASET_CONFIG.keys()))
    gen.generate()