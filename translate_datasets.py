from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
import translators as ts
import random
import math
import time
import os


SEED = 2025
random.seed(SEED)

SIZE = 10000
SPLITS_RATIO = (0.8, 0.1, 0.1)
LANG = "ru"
LANG_POSTFIX = "-" + LANG
SIZE_POSTFIX = '_10k'
BATCH_SIZE = 8

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
login(hf_token)


DATASETS = {
    'snli': {
        'hf_model': 'stanfordnlp/snli',
        'subsets': [],
        'splits': ['train'],
        'columns': ['premise', 'hypothesis', 'label'],
        'translate_columns': ['premise', 'hypothesis'],
        'size': SIZE,
        'lang': 'en',
    },
    'multi_nli': {
        'hf_model': 'nyu-mll/multi_nli',
        'subsets': [],
        'splits': ['train'],
        'columns': ['premise', 'hypothesis', 'label'],
        'translate_columns': ['premise', 'hypothesis'],
        'size': SIZE,
        'lang': 'en',
    },
    'anli': {
        'hf_model': 'facebook/anli',
        'subsets': [],
        'splits': ['train_r1', 'train_r2', 'train_r3'],
        'columns': ['premise', 'hypothesis', 'label'],
        'translate_columns': ['premise', 'hypothesis'],
        'size': SIZE,
        'lang': 'en',
    },
    'doc_nli': {
        'hf_model': 'saattrupdan/doc-nli',
        'subsets': [],
        'splits': ['train'],
        'columns': ['premise', 'hypothesis', 'label'],
        'translate_columns': ['premise', 'hypothesis'],
        'size': SIZE,
        'lang': 'en',
    },
    'nli_fever': {
        'hf_model': 'pietrolesci/nli_fever',
        'subsets': [],
        'splits': ['train'],
        'columns': ['premise', 'hypothesis', 'label'],
        'translate_columns': ['premise', 'hypothesis'],
        'size': SIZE,
        'lang': 'en',
    },
    'vitaminc': {
        'hf_model': 'tals/vitaminc',
        'subsets': [],
        'splits': ['train'],
        'columns': ['claim', 'evidence', 'label'],
        'translate_columns': ['claim', 'evidence'],
        'size': SIZE,
        'lang': 'en',
    },
    'qqp': {
        'hf_model': 'SetFit/qqp',
        'subsets': [],
        'splits': ['train'],
        'columns': ['text1', 'text2', 'label'],
        'translate_columns': ['text1', 'text2'],
        'size': SIZE,
        'lang': 'en',
    },
    'paws': {
        'hf_model': 'google-research-datasets/paws',
        'subsets': ['labeled_final', 'unlabeled_final', 'labeled_swap'],
        'splits': ['train'],
        'columns': ['sentence1', 'sentence2', 'label'],
        'translate_columns': ['sentence1', 'sentence2'],
        'size': SIZE,
        'lang': 'en',
    },
    'sberquad': {
        'hf_model': 'kuznetsoffandrey/sberquad',
        'subsets': [],
        'splits': ['train'],
        'columns': ['context', 'question', 'answers'],
        'translate_columns': [],
        'size': SIZE,
        'lang': 'ru',
    },
    'race': {
        'hf_model': 'ehovy/race',
        'subsets': ['all'],
        'splits': ['train'],
        'columns': ['article', 'answer', 'question', 'options'],
        'translate_columns': ['article', 'question', 'options'],
        'size': SIZE,
        'lang': 'en',
    },
    'ms_marco': {
        'hf_model': 'microsoft/ms_marco',
        'subsets': ['v2.1'],
        'splits': ['train'],
        'columns': ['answers', 'passages', 'query'],
        'translate_columns': ['answers', 'passages', 'query'],
        'size': SIZE,
        'lang': 'en',
    },
    'sick': {
        'hf_model': 'RobZamp/sick',
        'subsets': [],
        'splits': ['train'],
        'columns': ['sentence_A', 'sentence_B', 'relatedness_score'],
        'translate_columns': ['sentence_A', 'sentence_B'],
        'size': 'full',
        'lang': 'en',
    },
    'ru-stsbenchmark-sts': {
        'hf_model': 'ai-forever/ru-stsbenchmark-sts',
        'subsets': [],
        'splits': ['train', 'validation', 'test'],
        'columns': ['sentence1', 'sentence2', 'score'],
        'translate_columns': [],
        'size': 'full',
        'lang': 'ru',
    },
}


def translate_text_robust(text, src='en', dest='ru', retries=2, delay=1):
    """Attempts to translate text with basic retry logic."""

    # Handle empty strings or non-strings
    if not text or not isinstance(text, str):
        return text
    
    for attempt in range(retries + 1):
        try:
            translated = ts.translate_text(query_text=text, translator="yandex", from_language=src, to_language=dest)
            time.sleep(0.1) # small delay to potentially avoid rate limits
            return translated
        except Exception as e:
            print(f"Warning: Error translating text starting with '{str(text)[:50]}...'. Error: {e}. Attempt {attempt + 1}/{retries + 1}")
            if attempt < retries:
                time.sleep(delay * (attempt + 1)) # exponential backoff
            else:
                print(f"Error: Failed to translate text after {retries + 1} attempts: '{str(text)[:50]}...'")
                return text
            
    print(f"Error: Failed to translate text after {retries + 1} attempts: '{str(text)[:50]}...'")
    return text


class DatasetTranslator:
    def __init__(self,
                 datasets_config,
                 seed,
                 target_lang='ru',
                 lang_postfix='-ru',
                 splits_ratio=(0.8, 0.1, 0.1),
                 size_postfix=10000):
        
        self.config = datasets_config
        self.seed = seed
        self.target_lang = target_lang
        self.lang_postfix = lang_postfix
        self.splits_ratio = splits_ratio
        self.size_postfix = size_postfix
        self.hf_token = os.getenv('HF_TOKEN')

        if not self.hf_token:
            print("Warning: Hugging Face token not found. Set the HF_TOKEN in environment variables.")
        self.hf_api = HfApi()
        self.user_info = self.hf_api.whoami(token=self.hf_token) if self.hf_token else None
        self.username = self.user_info['name'] if self.user_info else None


    def _load_single_dataset(self, name, ds_config):
        """Loads and preprocesses a single dataset based on its configuration."""

        print(f"\n--- Loading dataset: {name} ---")
        hf_model = ds_config['hf_model']
        subsets = ds_config.get('subsets', [])
        splits = ds_config.get('splits', ['train'])
        columns = ds_config.get('columns')
        size = ds_config.get('size')
        concatenated_ds = None

        try:
            # Load data
            all_data_parts = []
            if subsets:
                for subset in subsets:
                    for split in splits:
                        print(f"Loading {hf_model} - Subset: {subset}, Split: {split}")
                        ds_part = load_dataset(hf_model, name=subset, split=split, trust_remote_code=True)
                        all_data_parts.append(ds_part)
            else:
                 for split in splits:
                    print(f"Loading {hf_model} - Split: {split}")
                    try:
                        ds_part = load_dataset(hf_model, split=split, trust_remote_code=True)
                    except Exception as load_err:
                        print(f"Initial load failed for {hf_model}/{split}, trying without specific subset name: {load_err}")
                        ds_part = load_dataset(hf_model, split=split, trust_remote_code=True)
                    all_data_parts.append(ds_part)


            if not all_data_parts:
                print(f"Error: No data loaded for {name}. Skipping.")
                return None

            # Concatenate dataset parts if multiple splits/subset
            if len(all_data_parts) > 1:
                print(f"Concatenating {len(all_data_parts)} parts...")
                concatenated_ds = concatenate_datasets(all_data_parts)
                print(f"Concatenated dataset size: {len(concatenated_ds)}")
            else:
                concatenated_ds = all_data_parts[0]
                print(f"Loaded dataset size: {len(concatenated_ds)}")

            # Select columns
            if columns:
                available_cols = concatenated_ds.column_names
                valid_columns = [col for col in columns if col in available_cols]
                if len(valid_columns) != len(columns):
                    print(f"Warning: Requested columns {columns} but only found {valid_columns} in {name}.")
                if not valid_columns:
                    print(f"Error: No valid columns found for {name} based on config. Available: {available_cols}. Skipping.")
                    return None
                print(f"Selecting columns: {valid_columns}")
                concatenated_ds = concatenated_ds.select_columns(valid_columns)
            else:
                print("Warning: No columns specified in config, keeping all.")


            # Crop dataset to defined size
            if isinstance(size, int) and size < len(concatenated_ds):
                print(f"Sampling {size} examples randomly (seed={self.seed})...")
                concatenated_ds = concatenated_ds.shuffle(seed=self.seed).select(range(size))
            elif size == 'full':
                print("Using full dataset size.")
            else:
                print(f"Using dataset size: {len(concatenated_ds)} (requested size {size} >= actual size or invalid).")

            return concatenated_ds

        except Exception as e:
            print(f"Error loading dataset {name} ({hf_model}): {e}")
            import traceback
            traceback.print_exc()
            return None


    def _translate_batch(self, batch, translate_columns, src_lang='en'):
        """Translates specified columns within a batch."""

        translations = {}
        for col in translate_columns:
            if col not in batch:
                print(f"Warning: Column '{col}' not found in batch for translation. Skipping.")
                continue

            translated_texts = []
            original_texts = batch[col]

            for item in original_texts:
                # Handle different data types (str, list of str, etc)
                if isinstance(item, str):
                    translated = translate_text_robust(item, src=src_lang, dest=self.target_lang)
                    translated_texts.append(translated)
                
                elif isinstance(item, list):
                    # Translate each string in the list
                    translated_list = [translate_text_robust(sub_item, src=src_lang, dest=self.target_lang) if isinstance(sub_item, str) else sub_item for sub_item in item]
                    translated_texts.append(translated_list)

                # Specific handling for ms_marco passages
                elif isinstance(item, dict) and col == 'passages':
                    translated_dict = item.copy()
                    if 'passage_text' in item and isinstance(item['passage_text'], list):
                        translated_dict['passage_text'] = [
                            translate_text_robust(p_text, src=src_lang, dest=self.target_lang) if isinstance(p_text, str) else p_text
                            for p_text in item['passage_text']
                        ]
                    translated_texts.append(translated_dict)
                else:
                    translated_texts.append(item)

            translations[col] = translated_texts

        batch.update(translations)
        return batch


    def translate_dataset(self, dataset: Dataset, ds_config: dict, batch_size=16):
        """Translates the specified columns in the dataset."""

        translate_columns = ds_config.get('translate_columns', [])
        name = next((k for k, v in self.config.items() if v['hf_model'] == ds_config['hf_model']), None)

        if not translate_columns:
            print(f"No columns specified for translation in '{name}'. Skipping translation.")
            return dataset

        if not dataset:
            print(f"Dataset for '{name}' is empty or None. Skipping translation.")
            return None

        print(f"--- Translating dataset: {name} ---")
        print(f"Columns to translate: {translate_columns}")

        # Check if columns exist
        missing_cols = [col for col in translate_columns if col not in dataset.column_names]
        if missing_cols:
            print(f"Warning: Columns {missing_cols} not found in dataset '{name}'. Skipping these.")
            translate_columns = [col for col in translate_columns if col in dataset.column_names]
            if not translate_columns:
                print("No valid columns left to translate. Skipping.")
                return dataset
            
        try:
            translated_dataset = dataset.map(
                self._translate_batch,
                fn_kwargs={'translate_columns': translate_columns, 'src_lang': 'en'},
                batched=True,
                batch_size=batch_size,
                desc=f"Translating {name}"
            )
            print(f"Translation finished for {name}.")
            return translated_dataset
        except Exception as e:
            print(f"Error during translation map operation for {name}: {e}")
            import traceback
            traceback.print_exc()
            return dataset

    def save_to_hub(self, dataset: (Dataset | DatasetDict), name: str, ds_config: dict):
        """Splits dataset if needed and saves to Hugging Face Hub."""

        if not self.hf_token or not self.username:
            print(f"Cannot save {name}: Missing Hugging Face token or username.")
            return

        size_info = ds_config.get('size', 'unknown')
        size_str = f"{self.size_postfix}" if isinstance(size_info, int) else "" if size_info=='full' else ""
        lang_info = ds_config.get('lang', 'unknown')
        lang_str = f"{self.lang_postfix}" if lang_info != "ru" else ""
        repo_name = f"{name}{lang_str}{size_str}"
        hf_repo_id = f"{self.username}/{repo_name}"

        print(f"--- Preparing dataset: {name} for Hugging Face Hub ---")
        print(f"Target repository: {hf_repo_id}")

        if not dataset:
            print(f"Dataset '{name}' is empty or None. Skipping save.")
            return

        ds_to_push = None

        # Splitting
        train_ratio, val_ratio, test_ratio = self.splits_ratio
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            print(f"Warning: Split ratios {self.splits_ratio} do not sum to 1. Normalizing.")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
            print(f"Normalized ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")

        if isinstance(dataset, Dataset):
            print(f"Input is a single Dataset. Splitting into Train/Validation/Test ({train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f})...")
            total_size = len(dataset)
            if total_size < 3:
                print(f"Warning: Dataset '{name}' too small ({total_size} rows) to split. Saving as 'train' split only.")
                ds_to_push = DatasetDict({'train': dataset})
            else:
                try:
                    test_size_abs = int(math.ceil(total_size * test_ratio))
                    if test_size_abs == 0 and test_ratio > 0: test_size_abs = 1
                    if test_size_abs >= total_size:
                        test_size_abs = max(1, total_size - 2) if total_size > 2 else 1

                    train_val_ds = dataset.select(range(total_size - test_size_abs))
                    test_ds = dataset.select(range(total_size - test_size_abs, total_size))

                    train_val_size = len(train_val_ds)
                    val_relative_ratio = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0
                    val_size_abs = int(math.ceil(train_val_size * val_relative_ratio))
                    if val_size_abs == 0 and val_ratio > 0: val_size_abs = 1
                    if val_size_abs >= train_val_size:
                        val_size_abs = max(1, train_val_size - 1) if train_val_size > 1 else 1

                    indices = list(range(train_val_size))
                    random.Random(self.seed).shuffle(indices)

                    val_indices = indices[:val_size_abs]
                    train_indices = indices[val_size_abs:]

                    validation_ds = train_val_ds.select(val_indices)
                    train_ds = train_val_ds.select(train_indices)

                    ds_to_push = DatasetDict({
                        'train': train_ds,
                        'validation': validation_ds,
                        'test': test_ds
                    })
                    print(f"Split complete: Train={len(train_ds)}, Validation={len(validation_ds)}, Test={len(test_ds)}")

                except Exception as split_err:
                    print(f"Error during splitting dataset {name}: {split_err}. Saving as 'train' split only.")
                    ds_to_push = DatasetDict({'train': dataset})

        elif isinstance(dataset, DatasetDict):
            print(f"Input is already a DatasetDict with splits: {list(dataset.keys())}. Checking if standard splits exist.")
            required_splits = {'train', 'validation', 'test'}
            has_standard_splits = required_splits.issubset(dataset.keys())
            if has_standard_splits:
                print("Found standard train/validation/test splits. Pushing as is.")
                ds_to_push = dataset
            elif 'train' in dataset:
                print(f"Warning: DatasetDict for {name} has splits {list(dataset.keys())} but not the standard train/validation/test. Attempting to split the 'train' part.")
                
                train_part = dataset.get('train')
                if train_part and isinstance(train_part, Dataset):
                    total_size = len(train_part)
                    if total_size < 3:
                        ds_to_push = DatasetDict({'train': train_part})
                    else:
                        test_size_abs = int(math.ceil(total_size * test_ratio))
                        if test_size_abs == 0 and test_ratio > 0: test_size_abs = 1
                        if test_size_abs >= total_size: test_size_abs = max(1, total_size - 2) if total_size > 2 else 1
                        train_val_ds = train_part.select(range(total_size - test_size_abs))
                        test_ds = train_part.select(range(total_size - test_size_abs, total_size))
                        train_val_size = len(train_val_ds)
                        val_relative_ratio = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0
                        val_size_abs = int(math.ceil(train_val_size * val_relative_ratio))
                        if val_size_abs == 0 and val_ratio > 0: val_size_abs = 1
                        if val_size_abs >= train_val_size: val_size_abs = max(1, train_val_size - 1) if train_val_size > 1 else 1

                        indices = list(range(train_val_size))
                        random.Random(self.seed).shuffle(indices)
                        val_indices = indices[:val_size_abs]
                        train_indices = indices[val_size_abs:]
                        validation_ds = train_val_ds.select(val_indices)
                        train_ds = train_val_ds.select(train_indices)
                        ds_to_push = DatasetDict({'train': train_ds, 'validation': validation_ds, 'test': test_ds})
                        print(f"Split 'train' part complete: Train={len(train_ds)}, Validation={len(validation_ds)}, Test={len(test_ds)}")
                else:
                    print(f"Cannot split DatasetDict for {name} as 'train' split is missing or invalid. Pushing original structure.")
                    ds_to_push = dataset
            else:
                print(f"DatasetDict for {name} has non-standard splits: {list(dataset.keys())}. Pushing as is.")
                ds_to_push = dataset

        else:
            print(f"Error: Object for {name} is not a Dataset or DatasetDict. Type: {type(dataset)}. Skipping save.")
            return

        # Pushing
        if ds_to_push:
            try:
                print(f"Pushing final dataset structure with splits: {list(ds_to_push.keys())} to {hf_repo_id}")
                ds_to_push.push_to_hub(
                    repo_id=hf_repo_id,
                    token=self.hf_token,
                )
                print(f"Successfully saved {name} to {hf_repo_id}")
            except Exception as e:
                print(f"Error saving dataset {name} to {hf_repo_id}: {e}")
                import traceback
                traceback.print_exc()
        else:
             print(f"Dataset {name} was not prepared for pushing (ds_to_push is None). Skipping push.")

    def process_all(self, translation_batch_size=16):
        """Loads, translates, and saves all datasets defined in the config."""
        print("=== Starting Dataset Translation Pipeline ===")
        if not self.username:
             print("Warning: Hugging Face Username not found. Saving to Hub will likely fail.")

        for name, ds_config in self.config.items():
            # Load dataset
            loaded_ds = self._load_single_dataset(name, ds_config)

            if loaded_ds is None:
                print(f"Skipping processing for {name} due to loading issues.")
                continue

            # Translate dataset
            if ds_config.get('translate_columns'):
                translated_ds = self.translate_dataset(loaded_ds, ds_config, batch_size=translation_batch_size)
                if translated_ds is None:
                    print(f"Skipping saving for {name} due to translation issues.")
                    continue
            else:
                print(f"Dataset {name} does not require translation.")
                translated_ds = loaded_ds

            # Save it
            self.save_to_hub(translated_ds, name, ds_config)
        print("\n=== Dataset Translation Pipeline Finished ===")


if __name__ == "__main__":

    # Define the dataset translator
    translator = DatasetTranslator(
        datasets_config=DATASETS, 
        seed=SEED,
        target_lang=LANG,
        lang_postfix=LANG_POSTFIX,
        splits_ratio=SPLITS_RATIO,
        size_postfix=SIZE_POSTFIX
    )

    # Perform the translation
    translator.process_all(translation_batch_size=BATCH_SIZE)
