from pytorch_lightning import Trainer, seed_everything
from src.dataloader import DSTDataLoader
from src.model import BERTAlignModel
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from tqdm import tqdm
import os
import torch
from pathlib import Path

ALL_TRAINING_DATASETS = {
        ### NLI
        'snli': {'task_type': 'nli', 'data_path': 'snli.json'},
        'anli': {'task_type': 'nli', 'data_path': 'anli.json'},
        'doc_nli': {'task_type': 'bin_nli', 'data_path': 'doc_nli.json'},
        'multi_nli': {'task_type': 'nli', 'data_path': 'multi_nli.json'},

        ### fact checking
        'nli_fever': {'task_type': 'fact_checking', 'data_path': 'nli_fever.json'},
        'vitaminc' : {'task_type': 'fact_checking', 'data_path': 'vitaminc.json'},

        ### QA
        'race': {'task_type': 'qa', 'data_path': 'race.json'},
        'ms_marco': {'task_type': 'qa', 'data_path': 'ms_marco.json'},


        ### paraphrase
        'rufact': {'task_type': 'paraphrase', 'data_path': 'rufact.json'},
        'qqp': {'task_type': 'paraphrase', 'data_path': 'qqp.json'},
        'paws': {'task_type': 'paraphrase', 'data_path': 'paws.json'},

        ### STS
        'ru_sts': {'task_type': 'sts', 'data_path': 'ru_sts.json'},
        'sick': {'task_type': 'sts', 'data_path': 'sick.json'},
    }


def train(datasets, args):
    dm = DSTDataLoader(
        dataset_config=datasets,
        model_name=args['model_name'],
        sample_mode='seq',
        train_batch_size=args['batch_size'],
        eval_batch_size=16,
        num_workers=args['num_workers'],
        train_eval_split=0.95,
        need_mlm=args['do_mlm']
    )
    dm.setup()

    model = BERTAlignModel(model=args['model_name'],
        using_pretrained=args['use_pretrained_model'],
        adam_epsilon=args['adam_epsilon'],
        learning_rate=args['learning_rate'],
        weight_decay=args['weight_decay'],
        warmup_steps_portion=args['warm_up_proportion']
    )
    model.need_mlm = args['do_mlm']

    training_dataset_used = '_'.join(datasets.keys())
    n_devices = args["devices"] if isinstance(args["devices"], int) else len(args["devices"])
    batch_string = f"{args['batch_size']}x{n_devices}x{args['accumulate_grad_batch']}"
    checkpoint_name = '_'.join((
        f"{args['ckpt_comment']}{args['model_name'].replace('/', '-')}",
        f"{'scratch_' if not args['use_pretrained_model'] else ''}{'no_mlm_' if not args['do_mlm'] else ''}{training_dataset_used}",
        str(args['max_samples_per_dataset']),
        batch_string
    ))

    checkpoint_callback = ModelCheckpoint(
        dirpath=args['ckpt_save_path'],
        filename=checkpoint_name + "_{epoch:02d}_{step}",
        every_n_train_steps=10000,
        save_top_k=1
    )
    trainer = Trainer(
        accelerator=args['accelerator'],
        devices=args['devices'],
        strategy=args['strategy'],
        max_epochs=args['num_epoch'],
        precision=16 if args['accelerator'] == 'gpu' else 16,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args['accumulate_grad_batch']
    )

    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(os.path.join(args['ckpt_save_path'], f"{checkpoint_name}_final.ckpt"))

    print("Training is finished.")


def main():
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')
    args = {}
    args['seed']=2025
    args['batch_size']=12
    args['accumulate_grad_batch']=1
    args['num_epoch']=3
    args['num_workers']=8
    args['warm_up_proportion']=0.06
    args['adam_epsilon']=1e-6
    args['weight_decay']=0.1
    args['learning_rate']=1e-5
    args['val_check_interval']=1. /4
    args['accelerator']='gpu' if torch.cuda.is_available() else 'cpu'
    args['devices']=[0] if torch.cuda.is_available() else 1
    args['strategy']='auto'
    args['model_name']='DeepPavlov/rubert-base-cased'
    args['ckpt_save_path']='ckpt'
    args['ckpt_comment']=""
    args['trainin_datasets']=list(ALL_TRAINING_DATASETS.keys())
    args['data_path']='data/training'
    args['max_samples_per_dataset']=500000
    args['do_mlm']=False
    args['use_pretrained_model']=True

    seed_everything(args['seed'])

    datasets = {
        name: {
            **ALL_TRAINING_DATASETS[name],
            "size": args['max_samples_per_dataset'],
            "data_path": Path(args['data_path']) / ALL_TRAINING_DATASETS[name]['data_path']
        }
        for name in tqdm(args['trainin_datasets'], desc='Loading datasets')
    }
    
    train(datasets, args)

if __name__ == "__main__":
    main()
    