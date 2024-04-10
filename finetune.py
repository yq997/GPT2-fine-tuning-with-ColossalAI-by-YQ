import argparse
from typing import Callable, List, Union
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import warnings

import evaluate
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

# ==============================
# Prepare Hyperparameters
# ==============================
max_length = 512
random_seed = 2024
predict_class = 2

NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

output_transform_fn = lambda x: x
criterion = lambda x: x.loss

def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}

# Sentiment data builder
class SentimentDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length=max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        label = self.data.iloc[idx]['Sentiment']
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length, add_special_tokens=True)
        input_ids = encoding['input_ids'].squeeze()  # Remove the batch dimension
        attention_mask = encoding['attention_mask'].squeeze()  # Remove the batch dimension
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    criterion,
    test_dataloader: Union[DataLoader, List[DataLoader]],
    num_labels: int,
    task_name: str,
    booster: Booster,
    coordinator: DistCoordinator,
):
    metric = evaluate.load("glue", task_name, process_id=coordinator.rank, num_process=coordinator.world_size)
    model.eval()

    def evaluate_subset(dataloader: DataLoader):
        use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
        is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()

        accum_loss = torch.zeros(1, device=get_accelerator().get_current_device())
        for batch in dataloader:
            batch = move_to_cuda(batch)
            labels = batch["labels"]
            if use_pipeline:
                pg_mesh = booster.plugin.pg_mesh
                pp_group = booster.plugin.pp_group
                current_pp_group_ranks = pg_mesh.get_ranks_in_group(pp_group)
                current_rank = dist.get_rank()
                batch = iter([batch])
                outputs = booster.execute_pipeline(batch, model, criterion, return_loss=True, return_outputs=True)

                if is_pp_last_stage:
                    logits = outputs["outputs"]["logits"]
                    val_loss = outputs["loss"]
                    accum_loss.add_(val_loss)

                    if num_labels > 1:
                        preds = torch.argmax(logits, axis=1)
                    elif num_labels == 1:
                        preds = logits.squeeze()

                    dist.broadcast_object_list([preds, val_loss], src=current_pp_group_ranks[-1], group=pp_group)

                    metric.add_batch(predictions=preds, references=labels)
                elif current_rank in current_pp_group_ranks:
                    object_list = [None, None]
                    dist.broadcast_object_list(object_list, src=current_pp_group_ranks[-1], group=pp_group)

                    metric.add_batch(
                        predictions=object_list[0].to(get_accelerator().get_current_device()), references=labels
                    )
                    accum_loss.add_(object_list[1].to(get_accelerator().get_current_device()))

            else:
                batch = move_to_cuda(batch)
                outputs = model(**batch)
                val_loss, logits = outputs[:2]
                accum_loss.add_(val_loss)

                if num_labels > 1:
                    preds = torch.argmax(logits, axis=1)
                elif num_labels == 1:
                    preds = logits.squeeze()

                metric.add_batch(predictions=preds, references=labels)

        results = metric.compute()
        dist.all_reduce(accum_loss.div_(len(dataloader)))
        if coordinator.is_master() and results is not None:
            results["loss"] = accum_loss.item() / coordinator.world_size

        return results

    if isinstance(test_dataloader, DataLoader):
        return evaluate_subset(test_dataloader)

def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    _criterion: Callable,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(
        range(total_step),
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]",
        disable=not (coordinator.is_master() or is_pp_last_stage),
    ) as pbar:
        # Forward pass
        for _ in pbar:
            if use_pipeline:
                outputs = booster.execute_pipeline(
                    train_dataloader_iter, model, _criterion, optimizer, return_loss=True
                )
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})
            else:
                data = next(train_dataloader_iter)
                data = move_to_cuda(data)
                outputs = model(**data)
                loss = _criterion(outputs, None)
                # Backward
                booster.backward(loss, optimizer)
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_f1", type=float, default=None, help="target f1 score. Raise exception if not reached")
    args = parser.parse_args()

    model_name = "gpt2"
    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(config={}, seed=42)
    coordinator = DistCoordinator()

    # local_batch_size = BATCH_SIZE // coordinator.world_size
    lr = LEARNING_RATE * coordinator.world_size

    # ==============================
    # Instantiate Plugin and Booster: only HybridParallelPlugin
    # ==============================
    booster_kwargs = {}
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=2,
        num_microbatches=None,
        microbatch_size=1,
        enable_all_optimization=True,
        zero_stage=1,
        precision="fp16",
        initial_scale=1,
    )

    booster = Booster(plugin=plugin, **booster_kwargs)

    # ==============================
    # Prepare Dataset and Dataloader
    # ==============================
    data = pd.read_csv('data/train.csv')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    train_dataset = SentimentDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = SentimentDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ====================================
    # Prepare model, optimizer
    # ====================================
    # gpt2 pretrained model for binary task

    cfg = AutoConfig.from_pretrained(model_name, num_labels=predict_class)

    model = GPT2ForSequenceClassification.from_pretrained(model_name, config=cfg).cuda()

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    # lr scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(WARMUP_FRACTION * total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, _criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
    )

    # ==============================
    # Train model
    # ==============================
    for epoch in range(NUM_EPOCHS):
        train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)

    results = evaluate_model(
        model,
        _criterion,
        test_dataloader,
        predict_class,
        args.task,
        booster,
        coordinator,
    )

    if coordinator.is_master():
        print(results)
        if args.target_f1 is not None and "f1" in results:
            assert results["f1"] >= args.target_f1, f'f1 score {results["f1"]} is lower than target {args.target_f1}'


if __name__ == "__main__":
    main()
