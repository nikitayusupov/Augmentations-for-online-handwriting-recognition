import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import CharErrorRate
import wandb
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import random

from diplom.data import build_i_am_online_datasets, i_am_online_collate_fn
from diplom.model import IAMOnLineModel
from diplom.postprocessing import IAMOnLineCTCDecoder, IAMOnLineCTCDecoderMultiprocessed


def visualize_random_images(dataset, num_images: int, dataset_part: str) -> None:
    assert dataset_part in ('Test', 'Train', 'All')
    images = []
    cnt = 0
    for idx in random.sample(range(len(dataset)), k=num_images):
        image_path = f"/home/nioljusupov/diploma_nikita/visualization/{idx}.jpg"
        dataset.visualize(idx, image_path)
        image = wandb.Image(image_path, caption=dataset[idx]["text"])
        images.append(image)
        if cnt % 10 == 0:
            print(f'{cnt} images visualized')
        cnt += 1
    wandb.log({f"{dataset_part} visualization": images})


def train_one_epoch(model, train_loader, device, ctc_loss, optimizer, batch_ix):
    model.train()
    for batch in tqdm(train_loader, desc="train_one_epoch"):
        batch_ix += 1

        texts_encoded_padded = batch["texts_encoded_padded"].to(device)
        texts_lengths = batch["texts_lengths"].to(device)
        features_padded = batch["features_padded"].to(device)
        features_lengths = batch["features_lengths"].to(device)

        logprobs = model(features_padded)
        
        loss = ctc_loss(
            logprobs.transpose(0, 1),
            texts_encoded_padded,
            features_lengths,
            texts_lengths,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=9)
        optimizer.step()

        wandb.log({
            "train_loss": float(loss.cpu().detach().item()),
            "batch_ix": batch_ix,
        })
    return batch_ix


@torch.no_grad()
def evaluate(model, decoder, loader, device, epoch_ix, split_name):
    assert split_name in ("val", "test")

    model.eval()
    cer = CharErrorRate()
    for batch in tqdm(loader, desc="evaluate"):
        features_padded = batch["features_padded"].to(device)
        features_lengths = batch["features_lengths"]

        probs = model(features_padded).cpu().exp()
        
        texts_predicted = decoder.decode(probs=probs, lengths=features_lengths)

        cer.update(preds=texts_predicted, target=batch["texts"])

    cer_value = float(cer.compute().cpu().item())

    wandb.log({
        f"{split_name}_cer": cer_value,
        "epoch_ix": epoch_ix,
    })
    return cer_value


@hydra.main(config_path="../configs", config_name="no_lm_no_aug_case_sensetive")
def train(cfg: DictConfig) -> None:
    run = wandb.init(project="diplom_nekita)", config=OmegaConf.to_container(cfg))
    wandb.define_metric("batch_ix")
    wandb.define_metric("train_loss", step_metric="batch_ix")
    wandb.define_metric("epoch_ix")
    wandb.define_metric("val_cer", step_metric="epoch_ix")
    wandb.define_metric("test_cer", step_metric="epoch_ix")

    print('Try to build_i_am_online_datasets')
    datasets = build_i_am_online_datasets(OmegaConf.to_container(cfg.dataset))
    print('Did build datasets')
    string_encoder = datasets["all"].string_encoder

    # print(f'Try to visualize random {cfg.num_images_visualize_before_train} images from train')
    # visualize_random_images(
    #     dataset=datasets["train"],
    #     num_images=cfg.num_images_visualize_before_train,
    #     dataset_part='Train'
    # )
    # print('Successfully visualized train images')

    # print(f'Try to visualize random {cfg.num_images_visualize_before_train} images from test')
    # visualize_random_images(
    #     dataset=datasets["test"],
    #     num_images=cfg.num_images_visualize_before_train,
    #     dataset_part='Test'
    # )
    # print('Successfully visualized test images')

    print(f'Try to visualize random {cfg.num_images_visualize_before_train} images from whole dataset')
    visualize_random_images(
        dataset=datasets["all"],
        num_images=cfg.num_images_visualize_before_train,
        dataset_part='All'
    )
    print('Successfully visualized images')

    for dataset_part_name, v in datasets.items():
        print(f"Size of '{dataset_part_name}' split = {len(v)}")

    train_loader = DataLoader(
        datasets["train"],
        collate_fn=i_am_online_collate_fn,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.dataloader_num_workers,
    )

    val_loader = DataLoader(
        datasets["val"],
        collate_fn=i_am_online_collate_fn,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.dataloader_num_workers,
    )

    test_loader = DataLoader(
        datasets["test"],
        collate_fn=i_am_online_collate_fn,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.dataloader_num_workers,
    )

    device = cfg.device
    model = IAMOnLineModel(
        in_features=5,
        num_classes=1 + len(string_encoder._encoder.classes_),
    )
    model = model.to(device)

    ctc_loss = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # decoder = IAMOnLineCTCDecoder(string_encoder=string_encoder)
    decoder = IAMOnLineCTCDecoderMultiprocessed(
        string_encoder=string_encoder,
        num_processes=cfg.decoder.num_processes
    )

    batch_ix = 0
    min_val_cer_value = float("inf")

    for epoch_ix in range(cfg.train.epochs):
        batch_ix = train_one_epoch(model, train_loader, device, ctc_loss, optimizer, batch_ix)

        torch.save(model.state_dict(), f"{epoch_ix}.pth")
        artifact = wandb.Artifact(f"{epoch_ix}.pth", type="model")
        artifact.add_file(f"{epoch_ix}.pth")
        run.log_artifact(artifact)

        if (epoch_ix + 1) % cfg.val.frequency == 0:
            val_cer_value = evaluate(model, decoder, val_loader, device, epoch_ix, "val")
            
            if val_cer_value < min_val_cer_value:
                evaluate(model, decoder, test_loader, device, epoch_ix, "test")
                min_val_cer_value = val_cer_value

    run.join()


if __name__ == "__main__":
    train()
