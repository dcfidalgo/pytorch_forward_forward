import pytorch_lightning as pl
from torch import nn

import torch
from torch.optim import Adam
from typing import List, Dict


class PositionalEncoding(nn.Module):
    def __init__(self, length: int):
        super().__init__()

        delta = 2 / (length - 1)
        pe = torch.arange(-1, 1 + delta, delta).unsqueeze(-1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, sequence length]
        """
        pe = self.pe.expand(len(x), len(self.pe), 1)
        x = x.unsqueeze(-1)

        return torch.concatenate([x, pe], dim=-1)


class TransformerLayer(nn.Module):
    def __init__(self, nhead: int, dim_feedforward: int = 2048, only_direction: bool = True):
        super().__init__()

        self._only_direction = only_direction
        self._layer = nn.TransformerEncoderLayer(
            d_model=2,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        if self._only_direction:
            x /= x.norm(2, dim=-1, keepdim=True) + 1e-4
        return self._layer(x)


class FFLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        only_direction: bool = True,
    ):
        super().__init__()
        self._linear = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self._relu = nn.ReLU()
        self._only_direction = only_direction

    def forward(self, x: torch.Tensor):
        if self._only_direction:
            x /= x.norm(2, dim=1, keepdim=True) + 1e-4
        return self._relu(self._linear(x))


class MNISTSupervisedModel(pl.LightningModule):
    def __init__(
        self,
        num_layers: int = 2,
        features: int = 2000,
        lr: float = 0.1,
        threshold: float = 2.0,
    ):
        super().__init__()
        self._lr = lr
        self._threshold = threshold

        # first layer keeps length information
        layers = [
            FFLayer(in_features=28 * 28, out_features=features, only_direction=False)
        ]
        for _ in range(num_layers - 1):
            layers.append(FFLayer(in_features=features, out_features=features))
        self._net = nn.Sequential(*layers)

        # activate manual optimization
        self.automatic_optimization = False

    def forward(self, x_pos, x_neg, optimize: bool = False) -> float:
        loss_tot = 0
        for layer, optimizer in zip(self._net, self.optimizers()):
            h_pos = layer(x_pos).pow(2).mean(1)
            h_neg = layer(x_neg).pow(2).mean(1)
            loss = self._compute_loss(h_pos, h_neg)

            if optimize:
                optimizer.optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # detach the tensors to "backprop" only through the last layer
            x_pos, x_neg = layer(x_pos).detach(), layer(x_neg).detach()

            loss_tot += float(loss.detach())

        return loss_tot

    def _compute_loss(self, x_pos, x_neg):
        return torch.log(
            1
            + torch.exp(torch.cat([-x_pos + self._threshold, x_neg - self._threshold]))
        ).mean()

    def training_step(self, batch):
        x_pos, x_neg, target = batch
        loss = self(x_pos, x_neg, optimize=True)

        self.log("train_loss", loss, prog_bar=True)

        with torch.no_grad():
            predicted_target = self._predict_target(x_pos)

        error_rate = 1 - ((target == predicted_target).sum() / len(target))

        self.log(
            "train_error_rate", error_rate, on_step=False, on_epoch=True, prog_bar=True
        )

    def validation_step(self, batch, batch_idx):
        x_pos, x_neg, target = batch
        loss = self(x_pos, x_neg)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        predicted_target = self._predict_target(x_pos)

        return {"target": target, "predicted_target": predicted_target}

    def _predict_target(self, image) -> torch.Tensor:
        goodness = torch.zeros(len(image), 10, device=image.device)
        for i in range(10):
            one_hot_encoded_target = torch.zeros(len(image), 10, device=image.device)
            one_hot_encoded_target[:, i] = 1.0
            image[:, :10] = one_hot_encoded_target

            ys = image
            for layer in self._net:
                ys = layer(ys)
                goodness[:, i] += ys.square().mean(1)

        return torch.argmax(goodness, dim=1)

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        target = torch.concat([output["target"] for output in outputs])
        predicted_target = torch.concat(
            [output["predicted_target"] for output in outputs]
        )

        error_rate = 1 - ((target == predicted_target).sum() / len(target))
        self.log("val_error_rate", error_rate, prog_bar=True)

    def configure_optimizers(self) -> List[Adam]:
        return [Adam(layer.parameters(), self._lr) for layer in self._net]
