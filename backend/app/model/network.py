from __future__ import annotations

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.game.board_encoding import encode_board
from app.game.move_encoding import NUM_MOVES
from app.infra.config import AppConfig, get_current_config


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class ChessNet(nn.Module):
    def __init__(self, cfg: AppConfig | None = None):
        super().__init__()
        self.cfg = cfg or get_current_config()
        in_channels = int(self.cfg.model.input_planes)
        trunk_channels = int(self.cfg.model.channels)
        num_res_blocks = int(self.cfg.model.res_blocks)
        dropout = float(self.cfg.model.value_dropout)

        self.conv_in = nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(trunk_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(trunk_channels) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Conv2d(trunk_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, NUM_MOVES)
        self.value_conv = nn.Conv2d(trunk_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_ln1 = nn.LayerNorm(256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_ln2 = nn.LayerNorm(128)
        self.value_dropout = nn.Dropout(p=dropout)
        self.value_fc3 = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x.float() * 2.0) - 1.0
        x = F.relu(self.bn_in(self.conv_in(x)), inplace=True)
        x = self.res_blocks(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        policy_logits = self.policy_fc(p.reshape(p.size(0), -1))
        v = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        v = v.reshape(v.size(0), -1)
        v = self.value_dropout(F.relu(self.value_ln1(self.value_fc1(v)), inplace=True))
        v = self.value_dropout(F.relu(self.value_ln2(self.value_fc2(v)), inplace=True))
        value = torch.tanh(self.value_fc3(v))
        return policy_logits, value

    @torch.no_grad()
    def predict(self, board: chess.Board, device=None):
        dev = torch.device(device) if device else next(self.parameters()).device
        was_training = self.training
        self.eval()
        x = encode_board(board, self.cfg).unsqueeze(0).to(dev)
        policy_logits, value = self(x)
        if was_training:
            self.train()
        return policy_logits.squeeze(0).cpu().numpy(), float(value.squeeze().item())
