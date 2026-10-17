"""Optional real two-process CPU weighted-gradient check; launch with torchrun."""

from types import SimpleNamespace

import torch
from accelerate import Accelerator

from lerobot.scripts.lerobot_train import update_policy


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch, reduction="mean"):
        loss = self.value * batch["values"]
        return loss if reduction == "none" else loss.mean(), {}


def main():
    accelerator = Accelerator(cpu=True)
    if accelerator.num_processes != 2:
        raise ValueError("Launch this check with exactly two processes")
    model = TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    if accelerator.process_index == 0:
        values = torch.tensor([1.0, 2.0])
        weights = torch.tensor([1.0, 0.0])
    else:
        values = torch.tensor([3.0, 4.0, 5.0])
        weights = torch.tensor([0.0, 1.0, 1.0])
    for reduction, expected in (("mean", 0.8), ("self_normalized", 1.0 - 1.0 / 3)):
        with torch.no_grad():
            accelerator.unwrap_model(model).value.fill_(1.0)
        provider = SimpleNamespace(reduction=reduction, compute_batch_weights=lambda batch: (weights, {}))
        update_policy(
            SimpleNamespace(),
            model,
            {"values": values},
            optimizer,
            0.0,
            accelerator,
            grbc_weights_provider=provider,
        )
        actual = accelerator.unwrap_model(model).value.item()
        if abs(actual - expected) > 1e-6:
            raise AssertionError((reduction, actual, expected))
    before = accelerator.unwrap_model(model).value.detach().clone()
    provider = SimpleNamespace(
        reduction="mean", compute_batch_weights=lambda batch: (torch.zeros_like(weights), {})
    )
    _, logs = update_policy(
        SimpleNamespace(),
        model,
        {"values": values},
        optimizer,
        0.0,
        accelerator,
        grbc_weights_provider=provider,
    )
    torch.testing.assert_close(accelerator.unwrap_model(model).value, before)
    assert logs["grbc_skipped_all_zero"] == 1
    if accelerator.is_main_process:
        print("Two-process CPU DDP: global mean, self-normalized, unequal local batches, all-zero skip: PASS")
    accelerator.end_training()


if __name__ == "__main__":
    main()
