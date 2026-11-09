"""Coordinate sensitivity audit for any task expert (reuses the DTD audit logic)."""
import argparse
from pathlib import Path

import torch

from audit_dtd_expert_controls import evaluate_expert
from run_dtd_carrier_diagnostics import Operator
from task_data import TASKS, test_protocol, test_set, train_set
from train_eight_expert_queue import BATCH_SIZE, HEADS, ROOT, build_model, existing, loader, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--folder", type=Path, required=True, help="arm folder holding epoch08.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    args = p.parse_args()

    existing.ROOT = ROOT
    checkpoint = read(args.folder / "complete.json")
    historical = read(args.folder / "historical_test.json")
    provenance = checkpoint["provenance"]
    head = HEADS / f"{args.task}.pt"
    if (provenance["task"] != args.task or provenance["head_sha256"] != existing.sha(head)
            or provenance["base_sha256"] != existing.sha(args.pretrained)
            or historical["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]):
        raise ValueError(f"{args.task}: audit inputs differ from the trained expert")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    encoder, model, _, flat = build_model(
        existing, existing.ClassificationHead.load(str(head)).cuda())
    del flat
    model.requires_grad_(False).eval()
    model.image_encoder.model.visual.requires_grad_(True)
    op = Operator(model, encoder.model.visual)

    train = train_set(args.task, encoder.val_preprocess)
    test = test_set(args.task, encoder.val_preprocess)
    if len(train) != provenance["train_n"] or test_protocol(args.task, test) != historical["provenance"]:
        raise ValueError(f"{args.task}: audit data differs from training and scoring")
    ids = torch.randperm(len(train), generator=torch.Generator().manual_seed(35036))[:64].tolist()
    probes = []
    for group in (ids[:32], ids[32:]):
        samples = [train[i] for i in group]
        probes.append((torch.stack([x for x, _ in samples]).cuda(),
                       torch.tensor([y for _, y in samples], device="cuda")))
    batches = [(x.cuda(), y.cuda()) for x, y in loader(test, 8, BATCH_SIZE, 42)]
    raw = torch.load(args.pretrained, map_location="cpu", mmap=True, weights_only=True)
    pretrained = raw if isinstance(raw.get("visual"), dict) else {
        "visual": {k.split("visual.", 1)[1]: v for k, v in raw.items() if k.startswith("visual.")}}

    result = evaluate_expert(args.folder, op, probes, batches, pretrained)
    args.output.mkdir(parents=True, exist_ok=True)
    existing.write(args.output / f"{args.task}_controls.json",
                   dict(test_provenance=historical["provenance"], probe_indices=[ids[:32], ids[32:]],
                        radii=[0.005, 0.02], arm=result))
    print(f"{args.task}: controls written, baseline acc {result['baseline_test']['accuracy']:.4f}, "
          f"components {result['components']}", flush=True)


if __name__ == "__main__":
    main()
