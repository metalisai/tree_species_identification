import argparse
from rfdetr import RFDETRSegLarge
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

#import warnings
#warnings.filterwarnings("error", category=RuntimeWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Train RF-DETR Segmentation Model")

    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to training checkpoint to start from")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--cp_interval", type=int, default=20, help="Checkpoint interval in epochs")
    parser.add_argument("--aug", action="store_true", help="enable augmentations")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoint is None:
        model = RFDETRSegLarge()
    else:
        model = RFDETRSegLarge(pretrain_weights=args.checkpoint)

    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=1e-4,
        lr_encoder=1.5e-4,
        output_dir=args.output_dir,
        device="cuda",
        tensorboard=True,
        run_test=False,
        checkpoint_interval=args.cp_interval,
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        eval_max_dets=500,
        aug_config={} if not args.aug else AUG_AERIAL
    )


if __name__ == "__main__":
    main()
