import os
import sys
from argparse import ArgumentParser

import torch
from models.yolo import Model


def main(args):
    out_dir, basename = os.path.split(args.model_path)
    basename = basename.split(".")[0] + ".pth"
    pth_file = os.path.join(out_dir, basename)

    model = Model(args.cfg, nc=args.num_classes)
    ckpt = torch.load(args.model_path, map_location="cpu")  # load checkpoint

    # load model
    ckpt["model"] = {
        k: v
        for k, v in ckpt["model"].float().state_dict().items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }
    model.load_state_dict(ckpt["model"], strict=False)
    print(
        "Transferred %g/%g items from %s"
        % (len(ckpt["model"]), len(model.state_dict()), args.model_path)
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        pth_file,
    )

    print(f"Saved model in pth format to {pth_file}")


def build_argparser():
    parser = ArgumentParser(prog="pt2pth.py")
    parser.add_argument(
        "--model-path",
        type=str,
        default="path/to/model.pt",
        help="model path",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="models/yolov5m.yaml",
        help="model config file",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of classes. We do not want to change this value in the model config file, so we define it here",
    )

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)