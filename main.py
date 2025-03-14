import argparse

from train import train_dgl, train_pyg
from utils import data_process


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help="选择使用的模型")
    parser.add_argument("-d", "--data", help="数据处理", action="store_true")

    args = parser.parse_args()

    if args.model == "deepdds":
        train_pyg()
    elif args.model == "enhanced":
        train_dgl()

    if args.data:
        data_process()


if __name__ == "__main__":
    main()