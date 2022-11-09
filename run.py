# Inherited from https://github.com/HazyResearch/KGEmb
"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os
import sys

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers

from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params
from utils.complexhyperbolic import euclidean_update, poincare_grad, poincare_update, full_p_exp_map, p_sum, RiemannianSGD, rgrad, expm

# set paths
KGHOME='/ComplexHyperbolicKGE'
LOG_DIR=os.path.join(KGHOME, 'logs')
DATA_PATH = '/KGEdata'

parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)
parser.add_argument(
    "--gpu", default="0", type=str, help="GPU")
parser.add_argument(
    "--dataset", default="WN18RR", choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10"], help="Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="FFTRotH", choices=all_models, help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=1000, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=1000, type=int, help="Batch size"
)
parser.add_argument(
    "--eval_batch_size", default=500, type=int, help="Batch size"
)
parser.add_argument(
    "--update_steps", default=1, type=int, help="Gradient accumulation steps"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=1e-1, type=float, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["float", "double", "cfloat", "cdouble", "single"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", type=int,
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--save_dir", default='.', help="Where to save the log and model"
)


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # save_dir = get_savedir(args.model, args.dataset)
    save_dir = args.save_dir

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
        #filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    test_relation_examples = []
    for i in range(0, 11):
        test_relation_examples.append(dataset.get_examples("test", rel_idx=i))
    filters = dataset.get_filters()

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create model
    model = getattr(models, args.model)(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    model.cuda()

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)

    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.update_steps,
                            args.neg_sample_size, bool(args.double_neg))
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")
    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, args.eval_batch_size)) 
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                model.cuda()
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()
            test_metrics = avg_both(*model.compute_metrics(test_examples, filters, args.eval_batch_size))
            logging.info(format_metrics(test_metrics, split="test"))

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.cuda()
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, args.eval_batch_size))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, args.eval_batch_size))
    logging.info(format_metrics(test_metrics, split="test"))
    for i in range(0, 11):
        logging.info("\t Results for relation {}".format(i))
        test_metrics = avg_both(*model.compute_metrics(test_relation_examples[i], filters, args.eval_batch_size))
        logging.info(format_metrics(test_metrics, split="test"))


if __name__ == "__main__":
    train(parser.parse_args())

