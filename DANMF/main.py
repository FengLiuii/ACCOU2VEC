"""Fitting a DANMF model."""
import argparse
from DANMF.danmf import DANMF
#from parser import parameter_parser
from utils import read_graph, tab_printer, loss_printer
def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Twitch Brasilians dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node identifiers.
    """
    parser = argparse.ArgumentParser(description="Run DANMF.")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default='C:/Users/LF/PycharmProjects/papercode/cresci-2015.edgelist.csv',
	                    help="Edge list csv.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="C:/Users/LF/PycharmProjects/papercode/canshu-cresci/cresci-2015.menbership1.csv",
	                help="Target embedding csv.")

    parser.add_argument("--membership-path",
                        nargs="?",
                        default="C:/Users/LF/PycharmProjects/papercode/canshu-cresci/cresci-2015.menbership.json",
                        # default="twibot.menbership.json",
	                help="Cluster membership json.")

    parser.add_argument("--iterations",
                        type=int,
                        default=200,
	                help="Number of training iterations. Default is 100.")

    parser.add_argument("--pre-iterations",
                        type=int,
                        default=200,
	                help="Number of layerwsie pre-training iterations. Default is 100.")

    parser.add_argument("--seed",
                        type=int,
                        default=100,
	                help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--lamb",
                        type=float,
                        default=0.01,
	                help="Regularization parameter. Default is 0.01.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 128 64 32.")

    parser.add_argument("--calculate-loss",
                        dest="calculate_loss",
                        action="store_true")

    parser.add_argument("--not-calculate-loss",
                        dest="calculate_loss",
                        action="store_false")

    parser.set_defaults(calculate_loss=True)

    parser.set_defaults(layers=[8, 2])

    return parser.parse_args()
def main():
    """
    Parsing command lines, creating target matrix, fitting DANMF and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    graph = read_graph(args)
    model = DANMF(graph, args)
    model.pre_training()
    model.training()
    if args.calculate_loss:
        loss_printer(model.loss)

if __name__ == "__main__":
    main()
