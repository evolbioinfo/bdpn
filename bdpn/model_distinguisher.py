import logging
from functools import reduce
from multiprocessing.pool import ThreadPool

import numpy as np
from wquantiles import quantile

from bdpn.tree_manager import TIME, read_forest, annotate_forest_with_time

DEFAULT_CHERRY_BLOCK_SIZE = 100

DEFAULT_NEIGHBOURHOOD_SIZE = 5

RANDOM_REPETITIONS = 1e3

DEFAULT_PERCENTILE = 0.25


class CherryLikeMotif(object):

    def __init__(self, clustered_tips, root=None):
        """
        A motif with exceptions.
        :param root: ete.TreeNode, the root of the motif subtree
        :param clustered_tips: list of clustered child tips
        """
        self.root = root if root else clustered_tips[0].get_common_ancestor(*clustered_tips)
        self.clustered_tips = clustered_tips

    def __str__(self):
        return "Motif with root {} and {} clustered tips ({})" \
            .format(self.root.name, len(self.clustered_tips), ', '.join(_.name for _ in self.clustered_tips))


def nonparametric_cherry_diff(cherries, k=DEFAULT_NEIGHBOURHOOD_SIZE, repetitions=RANDOM_REPETITIONS,
                              percentile=DEFAULT_PERCENTILE):
    """
    Calculates the branch differences in real (pseudo)cherries and in random ones.
    Random cherry configuration is generated by replacing each real (pseudo)cherry
    by a (pseudo)cherry whose tips are randomly drawn from a k-neighbourhood of the real (pseudo)cherry.
    The k-neighbourhood contains the k cherries whose roots are the closest in time to the (pseudo)cherry in question,
    including itself.

    We compare the first percentile of the real (pseudo)cherry branch differences
    to the distribution of those for the random configurations.

    :param repetitions: how many times a random (pseudo)cherry configuration should be generated.
    :param k: the size of the k-neighbourhood.
    :return: the proportion, between 0 and 1,
        of random 1st quantiles of (pseudo)cherry diffs less or equal to the real one.
    """
    cherries = sorted(cherries, key=lambda _: getattr(_.root, TIME))
    n = len(cherries)

    def get_diff(b1, b2):
        return abs(b1.dist - b2.dist)

    def get_diff_distribution(tip_list_collection):
        data, weights = [], []
        for tips in tip_list_collection:
            size = len(tips)
            # weight is calculated as (size / 2) / C^2_{size} = (size / 2) / (size * (size - 1) / 2) = 1 / (size - 1),
            # so that in total all possible cherries in this polytomy count as size / 2 cherries
            weight = 1 / (size - 1)
            # let's add all possible cherries with this weight
            for i in range(size - 1):
                for j in range(i + 1, size):
                    data.append(get_diff(tips[i], tips[j]))
                    weights.append(weight)
        return np.array(data, dtype=np.float64), np.array(weights, dtype=np.float64)

    def generate_random_cherries():
        for i in range(n):
            # replace the i-th (pseudo)cherry with a randomly generated one from the branches in its neighbourhood
            cherry = cherries[i]
            time_i = getattr(cherry.root, TIME)
            # get k nearest neighbours, our cherry being the closest
            neighbours = sorted(cherries[max(i - k, 0): min(i + k + 1, n)],
                                key=lambda _: (abs(getattr(_.root, TIME) - time_i),
                                               0 if _ == cherry else 1))[:k]
            # out of their tips pick as many as in our (pseudo)cherry
            yield np.random.choice(reduce(lambda l1, l2: l1 + l2, (_.clustered_tips for _ in neighbours), []),
                                   size=len(cherry.clustered_tips), replace=False)

    real_qt = quantile(*get_diff_distribution((_.clustered_tips for _ in cherries)), percentile)

    def work(args):
        return quantile(*get_diff_distribution(generate_random_cherries()), percentile)

    with ThreadPool() as pool:
        random_qts = pool.map(work, (() for _ in range(int(repetitions))))

    return sum(1 for _ in random_qts if _ <= real_qt) / repetitions


def pick_cherries(tree, include_polytomies=True):
    """
    Picks cherries that satisfy the given values of pn_lookback, pn_delay in the given tree.

    :param include_polytomies: bool, whether to transform polytomies in the tree into cherries,
        i.e. a polytomy of n tips will be transformed into n(n-1) cherries.
    :param tree: ete3.Tree, the tree of interest
    :return: iterator of Motif motifs
    """
    for cherry_root in (set(tip.up for tip in tree) if not tree.is_leaf() else set()):
        if not include_polytomies and len(cherry_root.children) != 2:
            continue
        tips = sorted([_ for _ in cherry_root.children if _.is_leaf()], key=lambda _: _.dist)
        if len(tips) < 2:
            continue
        yield CherryLikeMotif(clustered_tips=tips, root=cherry_root)


def pn_test(forest, block_size=DEFAULT_CHERRY_BLOCK_SIZE, k=DEFAULT_NEIGHBOURHOOD_SIZE, repetitions=RANDOM_REPETITIONS,
            percentile=DEFAULT_PERCENTILE):
    """
    Tests if the input forest was generated under a -PN model.

    The test detects cherries in the forest, sorts them by the times of their roots,
    and splits them into blocks of 'block_size' cherries.
    For each cherry in a block, the test calculates the difference between its tip times,
    hence obtaining an array of 'block_size' cherry tip differences.
    It then generates a collection of random cherry tip differences for the same block:
    For each original cherry root it picks k cherries with the roots that are the closest in time,
    randomly selects two tips among their tips, and calculates their time difference.
    An array of 'block_size' reshuffled cherry tip differences is thus obtained for the same block.
    The reshuffled cherry tip difference array generation is repeated 'repetition' times.
    Finally, the test reports the proportion of reshuffled arrays whose 'percentile' percentile is smaller
    than that of the real cherry array.

    The test therefore reports a probability of partner notification
    being present at time interval corresponding to each block.
    To estimate the probability of partner notification being present in the whole tree,
    it averages the block probabilities.

    :param forest: list of trees
    :param block_size: number of cherries per block
    :param k: number of cherry neighbours for random cherry generation
    :param repetitions: number of random cherry tip difference arrays
    :param percentile: percentile (between 0 and 1) to consider
        when comparing real and random cherry tip difference arrays
    :return: tuple(pval, results, root_times), where pval is the global -PN test result,
        results is an array storing results for each block,
        root_times is an array storing min and max root times for each block
    """
    annotate_forest_with_time(forest)

    all_cherries = []
    for tree in forest:
        all_cherries.extend(pick_cherries(tree, include_polytomies=True))
    all_cherries = sorted(all_cherries, key=lambda _: getattr(_.root, TIME))

    n_cherries = len(all_cherries)
    logging.info('Picked {} cherries with {} roots.'.format(n_cherries, len({_.root for _ in all_cherries})))

    results = []
    root_times = []
    if n_cherries < block_size:
        raise ValueError('The block size (argument --block_size) must be smaller '
                         'than the number of cherries in the input tree ({}).'.format(n_cherries))
    for i in range(0, n_cherries // block_size):
        cherries = all_cherries[i * block_size: (i + 1) * block_size]
        result = nonparametric_cherry_diff(cherries, repetitions=repetitions, k=k, percentile=percentile)
        results.append(result)

        root_times.append((getattr(cherries[0].root, TIME), getattr(cherries[-1].root, TIME)))

        logging.info(
            'For {n} cherries with roots in [{start}-{stop}), PN test {res}.'
            .format(res=result, start=getattr(cherries[0].root, TIME), stop=getattr(cherries[-1].root, TIME),
                    n=len(cherries)))

    return sum(results) / len(results), results, root_times


def main():
    """
    Entry point for PN test with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="""PN-test.
        
Checks if the input forest was generated under a -PN model.
    
The test detects cherries in the forest, sorts them by the times of their roots, 
and splits them into blocks of 'block_size' cherries. 
For each cherry in a block, the test calculates the difference between its tip times, 
hence obtaining an array of 'block_size' cherry tip differences. 
It then generates a collection of random cherry tip differences for the same block: 
For each original cherry root it picks k cherries with the roots that are the closest in time, 
randomly selects two tips among their tips, and calculates their time difference. 
An array of 'block_size' reshuffled cherry tip differences is thus obtained for the same block. 
The reshuffled cherry tip difference array generation is repeated 'repetition' times. 
Finally, the test reports the proportion of reshuffled arrays whose 'percentile' percentile is smaller 
than that of the real cherry array.

The test therefore reports a probability of partner notification 
being present at time interval corresponding to each block. 
To estimate the probability of partner notification being present in the whole tree, 
it averages the block probabilities.""")
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="input forest file in newick or nexus format")
    parser.add_argument('--block_size', default=DEFAULT_CHERRY_BLOCK_SIZE, type=int, help="number of cherries per block")
    parser.add_argument('--k', default=DEFAULT_NEIGHBOURHOOD_SIZE, type=int,
                        help="number of cherry neighbours for random cherry generation")
    parser.add_argument('--repetitions', default=RANDOM_REPETITIONS, type=int,
                        help="number of random cherry tip difference arrays")
    parser.add_argument('--percentile', default=DEFAULT_PERCENTILE, type=float,
                        help="percentile (between 0 and 1) to consider "
                             "when comparing real and random cherry tip difference arrays")
    parser.add_argument('--report_blocks', action="store_true", help="report p-values for each block as well")
    params = parser.parse_args()

    forest = read_forest(params.nwk)
    pval, results, root_times = pn_test(forest, block_size=params.block_size, k=params.k,
                                        repetitions=params.repetitions, percentile=DEFAULT_PERCENTILE)

    logging.info("Total PN test {}.".format(pval))

    with open(params.log, 'w+') as f:
        f.write('Total\t{}\n'.format(pval))
        if params.report_blocks:
            for years, val in zip(root_times, results):
                f.write('{}-{}\t{}\n'.format(*years, val))


if __name__ == '__main__':
    main()
