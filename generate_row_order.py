import pandas as pd


def generate(basename, plants):
    for i, treatment in basename.iterrows():
        for d, plant in enumerate(plants):
            yield i * len(plants) + d + 1, 'N{}IR{}Rep{}_{}'.format(*treatment, plant)


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'i', type=str, help='file with treatment/collection order')
        parser.add_argument('o', type=str, help='name for output file')
        parser.add_argument('plants', type=int, nargs='+',
                            help='plant ids collected')
        return parser.parse_args()

    args = parse_args()

    basename = pd.read_csv(args.i, sep=None, engine='python')
    r = basename.shape[0] * len(args.plants)

    with open(args.o, 'w') as outfile:
        for i, row in generate(basename, args.plants):
            outfile.write(f'{row}')
            if i % r:
                outfile.write('\n')
