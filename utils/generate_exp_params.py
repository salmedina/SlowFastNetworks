import argparse
import numpy as np


def log10uniform(min, max, size):
    min_e = int(np.log10(min))
    max_e = int(np.log10(max))
    bases = np.random.uniform(1, 10, size)
    exponents = np.random.randint(min_e, max_e, size)
    return [ b * 10. ** e for b, e in zip(bases, exponents)]


def main(num_exp, save_path):
    lrs = log10uniform(1e-8, 10, num_exp)
    momentums = np.random.uniform(0.1, 0.9, num_exp)
    wds = log10uniform(1e-8, 1e-1, num_exp)

    params = np.stack([lrs, momentums, wds], axis=1)
    np.save(save_path, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_exp', dest='num_exp', type=int, required=True, help='Number of experiments to be generated')
    parser.add_argument('--save_path', dest='save_path', type=str, required=True, help='Path where the params will be saved')
    args = parser.parse_args()

    main(args.num_exp, args.save_path)