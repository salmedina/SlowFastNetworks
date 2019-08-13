import argparse
import numpy as np

def main(num_exp, save_path):
    lrs = np.random.uniform(1e-8, 2, num_exp)
    momentums = np.random.uniform(0.1, 0.9, num_exp)
    wds = np.random.uniform(1e-8, 1e-1, num_exp)

    params = np.stack([lrs, momentums, wds], axis=1)
    np.save(save_path, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_exp', dest='num_exp', type=int, required=True, help='Number of experiments to be generated')
    parser.add_argument('--save_path', dest='save_path', type=str, required=True, help='Path where the params will be saved')
    args = parser.parse_args()

    main(args.num_exp, args.save_path)