from typing import List

import argparse
import itertools
from datetime import datetime
from simple_slurm import Slurm

def main(
    spans: List[int],
    datasets: List[str],
    num_trials: int = 5,
    time: str = '5:00:00',
    dryrun: bool = False
):
    slurm = Slurm(
        array=range(num_trials),
        gres='gpus:1',
        cpus_per_task=2,
        mem='10gb',
        job_name='tg',
        output=f'.logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        time=time
    )

    cmd_template = 'python train.py --dataset {dataset} --expire_span {span} --use_wandb --wandb_group "{time}"'

    grid = list(itertools.product(datasets, spans))
    print(f'Scheduling {len(grid)*num_trials} jobs.')

    for dataset, span in grid:
        cmd = cmd_template.format(dataset=dataset, span=span, time=datetime.now()) # , rank=Slurm.SLURM_ARRAY_TASK_ID)
        if dryrun:
            print(f'Generated command: {cmd}')
            continue

        print(f'Scheduling command:\n{cmd}')
        print(slurm.sbatch(cmd))

if __name__ == '__main__':
    """suggested spans:
        mooc:       -1 10 100 1000
        lastfm:     -1 1000 10000 100000
        wikipedia:  -1 1000 10000 100000
        reddit:     -1 1000 10000 100000
    """

    example_usage = """example usage:
    python submit.py -s -1 10 100 -d mooc
    python submit.py -s -1 1000 10000 100000 -d lastfm wikipedia reddit
    """

    parser = argparse.ArgumentParser(epilog=example_usage)
    parser.add_argument('-s', '--spans', nargs='+', type=int, help="Length(s) of expire_span (-1 for no expire_span)")
    parser.add_argument('-d', '--datasets', nargs='+', type=str, choices=['mooc', 'wikipedia', 'reddit', 'lastfm'])
    parser.add_argument('-n', '--num_trials', type=int, default=5, help="Number of trials for each experiment")
    parser.add_argument('-t', '--time', type=str, default='5:00:00')
    parser.add_argument('--dryrun', action='store_true', help="Print commands instead of scheduling")

    args = parser.parse_args()
    main(**vars(args))