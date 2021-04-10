import itertools
from datetime import datetime
from simple_slurm import Slurm

slurm = Slurm(
    array=range(3),
    gres='gpus:1',
    cpus_per_task=2,
    mem='20gb',
    job_name='tg',
    output=f'.logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    time='3:00:00'
)

datasets = ('lastfm', )
models = ('tgn', 'tsam')


for dataset, model in itertools.product(datasets, models):
    cmd = f'python train.py --dataset {dataset} --memory_type {model} --use_wandb --wandb_group "{datetime.now()}"'
    print(f'Scheduling command:\n{cmd}')
    print(slurm.sbatch(cmd)) # _run{Slurm.SLURM_ARRAY_TASK_ID}