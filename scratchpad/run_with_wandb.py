"""Read the user-provided W&B credential without exposing it in commands/logs."""
import argparse
import os
from pathlib import Path
import re
import runpy
import sys


def configure_wandb():
    contents = Path('/mnt/data/thanhpv/wandb.txt').read_text().strip()
    candidates = re.findall(r'(?<![A-Za-z0-9])(?:[a-fA-F0-9]{40}|wandb_v1_[A-Za-z0-9_-]+)(?![A-Za-z0-9])', contents)
    if len(set(candidates)) != 1:
        raise RuntimeError('Could not identify one unique W&B credential in wandb.txt')
    os.environ['WANDB_API_KEY'] = candidates[0]
    os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
    os.environ.setdefault('WANDB_ENTITY', 'hust_edu_vn')
    os.environ.setdefault('WANDB_PROJECT', 'paper-reproduction')
    os.environ.setdefault('WANDB_MODE', 'online')


if __name__ == '__main__':
    # The paper datamodel needs a Torch CPU loader alongside JAX CUDA. Re-exec
    # only this scoped pipeline after its conflict-isolated clone is validated.
    paper_python = Path('/mnt/data/thanhpv/miniconda3/envs/datamil-paper-h15-cpu-loader/bin/python')
    paper_ready = Path('/mnt/data/thanhpv/ldva/logs/datamil_paper_h15/environment.ready.json')
    if '--config.folder_name=book-caddy' in sys.argv and any('paper_h15' in arg for arg in sys.argv):
        # Wait before importing JAX/reserving the production pool. A smoke run
        # can still own GPU memory briefly after writing its completion marker.
        import psutil
        import time
        smoke_marker=paper_ready.parent/'gpu_smoke/datamodels/libero90_horizon/book-caddy-smoke/iter_0/complete.json'
        deadline=time.monotonic()+3600
        while True:
            active=False
            for process in psutil.process_iter(['cmdline']):
                command=process.info['cmdline'] or []
                if ('datamil/metagradients/do_libero_data_selection_parallel.py' in command
                        and '--config.folder_name=book-caddy-smoke' in command):
                    active=True
                    break
            if smoke_marker.exists() and not active:
                break
            if not active or time.monotonic()>deadline:
                raise RuntimeError('Paper GPU smoke has not passed; inspect gpu_smoke_console.log')
            print('Waiting for actual GPU smoke to complete and release GPU memory',flush=True)
            time.sleep(15)
    if (paper_ready.exists() and any('paper_h15' in arg for arg in sys.argv)
            and Path(sys.prefix) != paper_python.parent.parent):
        os.environ['DATAMIL_DM_CACHE_DIR'] = '/mnt/data/thanhpv/ldva/logs/datamil_paper_h15/jax_cache_cpu_loader'
        current_flags = os.environ.get('XLA_FLAGS', '')
        if 'xla_gpu_autotune_level' not in current_flags:
            os.environ['XLA_FLAGS'] = current_flags + ' --xla_gpu_autotune_level=0'
        production_dm = '--config.folder_name=book-caddy' in sys.argv
        production_policy = any('paper_h15_cotrain_config.py:' in arg for arg in sys.argv)
        if production_dm or production_policy:
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.80'
            os.environ['REPRO_VRAM_TARGET'] = '.90'
        os.execv(str(paper_python), [str(paper_python)] + sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('script', nargs='?')
    args, remaining = parser.parse_known_args()
    configure_wandb()
    if args.check:
        import wandb
        api = wandb.Api(timeout=30)
        viewer = api.viewer
        print('W&B authenticated; default entity:', api.default_entity)
    else:
        if not args.script:
            parser.error('script is required')
        sys.argv = [args.script] + remaining
        sys.path.insert(0, str(Path(args.script).resolve().parent))
        runpy.run_path(args.script, run_name='__main__')
