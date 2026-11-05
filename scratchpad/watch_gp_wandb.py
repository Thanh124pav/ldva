"""Mirror durable GP checkpoints to online W&B while experiments run."""
import json
from pathlib import Path
import pickle
import sys
import time
import wandb

root = Path(sys.argv[1]).resolve()
identity = root / 'online_run_id.txt'
run_id = identity.read_text().strip() if identity.exists() else wandb.util.generate_id()
identity.write_text(run_id)
run = wandb.init(project='paper-reproduction', id=run_id, resume='allow',
                 group='amf-gp', name='AMF-GP-released-notebook-live',
                 dir=str(root), config=dict(paper='2410.05026v3', seeds=10,
                 prior_supports=12, budget=40, eval_episodes_per_task=20,
                 shooting_candidates=20, protocol='released notebook',
                 paper_difference='Appendix M.2 uses 100 candidates'))
print('WANDB_RUN_URL=' + run.url, flush=True)
seen = set()
while True:
    checkpoint = root / 'checkpoint.pkl'
    if checkpoint.exists():
        state = pickle.loads(checkpoint.read_bytes())
        for support, methods in sorted(state['metrics'].items()):
            for method, seeds in methods.items():
                for seed, result in seeds.items():
                    key = (support, method, seed)
                    if key in seen:
                        continue
                    for round_id, value in enumerate(result['performance'].mean(axis=0)):
                        run.log(dict(prior_support=support, seed=seed, round=round_id,
                                     **{method+'/return':float(value),
                                        method+'/uncertainty':float(result['uncertainties'].mean(axis=0)[round_id])}))
                    seen.add(key)
        run.summary['completed_method_runs'] = len(seen)
        summary = json.loads((root / 'summary.json').read_text())
        for support, methods in summary.items():
            for method, stats in methods.items():
                for metric, value in stats.items():
                    run.summary[f'prior_{support}/{method}/{metric}'] = value
        if len(seen) == 240:
            artifact = wandb.Artifact('amf-gp-reproduction-results', type='evaluation')
            for filename in ['checkpoint.pkl','summary.json','results.md']:
                artifact.add_file(str(root / filename))
            run.log_artifact(artifact)
            run.finish()
            break
    time.sleep(15)
