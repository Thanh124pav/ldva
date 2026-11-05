# DataMIL / AMF reproduction handoff

Source changes in external Git repositories are preserved in `vendor_patches/`, not as embedded repositories or checkpoint/data files. `vendor_manifest.json` pins each upstream revision, including Octo and the older MetaWorld API.

Run `python reproduction/restore_vendor.py` to clone missing repos and apply patches. Existing checkouts must be at the recorded commit; unexpected modifications are rejected. Patches already fully applied are recognized. This does not install environments or download datasets.

Execution scripts and durable scientific notes are in `scratchpad/`. They currently target `/mnt/data/thanhpv/ldva` and existing Conda environments/data paths on the original host. Adjust host-specific paths before use elsewhere, including the editable Octo environment installation. W&B credentials are read privately from `/mnt/data/thanhpv/wandb.txt`; no credential is committed.

Current entry point: `scratchpad/resume_paper_queue.py`, launched in a detached screen outside sandbox for GPU access. It skips completed AMF work and resumes the unfinished DataMIL H15/T30 queue. See `scratchpad/resume_20261105_64276.md` for the latest allocation/runtime adjustment and `scratchpad/reproduction_overnight_report.md` for result caveats. A screen process does not survive Slurm allocation expiry.

DataMIL datamodel worker count is automatically reduced on cgroups below48GiB RAM. Scientific batch/LR/step budgets remain unchanged; bitwise data-order equivalence across different worker counts is not claimed. Forward checkpoints every100steps; backward phases restart if interrupted; final-policy checkpoints every1000steps.

AMF MetaWorld's original60 runs completed but did not match paper performance. The success-only data diagnostic seed completed at50%; it does not establish that filtering fixes the gap. Robomimic81% is one seed with simulator/data and beta differences. Smoke tests are execution evidence, not paper results. Consult the diagnosis and validation Markdown files before reporting reproduction success.

Datasets, checkpoint binaries, Conda environments, full console logs and W&B runtime files are intentionally excluded from Git. Existing unrelated deletions of root README.md and PLAN.md were not included in this reproduction commit.
