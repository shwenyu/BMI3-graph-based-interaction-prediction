# Hanyu Shou · BMI3 Python prototype

Run from the branch root with Python 3.10+ and NumPy installed.

    python source/download_data.py
    python source/run.py --dataset nr --seed 42

Choose e, ic, gpcr or nr. The downloader retrieves the Yamanishi gold-standard TXT files
from http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/ into input/.
Output paths include dataset and seed. Use a fresh --output directory to replay a run.
edges.csv records the identity and split of every known interaction.
validation.csv compares three fusion weights on a shared visible graph.
test.csv compares five methods after restoring validation edges.
per_drug.csv lists top-ten hits, precision and recall. Macro averages include drugs
with held-out positives. Original unrecorded pairs are evaluation background.
This AI-assisted prototype uses NumPy broadcasting (W1), edge lists, adjacency
matrices and visible degree (W7). Review the rules and hand-check examples as a group.
