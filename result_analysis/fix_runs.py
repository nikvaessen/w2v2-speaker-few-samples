import wandb
from dotenv import load_dotenv

load_dotenv("../.env")


api = wandb.Api()
runs = api.runs("wav2vec2")

fix_name = "extensive-routines-8"
test_eer_extended, test_eer_hard = 0.1103,0.18

for r in runs:
    if r.name == fix_name:
        r.summary["test_eer_hard"] = test_eer_hard
        r.summary["test_eer_extended"] = test_eer_extended
        r.update()
