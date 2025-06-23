from pathlib import Path

# Define the custom checkpoint directory
custom_checkpoint_path = (
    Path(
        "C:/Users/mimib/Desktop/Masterarbeit Produktionsmanagement/artifacts/custom_checkpoints"
    )
    if Path("C:/Users/mimib").is_dir()
    else Path("../custom_checkpoints")
).resolve()

custom_pretrained_models_path = (
    Path(
        "C:/Users/mimib/Desktop/Masterarbeit Produktionsmanagement/artifacts/pretrained_models"
    )
    if Path("C:/Users/mimib").is_dir()
    else Path("../pretrained_models")
).resolve()

custom_bench_policy_data_path = (
    Path(
        "C:/Users/mimib/Desktop/Masterarbeit Produktionsmanagement/artifacts/bench_policy_data"
    )
    if Path("C:/Users/mimib").is_dir()
    else Path("../bench_policy_data")
).resolve()

custom_interesting_runs_path = (
    Path("C:/Users/mimib/Desktop/Masterarbeit Produktionsmanagement/interesting_runs")
    if Path("C:/Users/mimib").is_dir()
    else Path("../interesting_runs")
).resolve()

custom_trained_models_path = (Path("../trained_models")).resolve()
