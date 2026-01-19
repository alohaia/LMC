import subprocess

from lmc import config

for m in config.mice:
    for c in config.conditions:
        print(f">>>>> Running for {m}, condition {c}. <<<<<")

        subprocess.run(
            [
                "python",
                "./microBlooM/blood_flow_model.py",
                "-m", m,
                "-c", c,
                "-f", "pkl",
            ],
            check=True,
        )

        subprocess.run(
            [
                "python",
                "./microBlooM/blood_flow_model.py",
                "-m", m,
                "-c", c,
                "-f", "vtp",
            ],
            check=True,
        )

        print(f">>>>> Run finished: {m}, condition {c}. <<<<<")

