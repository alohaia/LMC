import subprocess

from lmc import config

conditions = ["baseline", "MCAO0h", "MCAO1h"]

for m in config.mice:
    for c in conditions:
        print(f">>>>> Running for {m}, condition {c}. <<<<<")

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

