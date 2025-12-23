from pathlib import Path
from typing import Union
import pandas as pd
from pandas import DataFrame

def ensure_dirs(dirs: list[str]):
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)
        print(Path(dir), "created.")

def read_xlsx(path: Union[Path, str] = "") -> dict[str, DataFrame]:
    xl = pd.ExcelFile(path)
    dfs: dict[str, DataFrame] = {}

    for sheet_name in xl.sheet_names:
        dfs[sheet_name] = pd.DataFrame(xl.parse(sheet_name))

    return dfs
