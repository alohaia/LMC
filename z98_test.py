import pandas as pd
from lmc.core import io

xl_file = pd.ExcelFile("data/11_hypEVs/11_hypEVs.xlsx")
dfs = {sheet_name: xl_file.parse(sheet_name)
          for sheet_name in xl_file.sheet_names}

g = io.create(data_xlsx="./data/11_hypEVs/11_hypEVs.xlsx")

breakpoint()
