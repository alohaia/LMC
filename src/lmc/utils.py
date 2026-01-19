from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

def vtk2vtp(invtkfile, outvtpfile, binary=False):
    import vtk

    reader = vtk.vtkPolyDataReader()
    # reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(invtkfile)

    reader.Update()  # load to memory

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outvtpfile)

    writer.SetInputConnection(reader.GetOutputPort())

    if binary:
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()

    writer.Write() # equals to Update()

    print(f"Successfully converted {invtkfile} to {outvtpfile}")


def dist1d(arr: ArrayLike, title="Numerical Distribution", logscale: bool = False):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(np.shape(arr)[0]), arr, alpha=0.5, s=10)
    plt.title(title)
    if logscale:
        plt.yscale('symlog', linthresh=1e-5) 
        plt.ylabel("Value (SymLog Scale)")
        # plt.gca().yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    else:
        plt.ylabel("Value")
    plt.xlabel("Index")
    plt.grid(True, which='both', alpha=0.3)
    plt.axhline(0, color='red', linewidth=0.8, linestyle='--')
    plt.show()
