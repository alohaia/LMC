#!/usr/bin/env python

import sys
from lmc.utils import vtk2vtp

# https://gist.github.com/thomasballinger/1281457

if __name__ == '__main__':
    args = sys.argv
    binary = False

    if '-b' in args:
        args.remove('-b')
        binary = True

    if len(args) < 2:
        print('Batch converts vtk files to vtp files.\n'
              'Usage:\n'
              '\tvtk2vtp.py model1.vtk model2.vtk ...\n\n'
              '\t-b\tcauses output to be in binary format, much smaller vtp'
                  'file size.'
              '\n')

        sys.exit()

    infiles = args[1:]
    for vtkfile in infiles:
        if vtkfile[-4:] != '.vtk':
            print(vtkfile, "doesn't look like a vtk file, won't convert")
            continue
        vtk2vtp(vtkfile, vtkfile[:-4]+'.vtp', binary=binary)
