"""
regenerate_geometry_pnma_xyz.py — Re-generate the per-slab geometry JSONs
from the new orthogonal-cell Pnma POSCARs in data/pos3D2Dpnma_xyz/.

For each n=1..5:
    read data/pos3D2Dpnma_xyz/pnma_xyz_n{n}.vasp
    parse via parse_geometry.parse_structure()
    write results/data/geometry_n{n}.json   (OVERWRITES existing!)

The 3D bulk JSON (geometry_3D.json) is NOT touched — the 3D bulk POSCAR
is already orthogonal so no change is needed.

Backups should already exist as geometry_n*_organic.json.bak (run
_backup_and_regenerate.sh first).
"""

import os
import json
import parse_geometry as pg

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
_IN   = os.path.join(_ROOT, 'data', 'pos3D2Dpnma_xyz')
_OUT  = os.path.join(_ROOT, 'results', 'data')


def main():
    print(f'Input dir : {_IN}')
    print(f'Output dir: {_OUT}\n')

    for n in range(1, 6):
        label = f'n{n}'
        in_path = os.path.join(_IN, f'pnma_xyz_n{n}.vasp')
        out_path = os.path.join(_OUT, f'geometry_{label}.json')

        if not os.path.exists(in_path):
            print(f'SKIP: {in_path} not found')
            continue

        print(f'--- {label} ---')
        data = pg.parse_structure(label, in_path)
        pg.print_summary(data)

        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'  → wrote {out_path}\n')

    print('Done. New geometry_n*.json files based on the orthogonal Pnma POSCARs.')


if __name__ == '__main__':
    main()
