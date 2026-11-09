"""Project = a parent folder whose subfolders are each a measured scan folder.

combine_project() merges every subfolder's raw_data.csv into one raw_data.csv
in the parent folder's own output folder, adding a 'Folder' column. Genotypes
with the same name are pooled when plotted.
"""
from pathlib import Path

import pandas as pd

from csv_output import generate_tidy
from plotting import plot_results, plot_segments_facet

_OUT_ROOT = 'root_measure_lr'


def find_project_subfolders(parent):
    """Immediate subfolders of parent (excluding the app's output folder)."""
    parent = Path(parent)
    return sorted((p for p in parent.iterdir()
                   if p.is_dir() and not p.name.startswith('.')
                   and p.name != _OUT_ROOT),
                  key=lambda p: p.name.lower())


def project_subfolders_with_scans(parent):
    """Immediate subfolders of parent that contain at least one scan image."""
    from utils import list_images_in_folder
    out = []
    for sub in find_project_subfolders(parent):
        try:
            if list_images_in_folder(sub):
                out.append(sub)
        except OSError:
            continue
    return out


def scan_progress(sub):
    """(n_measured, n_total) scans for a folder, across all experiment names."""
    from utils import list_images_in_folder
    n_total = len(list_images_in_folder(sub))
    done = set()
    base = Path(sub) / _OUT_ROOT
    if base.is_dir():
        for raw in base.glob('**/data/raw_data.csv'):
            try:
                done.update(pd.read_csv(raw, usecols=['Image'])['Image']
                            .astype(str).tolist())
            except Exception:
                continue
    return len(done), n_total


def combine_project(parent, data_dir_fn, exp=''):
    """Merge each subfolder's raw_data.csv into <parent>'s data folder.

    Returns (raw_path, folders_used, folders_skipped) or None if no subfolder
    has measured data. Image names are prefixed with the subfolder name so
    identical scan filenames in different folders never collide.
    """
    parent = Path(parent)
    frames, used, skipped = [], [], []
    for sub in find_project_subfolders(parent):
        raw = data_dir_fn(sub, exp) / 'raw_data.csv'
        if not raw.exists():
            skipped.append(sub.name)
            continue
        df = pd.read_csv(raw, dtype={'Plate_Label': str})
        if df.empty:
            skipped.append(sub.name)
            continue
        df.insert(0, 'Folder', sub.name)
        if 'Image' in df.columns:
            df['Image'] = sub.name + '/' + df['Image'].astype(str)
        frames.append(df)
        used.append(sub.name)
    if not frames:
        return None
    out_dir = data_dir_fn(parent, exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    if 'Root_ID' in df.columns:
        df['Root_ID'] = range(1, len(df) + 1)
    raw_path = out_dir / 'raw_data.csv'
    if 'CR_Total' in df.columns:
        df['CR_Total'] = pd.to_numeric(df['CR_Total'], errors='coerce').astype('Int64')
    df.to_csv(raw_path, index=False)
    return raw_path, used, skipped


def plot_each_folder(parent, data_dir_fn, exp=''):
    """Plot every measured subfolder on its own (data, plots and statistics
    in that subfolder's own output folder). Returns (plotted, skipped) names."""
    plotted, skipped = [], []
    for sub in find_project_subfolders(Path(parent)):
        raw = data_dir_fn(sub, exp) / 'raw_data.csv'
        if not raw.exists():
            skipped.append(sub.name)
            continue
        try:
            if pd.read_csv(raw, nrows=1).empty:
                skipped.append(sub.name)
                continue
            plot_project(raw)
            plotted.append(sub.name)
        except Exception as e:
            skipped.append(f"{sub.name} ({e})")
    return plotted, skipped


def plot_project(raw_path):
    """Write tidy_data.csv, plots and statistics.txt next to raw_path."""
    raw_path = Path(raw_path)
    generate_tidy(raw_path, raw_path.with_name('tidy_data.csv'), csv_format='R')
    summaries = []

    def _add(s):
        if s:
            summaries.append(s)

    _add(plot_results(raw_path, value_col='Length_cm',
                      ylabel='Primary root length (cm)', csv_format='R'))
    cols = list(pd.read_csv(raw_path, nrows=0).columns)
    if 'LR_Density' in cols:
        _add(plot_results(raw_path, value_col='LR_Density',
                          ylabel='Lateral root density (roots/cm)',
                          csv_format='R'))
    if 'CR_Total' in cols and pd.read_csv(
            raw_path, usecols=['CR_Total'])['CR_Total'].notna().any():
        _add(plot_results(raw_path, value_col='CR_Total',
                          ylabel='Crown roots per plant', csv_format='R'))
    seg_cols = [c for c in cols if c.startswith('Segment_') and c.endswith('_cm')]
    for sc in seg_cols:
        num = sc.replace('Segment_', '').replace('_cm', '')
        _add(plot_results(raw_path, value_col=sc,
                          ylabel=f'Segment {num} length (cm)', csv_format='R'))
    if len(seg_cols) >= 2:
        plot_segments_facet(raw_path, seg_cols, csv_format='R')
    if summaries:
        raw_path.with_name('statistics.txt').write_text('\n\n'.join(summaries) + '\n')
    return summaries
