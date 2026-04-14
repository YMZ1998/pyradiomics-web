# -- coding: utf-8 --
import os
import subprocess
import sys
import traceback
from multiprocessing import Process, Queue

import pandas as pd
from tqdm import tqdm

DEFAULT_DATA_ROOT = r'D:\code\pyradiomics\Classification\test_data'


def list_kind_folders(data_root):
    """Return class names from sub-folder names under data_root."""
    kinds = []
    for name in os.listdir(data_root):
        full_path = os.path.join(data_root, name)
        if os.path.isdir(full_path) and not name.startswith('.'):
            kinds.append(name)
    kinds.sort()
    return kinds


def create_extractor(para_path='./CT-extractor.yaml'):
    """Create a pyradiomics extractor from yaml config."""
    from radiomics import featureextractor

    return featureextractor.RadiomicsFeatureExtractor(para_path)


def check_runtime_dependencies():
    """Check whether critical native deps can be imported in a clean subprocess."""
    checks = [
        ('python', 'import sys; print(sys.version)'),
        ('SimpleITK', 'import SimpleITK as sitk; print(sitk.Version_VersionString())'),
        ('radiomics', 'import radiomics; print(radiomics.__version__)'),
    ]
    for name, code in checks:
        proc = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                '[ENV ERROR] {} import failed, returncode={}, stderr={}'.format(
                    name, proc.returncode, (proc.stderr or '').strip()
                )
            )
        print('[ENV OK] {} -> {}'.format(name, (proc.stdout or '').strip()))


def find_image_and_mask_paths(case_dir):
    """Find original image path and seg mask path from one case folder."""
    ori_path = None
    lab_path = None
    for filename in sorted(os.listdir(case_dir)):
        full_path = os.path.join(case_dir, filename)
        if not os.path.isfile(full_path):
            continue
        if 'seg' in filename:
            lab_path = full_path
        else:
            ori_path = full_path
    if ori_path is None or lab_path is None:
        raise ValueError('Cannot find image/mask pair in {}'.format(case_dir))
    return ori_path, lab_path


def extract_single_case_features(extractor, ori_path, lab_path):
    """Extract one case features and filter diagnostics config/version keys."""
    features = extractor.execute(ori_path, lab_path)
    filtered = {}
    for key, value in features.items():
        if 'diagnostics_Versions' in key or 'diagnostics_Configuration' in key:
            continue
        filtered[key] = value
    return filtered


def _extract_case_worker(ori_path, lab_path, para_path, out_queue):
    """Run one case extraction in child process to avoid hard crash of main process."""
    try:
        extractor = create_extractor(para_path)
        features = extract_single_case_features(extractor, ori_path, lab_path)
        out_queue.put({'ok': True, 'features': features})
    except Exception as exc:
        out_queue.put({'ok': False, 'error': str(exc), 'traceback': traceback.format_exc()})


def extract_single_case_features_isolated(ori_path, lab_path, para_path, timeout=600):
    """Extract one case in a subprocess; return (ok, payload)."""
    out_queue = Queue(maxsize=1)
    proc = Process(target=_extract_case_worker, args=(ori_path, lab_path, para_path, out_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, {'error': 'timeout', 'traceback': 'timeout {}s'.format(timeout)}

    if proc.exitcode != 0:
        return False, {'error': 'subprocess crashed', 'traceback': 'exit code {}'.format(proc.exitcode)}

    if out_queue.empty():
        return False, {'error': 'no result from subprocess', 'traceback': ''}

    result = out_queue.get()
    return result.get('ok', False), result


def extract_kind_features(kind, extractor, data_root=DEFAULT_DATA_ROOT, para_path='./CT-extractor.yaml',
                          isolate_case=True):
    """Extract all cases for one class label (kind)."""
    kind_path = os.path.join(data_root, kind)
    case_folders = sorted(os.listdir(kind_path))
    if '.DS_Store' in case_folders:
        case_folders.remove('.DS_Store')

    rows = []
    failed_cases = []
    for folder in tqdm(case_folders):
        case_dir = os.path.join(kind_path, folder)
        ori_path, lab_path = find_image_and_mask_paths(case_dir)

        if isolate_case:
            ok, payload = extract_single_case_features_isolated(
                ori_path,
                lab_path,
                para_path=para_path,
            )
            if ok:
                rows.append(payload['features'])
            else:
                failed_cases.append((folder, ori_path, lab_path, payload))
                print('[FAILED] kind={}, case={}, image={}, mask={}, detail={}'.format(
                    kind, folder, ori_path, lab_path, payload
                ))
        else:
            rows.append(extract_single_case_features(extractor, ori_path, lab_path))

    if failed_cases:
        print('[WARN] kind={} failed {}/{} cases'.format(kind, len(failed_cases), len(case_folders)))
    else:
        print('[OK] kind={} all {} cases extracted'.format(kind, len(case_folders)))

    return pd.DataFrame(rows)


def extract_and_save_all_kinds(kinds=None, para_path='./CT-extractor.yaml', data_root=DEFAULT_DATA_ROOT,
                               isolate_case=True):
    """Run feature extraction for all kinds and save <kind>.csv."""
    check_runtime_dependencies()

    if kinds is None:
        kinds = list_kind_folders(data_root)

    extractor = create_extractor(para_path)
    for kind in kinds:
        print('{}: 开始提取特征'.format(kind))
        df = extract_kind_features(
            kind,
            extractor,
            data_root=data_root,
            para_path=para_path,
            isolate_case=False,
        )
        df.to_csv('test_data/{}.csv'.format(kind), index=0)
    print('完成')


if __name__ == '__main__':
    extract_and_save_all_kinds()
