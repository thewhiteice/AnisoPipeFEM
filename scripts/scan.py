import glob
import os
from pathlib import Path

from src.burst_pressure_prediction import predict_burst_pressure
from src.io_utils import make_result_dir, save_json
from src.paths import CONFIG_DIR, RESULT_DIR


def main():
    # 从环境变量读，.sh 里传
    glob_pattern = os.environ.get("SCAN_GLOB", str(CONFIG_DIR / "*.yaml"))
    param = os.environ["SCAN_PARAM"]
    values = [float(x) for x in os.environ["SCAN_VALUES"].split(",")]

    for path in sorted(glob.glob(glob_pattern, recursive=True)):
        path = Path(path)
        subdir = "refs" if "refs" in path.parts else ""

        results = []
        for v in values:
            r = predict_burst_pressure(path, **{param: v})
            r["scanned_param"] = param
            r["scanned_value"] = v
            results.append(r)
            if r["p_burst_MPa"] is None:
                print(f"[{r['slug']}] {param}={v:.3e} -> no knee")
            else:
                print(f"[{r['slug']}] {param}={v:.3e} -> {r['p_burst_MPa']:.2f} MPa")

        slug = path.stem
        out_dir = make_result_dir(RESULT_DIR / "scan", slug, subdir=subdir)
        save_json(out_dir, results, filename=f"scan_{param}.json")
        print(f"  saved: {out_dir}")


if __name__ == "__main__":
    main()
