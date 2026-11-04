import numpy as np

from src.burst_pressure_prediction import predict_burst_pressure
from src.io_utils import make_result_dir, save_json
from src.paths import CONFIG_DIR, RESULT_DIR

path = CONFIG_DIR / "defaultIV.yaml"
slug = path.stem
out_dir = make_result_dir(RESULT_DIR, slug, timestamp=False)

r = predict_burst_pressure(path, d_limit=0.95)
history = r["history"]

# 保存标量结果用 JSON
save_json(
    out_dir,
    [{"p_burst_MPa": r["p_burst_MPa"], "ratio": r["ratio"]}],
    filename="defaultIV_solution.json",
)

# 保存数组历史用 npz (必须用 numpy 专属格式)
np.savez(
    out_dir / "history.npz",
    p=np.array([h[0] for h in history]),
    u=np.array([h[1] for h in history]),
    n=np.array([h[2] for h in history]),
    d_ft=np.array([h[3] for h in history]),
    d_fc=np.array([h[4] for h in history]),
)
print(f"  saved to: {out_dir}")
