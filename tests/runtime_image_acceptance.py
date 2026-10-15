import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sklearn.linear_model import LinearRegression

assert os.getuid() == 10001
assert platform.machine() == "x86_64"
output = Path("output")
output.mkdir(parents=True, exist_ok=True)
frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
path = output / "offline-mini.parquet"
frame.to_parquet(path, index=False)
pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
model = LinearRegression().fit(frame[["feature"]], frame["target"])
assert np.allclose(model.predict(frame[["feature"]]), frame["target"])
values = torch.tensor([1.0, 2.0], requires_grad=True)
values.square().sum().backward()
assert torch.allclose(values.grad, torch.tensor([2.0, 4.0]))
assert torch.version.cuda is None
metadata = Metadata.detect_from_dataframe(data=frame, table_name="offline_sample")
assert metadata.to_dict()
(output / "evidence.json").write_text(
    json.dumps(
        {
            "uid": os.getuid(),
            "architecture": platform.machine(),
            "parquet_rows": len(frame),
            "linear_regression_pass": True,
            "cpu_backward_pass": True,
            "metadata_pass": True,
        }
    )
)
print("DEFAULT-OUTPUT ACCEPTANCE PASS uid=10001 architecture=x86_64 rows=4")
