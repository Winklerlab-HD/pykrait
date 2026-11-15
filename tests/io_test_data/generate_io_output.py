import numpy as np
from pykrait.io.io import load_timelapse_lazy

arr, frame_interval, y_um, x_um = load_timelapse_lazy(
    "tests/io_test_data/iotestdata1.tif"
)

# Save as NumPy arrays (after computing from Dask)
np.save("tests/io_test_data/expected_array_iotestdata1.npy", arr.compute())
np.save(
    "tests/io_test_data/expected_metadata_iotestdata1.npy",
    np.array([frame_interval, y_um, x_um]),
)
