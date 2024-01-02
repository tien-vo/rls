__all__ = ["work_dir", "data_dir", "data_store"]

from pathlib import Path

import zarr

work_dir = (Path(__file__).parent / ".." / "..").resolve()
data_dir = work_dir / "data"
data_store = zarr.DirectoryStore(data_dir)
