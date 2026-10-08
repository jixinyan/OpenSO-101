import pytest

from openso101.il.datasets.export import _push_detect_input_format, _push_validate_local_dataset


def test_missing_dataset_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="数据集目录不存在"):
        _push_validate_local_dataset(tmp_path / "missing")


def test_empty_directory_format(tmp_path):
    with pytest.raises(ValueError, match="无法识别数据集格式"):
        _push_detect_input_format(tmp_path)


def test_missing_lerobot_metadata(tmp_path):
    (tmp_path / "meta").mkdir()
    with pytest.raises(ValueError, match="metadata 缺少文件"):
        _push_validate_local_dataset(tmp_path, input_format="lerobot")


def test_unknown_dataset_format(tmp_path):
    with pytest.raises(ValueError, match="无法识别数据集格式"):
        _push_validate_local_dataset(tmp_path, input_format="unknown")
