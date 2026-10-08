import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import httpx

from openso101.scenes.models import file_digest


def fetch_assets(manifest, root, output):
    root = root.resolve()
    content = json.loads(manifest.read_text())
    if content["schema_version"] != 1 or not content["assets"]:
        raise ValueError("CPU 资产列表需要有效的版本和资产")
    if output.exists():
        raise FileExistsError(output)
    assets = []
    with httpx.Client(timeout=30) as client:
        for item in content["assets"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root / "outputs"):
                raise ValueError("CPU 资产必须保存到 outputs 目录")
            if path.exists():
                if file_digest(path) != item["sha256"]:
                    raise ValueError(f"已有 CPU 资产的 SHA256 不一致：{path}")
                downloaded = False
            else:
                response = client.get(item["url"])
                response.raise_for_status()
                if hashlib.sha256(response.content).hexdigest() != item["sha256"]:
                    raise ValueError("CPU 资产的下载内容 SHA256 不一致")
                model = ElementTree.fromstring(response.content)
                if model.tag != "mujoco" or model.attrib["model"] != item["xml_model"]:
                    raise ValueError("CPU 资产的 MuJoCo XML 模型名称不一致")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(response.content)
                downloaded = True
            assets.append({"path": str(path), "url": item["url"], "sha256": file_digest(path),
                           "downloaded": downloaded})
    report = {"status": "cpu_asset_sources_verified", "manifest_sha256": file_digest(manifest),
              "assets": assets, "gpu_tests_started": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/validation/cpu_assets.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch_assets(args.manifest, args.root, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
