# coding: utf-8
"""
将 OpenAI 兼容接口需要的额外 Python 包安装到便携版 internal 目录。

当前接口实现只使用标准库，所以 requirements-openai-api.txt 默认没有实际依赖。
保留这个脚本是为了以后项目更新或接口改造时，可以把新增依赖集中写入
requirements-openai-api.txt，并单独安装到 dist/CapsWriter-Offline/internal，
避免改动主项目依赖和打包配置。

用法：
    D:/anaconda3/envs/c/python.exe package_openai_api_internal.py
    D:/anaconda3/envs/c/python.exe package_openai_api_internal.py --internal dist/MyBuild/internal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REQUIREMENTS = BASE_DIR / "requirements-openai-api.txt"
DEFAULT_INTERNAL = BASE_DIR / "dist" / "CapsWriter-Offline" / "internal"


def has_real_requirements(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def install_requirements(requirements: Path, internal: Path) -> None:
    internal.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(internal),
        "-r",
        str(requirements),
    ]
    print("执行:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="安装 OpenAI 兼容接口额外依赖到 internal")
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS), help="依赖清单路径")
    parser.add_argument("--internal", default=str(DEFAULT_INTERNAL), help="便携版 internal 目录")
    args = parser.parse_args()

    requirements = Path(args.requirements).resolve()
    internal = Path(args.internal).resolve()

    if not requirements.exists():
        print(f"依赖清单不存在: {requirements}")
        return 1

    if not has_real_requirements(requirements):
        print("OpenAI 兼容接口当前没有额外 pip 依赖，无需安装。")
        print(f"如后续添加依赖，将安装到: {internal}")
        return 0

    install_requirements(requirements, internal)
    print(f"已安装到: {internal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
