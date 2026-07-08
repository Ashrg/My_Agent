from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"

def _load_template(filename: str) -> str:
    """从 templates/ 目录加载模板文件。"""
    path = _TEMPLATES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"模板文件不存在: {path}")
    return path.read_text(encoding="utf-8")

# 保持原有变量名兼容
react_system_prompt_template = _load_template("system_prompt.md")