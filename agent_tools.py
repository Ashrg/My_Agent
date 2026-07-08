import subprocess
from typing import List, Dict, Any, Optional
import py_compile


def read_file(file_path):
    """读取指定文件的全部内容
    
    Args:
        file_path (str): 要读取的文件的绝对路径或相对路径，支持各种文本文件格式 根据不同的操作系统注意路径写法
                        例如: "D:/example.txt", "/home/user/data.json", "config.ini"
    
    Returns:
        str: 文件的完整文本内容，保持原有的换行符和格式
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    

def write_file(file_path, content):
    """ 将指定内容写入文件，如果文件不存在则创建，如果存在则覆盖
    
    Args:
        file_path (str): 目标文件的绝对路径，支持创建新文件
        content (str): 要写入文件的文本内容，支持包含换行符的多行文本
    
    Returns:
        str: 成功时返回 "写入成功"，用于确认操作完成
    """    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
        return "写入完成"
    
def run_terminal_command(command, dangerous = True):
    """执行系统终端命令并返回详细的执行结果，如果指令不危险，请添加 "safe" 级别

    Args:
        command (str): 要执行的终端命令字符串，支持各种系统命令和参数
        dangerous (bool): 命令的安全级别，默认为 True，可选 False 表示安全命令，高危指令执行前会提示用户确认

    Returns:
        dict: 包含执行结果的字典，根据执行状态返回不同格式
    """
    
    if dangerous:
        confirm = input(f"提醒:即将执行'{command}, 请确认是否继续(y/n):").strip().lower()
        
        if confirm != "y":
            return {"status":"aborted", "message":"用户中止命令" }
    
    try:
        result = subprocess.run(command, shell= True, capture_output= True, text= True, check= True, encoding= 'utf-8', errors='replace',)
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "returncode": e.returncode, "error": e.stderr}
    except Exception as e:
        return {"status": "exception", "error": str(e)}

def check_python_file(file_path: str) -> Dict[str, Any]:
    """
    检查 Python 文件语法，并返回结构化错误信息。
    """
    try:
        py_compile.compile(file_path, doraise=True)

        return {
            "success": True,
            "message": "代码语法检查通过",
            "file": file_path,
        }

    except py_compile.PyCompileError as e:
        cause = e.exc_value

        if isinstance(cause, SyntaxError):
            return {
                "success": False,
                "error_type": type(cause).__name__,
                "message": cause.msg,
                "file": cause.filename,
                "line": cause.lineno,
                "offset": cause.offset,
                "code": cause.text.strip() if cause.text else None,
            }

        return {
            "success": False,
            "error_type": "PyCompileError",
            "message": str(e),
            "file": file_path,
        }

    except Exception as e:
        return {
            "success": False,
            "error_type": type(e).__name__,
            "message": str(e),
            "file": file_path,
        }