#!/usr/bin/env python
"""
一键运行单元测试 - 直接运行！

在 VS Code 里：
1. 打开这个文件
2. 点击右上角的运行按钮 ▶️
3. 查看测试结果

测试新算法的核心功能是否正常工作
"""

import sys
from pathlib import Path

# 添加 src 到路径
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_multi_bfs_functions import run_all_tests


def main():
    print("\n" + "="*70)
    print("🧪 Multi-source BFS 单元测试")
    print("="*70)
    print("\n这会测试新算法的三个核心函数:")
    print("  1. 空间分布种子选择")
    print("  2. 回溯平衡算法")
    print("  3. 完整的 Multi-source BFS")
    print()

    success = run_all_tests()

    if success:
        print("\n✅ 所有测试通过！新算法工作正常。\n")
    else:
        print("\n❌ 有测试失败，请检查错误信息。\n")


if __name__ == "__main__":
    main()
