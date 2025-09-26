#!/usr/bin/env python3
"""
Tushare API credits expiration tester (single key)
Managed with uv
"""

import os
import sys
from datetime import datetime

try:
    import pandas as pd
    import tushare as ts
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: uv sync")
    sys.exit(1)

def test_api_credits(api_key: str, source_name: str) -> bool:
    try:
        print(f"\n=== 测试 {source_name} ===")
        print(f"Token: {api_key[:10]}...")

        pro = ts.pro_api(token=api_key)
        df = pro.user(token=api_key)

        if df.empty:
            print("❌ 未获取到积分信息")
            return False

        print("✅ Token 有效")
        print("\n积分到期信息:")
        print(df.to_string(index=False))

        total_credits = df["到期积分"].sum()
        print(f"\n总积分: {total_credits:,.4f}")

        df["到期时间"] = pd.to_datetime(df["到期时间"])
        nearest_expiry = df["到期时间"].min()
        print(f"最近到期日期: {nearest_expiry.strftime('%Y-%m-%d')}")

        days_until_expiry = (nearest_expiry - datetime.now()).days
        if days_until_expiry > 0:
            print(f"距离到期: {days_until_expiry} 天")
        else:
            print(f"⚠️ 已过期 {abs(days_until_expiry)} 天")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def main() -> None:
    print("Tushare API 积分到期日期测试工具")
    print("=" * 50)

    load_dotenv()

    api_key = os.getenv("TUSHARE_TOKEN")
    source_name = "TUSHARE_TOKEN"
    if not api_key:
        api_key = os.getenv("TUSHARE_API_KEY")
        source_name = "TUSHARE_API_KEY"

    if not api_key:
        print("❌ 未找到环境变量：TUSHARE_TOKEN 或 TUSHARE_API_KEY")
        return

    success = test_api_credits(api_key, source_name)
    print("\n" + "=" * 50)
    print(f"测试完成: {'成功' if success else '失败'}")


if __name__ == "__main__":
    main()
