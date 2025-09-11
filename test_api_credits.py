#!/usr/bin/env python3
"""
测试Tushare API积分到期日期查询脚本
使用 uv 管理的现代 Python 项目
"""

import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import tushare as ts
    import pandas as pd
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: uv sync")
    sys.exit(1)

def test_api_credits(api_key, key_name):
    """测试单个API key的积分信息"""
    try:
        print(f"\n=== 测试 {key_name} ===")
        print(f"API Key: {api_key[:10]}...")
        
        pro = ts.pro_api(token=api_key)
        df = pro.user(token=api_key)
        
        if df.empty:
            print("❌ 未获取到积分信息")
            return False
            
        print("✅ API Key 有效")
        print("\n积分到期信息:")
        print(df.to_string(index=False))
        
        # 计算总积分
        total_credits = df['到期积分'].sum()
        print(f"\n总积分: {total_credits:,.4f}")
        
        # 找到最近的到期日期
        df['到期时间'] = pd.to_datetime(df['到期时间'])
        nearest_expiry = df['到期时间'].min()
        print(f"最近到期日期: {nearest_expiry.strftime('%Y-%m-%d')}")
        
        # 计算距离到期的天数
        days_until_expiry = (nearest_expiry - datetime.now()).days
        if days_until_expiry > 0:
            print(f"距离到期: {days_until_expiry} 天")
        else:
            print(f"⚠️ 已过期 {abs(days_until_expiry)} 天")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("Tushare API 积分到期日期测试工具")
    print("=" * 50)
    
    # 加载环境变量
    load_dotenv()
    
    api_key_1 = os.getenv('TUSHARE_API_KEY')
    api_key_2 = os.getenv('TUSHARE_API_KEY_2')
    
    if not api_key_1:
        print("❌ 未找到 TUSHARE_API_KEY 环境变量")
        return
    
    if not api_key_2:
        print("❌ 未找到 TUSHARE_API_KEY_2 环境变量")
        return
    
    # 测试两个API key
    success_count = 0
    
    if test_api_credits(api_key_1, "TUSHARE_API_KEY"):
        success_count += 1
        
    if test_api_credits(api_key_2, "TUSHARE_API_KEY_2"):
        success_count += 1
    
    print(f"\n" + "=" * 50)
    print(f"测试完成: {success_count}/2 个API Key有效")

if __name__ == "__main__":
    main()