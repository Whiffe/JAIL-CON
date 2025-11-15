#!/usr/bin/env python3
"""
从CSV文件中随机选择指定行数的数据
python random_select_excel.py --excel ./harmful_behaviors_Adv.csv

https://claude.ai/share/02d168b9-60a3-42ed-b93c-c0ff565cbf21
"""
import pandas as pd
import argparse
import sys


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='从CSV文件中随机选择200行数据')
    parser.add_argument('--excel', type=str, required=True, 
                        help='输入CSV文件的路径，例如: ./harmful_behaviors_Adv.csv')
    
    args = parser.parse_args()
    
    try:
        # 读取CSV文件
        print(f"正在读取文件: {args.excel}")
        df = pd.read_csv(args.excel)
        
        # 检查数据行数
        total_rows = len(df)
        print(f"文件总行数(不含表头): {total_rows}")
        
        if total_rows < 200:
            print(f"警告: 文件只有 {total_rows} 行数据，少于200行")
            print("将选择所有数据行")
            selected_df = df
        else:
            # 从第二行开始随机选择200行(索引从0开始，所以是df的所有行)
            selected_df = df.sample(n=200, random_state=None)
            print(f"已随机选择 200 行数据")
        
        # 生成输出文件名
        output_path = './harmful_behaviors_Adv_random200.csv'
        
        # 保存到新文件
        selected_df.to_csv(output_path, index=False)
        print(f"新文件已保存到: {output_path}")
        print("完成!")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{args.excel}'")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
