import subprocess
import pandas as pd

def main():
    csv_file = 'test/all_sum_simple_v5.csv'
    df = pd.read_csv(csv_file, header=0, dtype=str, keep_default_na=False)
    results = []
    for idx, row in df.iterrows():
        if len(row) >= 15:
            val6 = str(row.iloc[5]) if pd.notna(row.iloc[5]) else ''
            val7 = str(row.iloc[6]) if pd.notna(row.iloc[6]) else ''
            val8 = str(row.iloc[7]) if pd.notna(row.iloc[7]) else ''
            # val15 = str(row.iloc[14]) if pd.notna(row.iloc[14]) else ''
            input_str = f"以下是一份系统运维工单的信息：\n集群：{val6}；\n摘要：{val7}；\n具体描述：{val8}。\n请处理"
            try:
                result = subprocess.run([
                    'python', '/home/tanxh/mas/agents/cliAI.py', input_str
                ], capture_output=True, text=True)
                output = result.stdout.strip()
            except Exception as e:
                output = f'Error: {e}'
            results.append(f'🎯 Input: {input_str}\n🤖 Output: {output}\n\n')
    with open('test_results.txt', 'w', encoding='utf-8') as out_f:
        out_f.writelines(results)

if __name__ == '__main__':
    main()
