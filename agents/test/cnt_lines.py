import os

def count_lines(directory):
    total_lines = 0
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_count += 1
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    
                    for line in lines:
                        line = line.strip()
                        if line == '':
                            blank_lines += 1
                        elif line.startswith('#'):
                            comment_lines += 1
                        else:
                            code_lines += 1
    
    print(f"\n📊 Python 代码统计报告")
    print("=" * 50)
    print(f"📁 Python 文件总数：{file_count}")
    print(f"📝 总行数：{total_lines}")
    print(f"💻 代码行数：{code_lines}")
    print(f"📢 注释行数：{comment_lines}")
    print(f"⚪ 空行数：{blank_lines}")
    print("=" * 50)

if __name__ == '__main__':
    count_lines('/home/tanxh/mas/agents')