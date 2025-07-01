import re
def remove_lines(input_str: str) -> str:
    # 将输入字符串按行分割
    lines = [line for line in input_str.splitlines() if line.strip() != ""]
    # 找到目标子字符串的位置
    target_index = -1
    for i, line in enumerate(lines):
        if '<EOS>' in line:
            target_index = i+1
            if target_index >= len(lines):
                return None,None
            match = re.match(r'(<Tab>)+', lines[target_index])
            if match:
                level = match.group(0)
            else:
                level = ''
            break

    # 如果没有找到目标子字符串，直接返回原始字符串
    if target_index == -1:
        return None,None
    # 逐行向前检索
    for i in range(target_index - 1, -1, -1):
        # 如果行以 "<Tab><Tab>" 开头，则删除该行
        if lines[i].startswith(level):
            del lines[i]
        else:
            # 遇到不以 "<Tab><Tab>" 开头的行时停止检索
            break

    return "\n".join(lines), "\n".join(lines).split('<EOS>')[0]+'<EOS>'