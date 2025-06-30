import os
import re
from PyPDF2 import PdfReader
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """
    提取 PDF 中文本。
    :param pdf_path: PDF 文件路径
    :return: 合并后的原始文本字符串
    """
    try:
        reader = PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return " ".join(texts)
    except Exception as e:
        print(f"[错误] 读取 {pdf_path} 时出错：{e}")
        return ""

def categorize_pdfs(root_dir):
    """
    遍历 root_dir 下所有 PDF，将它们分类为：
      1. 仅包含 “EEG”（大写、独立三字母单词）的 PDF
      2. 同时包含 “EEG” 和 “Emotion”（不区分大小写、作为独立单词）的 PDF
    :param root_dir: 根目录
    :return: (eeg_only_list, eeg_emotion_list)
    """
    # 收集所有 PDF 文件路径
    pdf_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(dirpath, fname))

    eeg_only = []
    eeg_emotion = []
    eeg_pattern = re.compile(r'\bEEG\b')
    emotion_pattern = re.compile(r'\bemotion\b', re.IGNORECASE)

    for path in tqdm(pdf_paths, desc="Processing PDFs", unit="file"):
        raw_text = extract_text_from_pdf(path)
        if not raw_text:
            continue

        has_eeg = bool(eeg_pattern.search(raw_text))
        has_emotion = bool(emotion_pattern.search(raw_text))

        if has_eeg:
            if has_emotion:
                eeg_emotion.append(path)
            else:
                eeg_only.append(path)

    return eeg_only, eeg_emotion

if __name__ == "__main__":
    # 将此路径替换为你的 PDF 根目录
    root_folder = r"D:\dachuang\ID+论文名\id+名\id+名"

    eeg_only_files, eeg_emotion_files = categorize_pdfs(root_folder)

    print("\n========== 仅包含 ‘EEG’ 的 PDF ==========")
    if eeg_only_files:
        for f in eeg_only_files:
            print(f)
    else:
        print("未找到仅包含 ‘EEG’ 的 PDF。")

    print("\n===== 同时包含 ‘EEG’ 和 ‘Emotion’ 的 PDF =====")
    if eeg_emotion_files:
        for f in eeg_emotion_files:
            print(f)
    else:
        print("未找到同时包含 ‘EEG’ 和 ‘Emotion’ 的 PDF。")
