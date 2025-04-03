# ----------------------- 配置区域 -----------------------
API_ENDPOINT = "http://100.64.1.1:8000/v1/chat/completions"
PDF_INPUT_DIR = "InputPDF"
OUTPUT_DIR = "Results"
CONTEXT_WINDOW = 12800 # 上下文窗口大小（字符数）
REQUEST_TIMEOUT = 120  # 请求超时时间（秒）
TARGET_SUMMARY_LENGTH = 5000         # 期望生成的最终综述的目标长度（以字符计）
INITIAL_MAX_TOKENS = TARGET_SUMMARY_LENGTH*0.25  # 初始的 MAX_TOKENS 值
DYNAMIC_ADJUSTMENT_INTENSITY = 0.08   # 动态调整 token 的强度 (需要根据实验调整)
DYNAMIC_MAX_TOKENS_LOWER_BOUND = 500   # 动态计算出的 MAX_TOKENS 的最小值
MAX_TOKENS = TARGET_SUMMARY_LENGTH*1.2                  # API 响应中期望返回的最大 token 数 (上限)
# ------------------------------------------------------

import json
import re
from pathlib import Path
from datetime import datetime
from PyPDF2 import PdfReader
import requests
import nltk
from nltk.tokenize import sent_tokenize

# 确保 NLTK punkt tokenizer 已下载，天知道这玩意为什么还有外置库
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloaderError:
    nltk.download('punkt')

class IterativeSummarizer:
    """
    迭代式文本摘要生成器，动态提取文本，使用 NLTK 进行句子分割，并显示处理进度。
    """
    FINAL_SUMMARY_PROMPT = """你是一位资深研究员，请根据以下提供的综述，生成一个高度概括性的总结，突出最重要的发现、结论和创新点。请使用简洁明了的语言，并尽量控制在一定的字数内（例如，不超过 500 字）。"""

    def __init__(self, target_summary_length):
        """
        初始化会话，设置请求头。
        """
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.processed_length = 0  # 记录已处理的文本长度
        self.full_text = ""
        self.TARGET_SUMMARY_LENGTH = target_summary_length
        self.total_summary_characters = 0
        self.total_summary_tokens = 0
        self.dynamic_chars_per_token = 3.0  # 初始估计值
        self.integral_error = 0
        self.previous_error = 0

    def _get_next_chunk(self, text: str, max_length: int) -> str:
        """
        从文本中动态提取不超过最大长度的下一个文本块，基于句子。

        Args:
            text: 原始文本。
            max_length: 允许提取的最大长度。

        Returns:
            提取的下一个文本块。
        """
        start_index = self.processed_length
        remaining_text = text[start_index:]
        if not remaining_text:
            return ""

        sentences = sent_tokenize(remaining_text)
        extracted_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 

            if current_length + sentence_length <= max_length:
                if extracted_chunk:
                    extracted_chunk += " " + sentence
                else:
                    extracted_chunk = sentence
                current_length += sentence_length
            else:
                break  # 句子超出最大长度

        return extracted_chunk

    def _api_call(self, context: str, current_chunk: str, previous_summary_length: int, max_tokens: int) -> str:
        """
        调用 API 接口进行文本摘要，并在 Prompt 中引导模型控制综述长度和处理进度。
        """
        remaining_length = len(self.full_text) - self.processed_length
        total_length = len(self.full_text)

        system_prompt_content = f"""你是一位专业的科研助理，任务是根据提供的先前摘要和当前内容，生成一份整合的科研综述。这是一个迭代式的过程。

**目标：** 生成的综述的长度尽量接近 {self.TARGET_SUMMARY_LENGTH} 字符。

**上次综述长度：** {previous_summary_length} 字符。

**处理进度：** 当前处理到文档的 {self.processed_length} 字符，剩余 {remaining_length} 字符，总共 {total_length} 字符。

请将当前内容的关键信息整合到先前摘要中, 形成一个不断完善和扩展的综述。请注意以下要求：

1.  **格式:** 使用 Markdown 格式。
2.  **关键细节:** 保留所有关键的技术细节和实验数据。
3.  **创新点标注:** 在创新点之后用 `[创新]` 标注。
4.  **方法对比:** 明确指出不同方法之间的关键差异。
5.  **整合:** 避免重复，保持逻辑连贯。如果当前内容与先前摘要冲突，以当前内容为准。

**请根据以上目标和要求生成新的综述。**
"""

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt_content
                },
                {
                    "role": "user",
                    "content": f"先前摘要：{context}\n\n当前内容：{current_chunk}"
                }
            ],
            "max_tokens": max_tokens
        }

        try:
            response = self.session.post(
                API_ENDPOINT,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            response_json = response.json()
            summary = response_json['choices'][0]['message']['content']
            if 'usage' in response_json and 'total_tokens' in response_json['usage']:
                tokens_used = response_json['usage']['total_tokens']
                self.total_summary_tokens += tokens_used
                self.total_summary_characters = len(context) + len(summary) # 近似计算
                if self.total_summary_tokens > 0:
                    self.dynamic_chars_per_token = self.total_summary_characters / self.total_summary_tokens
            return summary
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return context

    def process(self, text: str) -> str:
        """
        迭代处理文本，使用动态 token/字符比例和 PID 控制（简化为比例控制）调整 MAX_TOKENS。
        """
        self.full_text = text
        accumulated_summary = ""
        self.processed_length = 0
        total_length = len(text)

        while self.processed_length < total_length:
            available_space = CONTEXT_WINDOW - len(accumulated_summary) - 200
            next_chunk = self._get_next_chunk(text, available_space)

            if not next_chunk:
                break

            chunk_length = len(next_chunk)
            print(f"处理文本块 (长度: {chunk_length})，已处理: {self.processed_length}/{total_length}", end=" ")

            target_tokens = TARGET_SUMMARY_LENGTH / self.dynamic_chars_per_token if TARGET_SUMMARY_LENGTH > 0 and self.dynamic_chars_per_token > 0 else MAX_TOKENS
            current_summary_tokens = len(accumulated_summary) // self.dynamic_chars_per_token if self.dynamic_chars_per_token > 0 else 0

            # 简单的比例控制,PID大成功
            error = target_tokens - current_summary_tokens
            adjustment = error * DYNAMIC_ADJUSTMENT_INTENSITY
            dynamic_max_tokens = int(INITIAL_MAX_TOKENS + adjustment)

            dynamic_max_tokens = min(MAX_TOKENS, max(DYNAMIC_MAX_TOKENS_LOWER_BOUND, dynamic_max_tokens))

            progress = (self.processed_length / total_length) * 100
            bar_length = 40
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"[{bar}] {progress:.2f}%")

            accumulated_summary = self._api_call(accumulated_summary, next_chunk, len(accumulated_summary), dynamic_max_tokens)
            self.processed_length += chunk_length

            print(f"当前摘要长度: {len(accumulated_summary)} 字符 (预计 token: {len(accumulated_summary) // self.dynamic_chars_per_token:.2f}), 动态 MAX_TOKENS: {dynamic_max_tokens:.2f}, 动态字符/Token比例: {self.dynamic_chars_per_token:.2f}\n{'-'*40}")

        return accumulated_summary

class PDFProcessor:
    """
    用于加载和合并 PDF 文件的文本内容。
    """
    @staticmethod
    def load_and_merge() -> str:
        """
        加载指定目录下所有 PDF 文件的文本内容，并将它们合并成一个字符串。

        Returns:
            合并后的 PDF 文本。
        """
        merged_text = []
        pdf_dir = Path(PDF_INPUT_DIR)
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                with pdf_file.open("rb") as f:
                    text = "\n".join([
                        p.extract_text() or ""
                        for p in PdfReader(f).pages
                    ])
                    merged_text.append(f"[文件：{pdf_file.name}]\n{text}")
                    print(f"已加载: {pdf_file.name}")
            except Exception as e:
                print(f"文件错误 {pdf_file.name}: {e}")
        return "\n\n".join(merged_text)

def main():
    """
    主程序入口。
    """
    # 输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # 处理流程
    print("\n" + "="*40)
    print("迭代式论文分析系统")
    print("="*40)

    # 1. 加载PDF
    print("\n📂 正在加载PDF文件...")
    full_text = PDFProcessor.load_and_merge()
    print(f"总文本长度: {len(full_text)} 字符")

    # 2. 生成摘要
    print("\n🔨 开始迭代处理...")
    summarizer = IterativeSummarizer(TARGET_SUMMARY_LENGTH)
    final_summary = summarizer.process(full_text)

    # 3. 保存结果到 Markdown 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"综述报告_{timestamp}.md"

    report = f"# 迭代式论文综述\n\n" \
             f"**生成时间**: {datetime.now()}\n" \
             f"**处理文件**: {len(list(Path(PDF_INPUT_DIR).glob('*.pdf')))} 篇\n\n" \
             f"{final_summary}"

    output_path.write_text(report, encoding="utf-8")
    print(f"\n✅ 报告已保存至: {output_path}")

if __name__ == "__main__":
    main()