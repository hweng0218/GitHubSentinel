# src/llm.py

import os
import time
import random
from openai import OpenAI
from logger import LOG


class LLM:
    def __init__(self):
        self.client = OpenAI()
        LOG.add("daily_progress/llm_logs.log", rotation="1 MB", level="DEBUG")

    def generate_daily_report(self, markdown_content, dry_run=False):
        # Validate input
        if not markdown_content:
            raise ValueError("Markdown content cannot be empty.")

        # Construct the prompt
        prompt = (
            "请根据以下内容生成一份详细的项目进展报告。报告应包括以下部分：\n"
            "1. **新增功能**：详细列出最近添加的功能，包括功能的名称、目的、影响及实施细节。\n"
            "2. **主要改进**：详细描述对现有功能的改进，包括改进的具体内容、目的和对项目的影响。\n"
            "3. **修复问题**：详细列出修复的主要问题，包括问题的描述、影响及修复措施。\n\n"
            "请确保：\n"
            "- 每个部分的总结应控制在 100-150 字之间。\n"
            "- 使用 Markdown 格式，以便更好地进行报告排版。\n\n"
            "以下是需要总结的最新进展内容：\n"
            f"{markdown_content}"
        )

        if dry_run:
            LOG.info("Dry run mode enabled. Saving prompt to file.")
            with open("daily_progress/prompt.txt", "w+") as f:
                f.write(prompt)
            LOG.debug("Prompt saved to daily_progress/prompt.txt")
            return "DRY RUN"

        LOG.info("Starting report generation using GPT model.")

        for attempt in range(3):  # Retry mechanism
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Invalid response from GPT model.")

                LOG.debug("GPT response: {}", response)
                return response.choices[0].message.content
            except ValueError as ve:
                LOG.error("Value error on attempt {}: {}", attempt + 1, ve)
                # No sleep, as this is likely a consistent issue
                break
            except Exception as e:
                LOG.error("Attempt {}: An error occurred while generating the report: {}", attempt + 1, e)
                time.sleep(random.uniform(1, 3))  # Random delay for retry
        raise RuntimeError("Failed to generate report after multiple attempts.")
