"""Paired answer-trial helpers; legacy refinement helpers retained for old experiments."""
import re
import time
from langchain_core.messages import HumanMessage, SystemMessage


REFINEMENT_INSTRUCTION = (
    'Bạn chỉ viết lại câu hỏi pháp lý bằng tiếng Việt để tìm kiếm tài liệu. '
    'Giữ nguyên ý định và các dữ kiện của câu hỏi gốc; chỉ cải thiện cách diễn đạt và từ khóa. '
    'Không trả lời câu hỏi. Không bổ sung hoặc đề xuất điều, khoản, số hiệu hoặc văn bản pháp luật '
    'làm tài liệu tham khảo. Chỉ giữ những dẫn chiếu đã có trong câu hỏi gốc. '
    'Chỉ xuất một câu hỏi được viết lại, không giải thích, không phân tích.'
)


def legal_references(text):
    return {m.group().casefold() for m in re.finditer(
        r'(?:điều|khoản)\s+\d+[a-zđ]?|\d+/\d{4}/[\wđ-]+', text, flags=re.IGNORECASE)}


def make_trial_llm(provider, model, *, base_url='http://localhost:11434'):
    if provider == 'ollama':
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url, temperature=0.1,
                          reasoning=False, num_ctx=8192, num_predict=2048, keep_alive=-1)
    raise ValueError('Unknown trial provider')


def invoke_trial(llm, messages, *, max_attempts=5):
    """Honor provider cooldowns; keep transport retries distinct from logical calls."""
    started = time.perf_counter()
    retries = 0
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(messages)
            if not isinstance(response.content, str) or not response.content.strip():
                raise ValueError('Model returned no plain-text answer')
            return {'text': response.content.strip(), 'seconds': time.perf_counter() - started,
                    'usage': response.usage_metadata or {}, 'rate_limit_retries': retries,
                    'finish_reason': response.response_metadata.get('finish_reason') or response.response_metadata.get('done_reason')}
        except Exception as error:
            if getattr(error, 'status_code', None) != 429 or attempt + 1 == max_attempts:
                raise
            headers = getattr(getattr(error, 'response', None), 'headers', {})
            try:
                wait = max(60.0, float(headers.get('retry-after', 60)))
            except (TypeError, ValueError):
                wait = 60.0
            print(f'Provider rate limit; waiting {wait:.0f}s before retry.', flush=True)
            time.sleep(wait)
            retries += 1


def refine_once(llm, question):
    result = invoke_trial(llm, [SystemMessage(content=REFINEMENT_INSTRUCTION), HumanMessage(content=question)])
    candidate = result['text']
    added = sorted(legal_references(candidate) - legal_references(question))
    result.update({'original_question': question, 'candidate_query': candidate,
                   'refined_question': question if added else candidate,
                   'accepted': not added, 'introduced_references': added})
    return result
