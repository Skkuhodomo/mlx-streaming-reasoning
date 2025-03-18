import re
import time

import mlx.core as mx
import streamlit as st
from mlx_lm.utils import load, generate_step

title = "MLX Chat"
ver = "0.8"
debug = False

# 모델 정의
MODELS = {
    "reasoning": "mlx-community/QwQ-32B-4bit",
    "chat": "mlx-community/EXAONE-3.5-2.4B-Instruct-4bit"
}

# 시스템 프롬프트 정의
SYSTEM_PROMPTS = {
    "reasoning": """You are a helpful AI assistant that shows your reasoning process.
When you need to think about something, wrap your thoughts in <think> tags.
For example:
<think>
1. First, I need to understand what the user is asking
2. Then, I should consider the key factors
3. Finally, I can provide a clear answer
</think>
Your final answer should be concise and clear.""",
    "chat": "You are a helpful AI assistant trained on a vast amount of human knowledge. Answer as concisely as possible."
}

def extract_think_content(text):
    """<think> 태그 내의 내용을 추출합니다."""
    pattern = r'<think>(.*?)</think>'
    matches = re.finditer(pattern, text, re.DOTALL)
    results = [match.group(1).strip() for match in matches]
    return results

def remove_think_tags(text):
    """<think> 태그를 제거합니다."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def generate(the_prompt, the_model, the_tokenizer):
    tokens = []
    skip = 0
    # context 길이 제한 없이 generate_step에서 반환하는 모든 토큰 처리
    for (token, prob) in generate_step(mx.array(the_tokenizer.encode(the_prompt)), the_model, max_tokens=-1):
        if token == the_tokenizer.eos_token_id:
            break

        tokens.append(token)
        text = the_tokenizer.decode(tokens)
        trim = None
        for sw in STOP_TOKENS:
            if text[-len(sw):].lower() == sw:
                # 종료어 발견 시 생성 중단
                return
            else:
                for i, _ in enumerate(sw, start=1):
                    if text[-i:].lower() == sw[:i]:
                        trim = -i
        yield text[skip:trim]
        skip = len(text)

def show_chat(the_prompt, previous=""):
    # 메시지 생성 시점의 모델 종류는 st.session_state["model_type"]에서 가져옴.
    current_model_type = st.session_state["model_type"]
    active_reasoning_mode = (current_model_type == "reasoning")
    st.session_state["is_generating"] = True

    if debug:
        print(the_prompt)
        print("-" * 80)

    current_model, current_tokenizer = get_model(current_model_type)

    with st.chat_message("assistant"):
        # UI 순서: 1) 사고 과정(expander, reasoning 모드일 경우), 2) 최종 답변
        thinking_expander = None
        thinking_placeholder = None
        if active_reasoning_mode:
            thinking_expander = st.expander("🤔 AI의 사고 과정", expanded=False)
            thinking_placeholder = thinking_expander.empty()
        final_answer_placeholder = st.empty()

        response = previous
        thinking_text = ""
        final_text = ""
        in_thinking_mode = active_reasoning_mode
        has_thinking = False

        # 업데이트 최적화 변수들
        last_update_time = time.time()
        update_interval = 0.1
        min_buffer_size = 150
        thinking_buffer = ""
        last_thinking_update = ""

        # 문단 경계 패턴
        paragraph_markers = ["\n\n", "\n", ".", "!", "?", ":", ";"]

        chunks_accumulated = []
        for chunk in generate(the_prompt, current_model, current_tokenizer):
            chunks_accumulated.append(chunk)
            response += chunk

            if not previous:
                response = re.sub(r"^/\*+/", "", response)
                response = re.sub(r"^:+", "", response)
            response = response.replace('', '')
            current_time = time.time()

            if active_reasoning_mode:
                if "</think>" in chunk and in_thinking_mode:
                    in_thinking_mode = False
                    end_index = chunk.find("</think>")
                    if end_index > 0:
                        thinking_text += chunk[:end_index]
                        thinking_buffer += chunk[:end_index]
                    end_tag_end = end_index + len("</think>")
                    if end_tag_end < len(chunk):
                        final_text += chunk[end_tag_end:]
                    if thinking_text.strip() and thinking_text != last_thinking_update:
                        has_thinking = True
                        if thinking_placeholder:
                            thinking_placeholder.markdown(thinking_text)
                        last_thinking_update = thinking_text
                        thinking_buffer = ""
                        last_update_time = current_time

                elif "<think>" in chunk and not in_thinking_mode:
                    in_thinking_mode = True
                    start_index = chunk.find("<think>")
                    if start_index > 0:
                        final_text += chunk[:start_index]
                    if start_index + len("<think>") < len(chunk):
                        new_content = chunk[start_index + len("<think>"):]
                        thinking_text += new_content
                        thinking_buffer += new_content
                        has_thinking = True

                else:
                    if in_thinking_mode:
                        thinking_text += chunk
                        thinking_buffer += chunk
                        has_thinking = True
                        time_based_update = current_time - last_update_time > update_interval
                        boundary_based_update = any(marker in thinking_buffer for marker in paragraph_markers)
                        size_based_update = len(thinking_buffer) > min_buffer_size
                        should_update = time_based_update or (boundary_based_update and size_based_update)
                        if should_update and thinking_text != last_thinking_update:
                            if thinking_placeholder:
                                thinking_placeholder.markdown(thinking_text)
                            last_thinking_update = thinking_text
                            thinking_buffer = ""
                            last_update_time = current_time
                    else:
                        final_text += chunk
                final_answer_placeholder.markdown(final_text + "▌")
            else:
                final_text += chunk
                final_answer_placeholder.markdown(final_text + "▌")

        if active_reasoning_mode:
            if has_thinking and thinking_text != last_thinking_update:
                if thinking_placeholder:
                    thinking_placeholder.markdown(thinking_text)
            elif not has_thinking and thinking_expander:
                thinking_expander.empty()
            clean_response = final_text.strip()
            final_answer_placeholder.markdown(clean_response)
            response_to_save = {
                "content": clean_response, 
                "thinking": thinking_text, 
                "reasoning_mode": True,
                "model_type": current_model_type
            }
        else:
            final_answer_placeholder.markdown(final_text)
            response_to_save = {
                "content": final_text, 
                "reasoning_mode": False,
                "model_type": current_model_type
            }

    st.session_state["is_generating"] = False

    if previous:
        st.session_state.messages[-1] = {"role": "assistant", **response_to_save}
    else:
        st.session_state.messages.append({"role": "assistant", **response_to_save})

def remove_last_occurrence(array, criteria_fn):
    for i in reversed(range(len(array))):
        if criteria_fn(array[i]):
            del array[i]
            break

def build_memory():
    if len(st.session_state.messages) > 2:
        return st.session_state.messages[1:-1]
    return []

def queue_chat(the_prompt, continuation=""):
    st.session_state["prompt"] = the_prompt
    st.session_state["continuation"] = continuation
    st.rerun()

def get_model(model_type):
    model_ref = MODELS[model_type]
    return load_model_and_cache(model_ref)

def get_chat_template(tokenizer_obj):
    return tokenizer_obj.chat_template or (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

def has_system_role_support(chat_template_str):
    return "system role not supported" not in chat_template_str.lower()

# 종료 토큰
STOP_TOKENS = ["<|im_start|>", "<|im_end|>", "<s>", "</s>"]

assistant_greeting = "안녕하세요! 무엇을 도와드릴까요?"

st.set_page_config(
    page_title=title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# (스타일 코드는 생략 가능)
st.markdown("""
<style>
/* 스타일 코드 (생략) */
</style>
""", unsafe_allow_html=True)

st.title(title)
st.markdown(r"<style>.stDeployButton{display:none}</style>", unsafe_allow_html=True)

# 세션 상태 초기화
if "model_type" not in st.session_state:
    st.session_state["model_type"] = "chat"  # 실제 적용 모델 (send 시 반영)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": assistant_greeting}]

@st.cache_resource(show_spinner=True)
def load_model_and_cache(ref):
    return load(ref, {"trust_remote_code": True})

# 초기 모델 로드
initial_model, initial_tokenizer = load_model_and_cache(MODELS[st.session_state["model_type"]])
model = initial_model
tokenizer = initial_tokenizer

current_system_prompt = SYSTEM_PROMPTS[st.session_state["model_type"]]
initial_chat_template = get_chat_template(tokenizer)
supports_system_role = has_system_role_support(initial_chat_template)
system_prompt = st.sidebar.text_area(
    "System 프롬프트",
    current_system_prompt,
    disabled=not supports_system_role or st.session_state.get("is_generating", False)
)

st.sidebar.markdown("---")
sidebar_actions = st.sidebar.columns(2)

time.sleep(0.05)

if sidebar_actions[0].button("🗑️ 대화 지우기", use_container_width=True, help="이전 대화 내용을 모두 지웁니다.", disabled=st.session_state.get("is_generating", False)):
    st.session_state.messages = [{"role": "assistant", "content": assistant_greeting}]
    st.session_state["is_generating"] = False
    st.rerun()

if sidebar_actions[1].button("🔄 계속하기", use_container_width=True, help="마지막 응답을 이어서 계속 생성합니다.", disabled=st.session_state.get("is_generating", False)):
    user_prompts = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    if user_prompts:
        last_user_prompt = user_prompts[-1]
        assistant_responses = [msg for msg in st.session_state.messages if msg["role"] == "assistant" and msg["content"] != assistant_greeting]
        if assistant_responses:
            last_response = assistant_responses[-1]
            last_assistant_response = last_response["content"]
            if "model_type" in last_response:
                st.session_state["model_type"] = last_response["model_type"]
            last_assistant_response_lines = last_assistant_response.split('\n')
            if len(last_assistant_response_lines) > 1:
                last_assistant_response_lines.pop()
                last_assistant_response = "\n".join(last_assistant_response_lines)
            messages = [
                {"role": "user", "content": last_user_prompt},
                {"role": "assistant", "content": last_assistant_response},
            ]
            if supports_system_role:
                messages.insert(0, {"role": "system", "content": system_prompt})
            full_prompt = tokenizer.apply_chat_template(messages,
                                                        tokenize=False,
                                                        add_generation_prompt=False,
                                                        chat_template=initial_chat_template)
            full_prompt = full_prompt.rstrip("\n")
            remove_last_occurrence(st.session_state.messages,
                                   lambda msg: msg["role"] == "assistant" and msg["content"] != assistant_greeting)
            queue_chat(full_prompt, last_assistant_response)

# 페이지 구성: 채팅 영역과 입력 영역
main_container = st.container()
messages_container = main_container.container()
input_section = main_container.container()
input_section.markdown('<div class="chat-input-area"></div>', unsafe_allow_html=True)

with messages_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("reasoning_mode") and msg.get("thinking"):
                with st.expander("🤔 AI의 사고 과정", expanded=False):
                    st.markdown(msg["thinking"])
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])
    if "prompt" in st.session_state and st.session_state["prompt"]:
        show_chat(st.session_state["prompt"], st.session_state["continuation"])
        st.session_state["prompt"] = None
        st.session_state["continuation"] = None

with input_section:
    # 모델 선택 UI와 채팅 입력을 같은 행에 배치 (모델 선택은 send 전까지만 의미 있음)
    cols = st.columns([2, 8])
    with cols[0]:
        # 모델 선택은 채팅 입력 옆에 표시되며, send 시에만 반영됩니다.
        model_options = {"chat": "일반", "reasoning": "사고 과정"}
        pending_model = st.selectbox(
            "모델 선택",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0 if st.session_state["model_type"] == "chat" else 1,
            key="pending_model_selectbox"
        )
    with cols[1]:
        input_container = st.container()
        if not st.session_state.get("is_generating", False):  
                prompt = st.chat_input("메시지를 입력하세요...")
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    # send 시, pending 모델 값을 실제 모델로 반영
                    st.session_state["model_type"] = pending_model
                    message_model, message_tokenizer = get_model(st.session_state["model_type"])
                    message_chat_template = get_chat_template(message_tokenizer)
                    message_supports_system = has_system_role_support(message_chat_template)
                    current_sys_prompt = SYSTEM_PROMPTS[st.session_state["model_type"]]
                    used_system_prompt = system_prompt if system_prompt.strip() and system_prompt != current_sys_prompt else current_sys_prompt
                    messages = []
                    if message_supports_system:
                        messages += [{"role": "system", "content": used_system_prompt}]
                    messages += build_memory()
                    messages += [{"role": "user", "content": prompt}]
                    full_prompt = message_tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        chat_template=message_chat_template
                    )
                    full_prompt = full_prompt.rstrip("\n")
                    queue_chat(full_prompt)
        else:
                st.markdown('<div class="disabled-input-container"><span style="color:#868991;">메시지를 입력하세요... (응답 생성 중)</span></div>', unsafe_allow_html=True)
                prompt = None

st.sidebar.markdown("---")
st.sidebar.markdown(f"v{ver} / st {st.__version__}")