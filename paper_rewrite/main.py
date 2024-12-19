import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent, polish_paper

st.title("📑 AI智能PDF问答+论文润色工具")

# 添加功能选择
mode = st.sidebar.radio("选择功能：", ["PDF问答", "论文润色"])

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
    base_url = st.text_input("请输入Base URL：", value="https://api.openai.com/v1")
    
    model_name = st.selectbox(
        "选择模型：",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o-mini", "gpt-4o"],
        index=0
    )
    
    # 修复温度控制滑块
    temperature = st.slider(
        "回答创造性（温度）：",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

if mode == "PDF问答":
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )

    uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")
    
    # 添加预设问题模板
    preset_questions = [
        "总结这篇文章的主要内容",
        "这篇文章的创新点是什么？",
        "这篇文章的研究方法是什么？",
        "这篇文章的实验结果如何？",
        "自定义问题"
    ]
    
    question_type = st.selectbox("选择问题类型：", preset_questions)
    if question_type == "自定义问题":
        question = st.text_input("输入你的问题：", disabled=not uploaded_file)
    else:
        question = question_type

    if uploaded_file and question and not openai_api_key:
        st.info("请输入你的OpenAI API密钥")

    if uploaded_file and question and openai_api_key:
        with st.spinner("AI正在思考中，请稍等..."):
            response = qa_agent(openai_api_key, base_url, st.session_state["memory"],
                              uploaded_file, question, model_name, temperature)
        st.write("### 答案")
        st.write(response["answer"])
        st.session_state["chat_history"] = response["chat_history"]

        # 添加答案操作按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("复制答案"):
                st.write("答案已复制到剪贴板！")
                st.session_state['clipboard'] = response["answer"]
        with col2:
            if st.button("导出聊天记录"):
                st.download_button(
                    label="下载聊天记录",
                    data=str(st.session_state["chat_history"]),
                    file_name="chat_history.txt"
                )

    if "chat_history" in st.session_state:
        with st.expander("历史消息", expanded=False):
            for i in range(0, len(st.session_state["chat_history"]), 2):
                with st.chat_message("user"):
                    st.write(st.session_state["chat_history"][i].content)
                with st.chat_message("assistant"):
                    st.write(st.session_state["chat_history"][i+1].content)
                if i < len(st.session_state["chat_history"]) - 2:
                    st.divider()

else:  # 论文润色模式
    st.write("### 论文润色")
    polish_type = st.selectbox(
        "选择润色类型：",
        ["学术润色", "语法修改", "简单润色", "自定义润色"]
    )
    
    # 添加示例文本
    if st.checkbox("查看示例文本"):
        st.info("这是一个示例文本，展示了常见的学术写作。您可以参考这个格式来输入您的文本。")
        st.write("""—This paper introduces a highly flexible, quantized,
                    memory-efficient, and ultra-lightweight object detection network,
                    called TinyissimoYOLO. It aims to enable object detection on
                    microcontrollers in the power domain of milliwatts, with less
                    than 05MB memory available for storing convolutional neural
                    network (CNN) weights. The proposed quantized network archi
                    tecture with 422k parameters, enables real-time object detection
                    on embedded microcontrollers, and it has been evaluated to
                    exploit CNN accelerators. In particular, the proposed network has
                    been deployed on the MAX78000 microcontroller achieving high
                    frame-rate of up to 180fps and an ultra-low energy consumption
                    of only 196 J per inference with an inference efficiency of more
                    than 106MAC/Cycle. TinyissimoYOLO can be trained for any
                    multi-object detection. However, considering the small network
                    size, adding object detection classes will increase the size and
                    memory consumption of the network, thus object detection
                    with up to 3 classes is demonstrated. Furthermore, the net
                    work is trained using quantization-aware training and deployed
                    with 8-bit quantization on different microcontrollers, such as
                    STM32H7A3, STM32L4R9, Apollo4b and on the MAX78000’s
                    CNN accelerator. Performance evaluations are presented in this
                    paper.""")
    
    # 添加自定义润色要求
    custom_requirements = []
    if polish_type == "自定义润色":
        st.write("### 自定义润色要求")
        st.info("您可以添加最多5个自定义要求")
        
        # 预设一些常用要求供选择
        preset_requirements = [
            "提高学术性",
            "增加专业术语",
            "简化表达",
            "改善语法",
            "增加连贯性",
            "调整语气",
            "添加过渡句",
            "优化段落结构",
            "强调创新点",
            "其他"
        ]
        
        for i in range(5):
            col1, col2 = st.columns([3, 1])
            with col1:
                req = st.selectbox(
                    f"要求 {i+1}",
                    ["无"] + preset_requirements,
                    key=f"req_{i}"
                )
            if req == "其他":
                with col2:
                    req = st.text_input("自定义要求", key=f"custom_req_{i}")
            if req and req != "无":
                custom_requirements.append(req)
    
    text_input = st.text_area("请输入需要润色的文本：", height=200)
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.number_input("最大输出长度", min_value=100, value=1000, step=100)
    with col2:
        language = st.selectbox("输出语言", ["中文", "英文", "中英对照"])
    
    if st.button("开始润色", type="primary") and text_input and openai_api_key:
        with st.spinner("AI正在润色中，请稍等..."):
            polished_text = polish_paper(
                text_input, 
                polish_type, 
                model_name, 
                openai_api_key, 
                base_url,
                temperature,
                max_length,
                language,
                custom_requirements
            )
        st.write("### 润色结果")
        st.write(polished_text)
        
        # 添加结果操作按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("复制结果"):
                st.write("结果已复制到剪贴板！")
                st.session_state['clipboard'] = polished_text
        with col2:
            st.download_button(
                label="导出结果",
                data=polished_text,
                file_name="polished_text.txt"
            )

# 添加页脚
st.markdown("---")
st.markdown("Made with ❤️ by lucky_鹿人")
