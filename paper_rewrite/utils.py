from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def qa_agent(openai_api_key, base_url, memory, uploaded_file, question, model_name="gpt-3.5-turbo", temperature=0.7):
    model = ChatOpenAI(
        model=model_name,
        openai_api_key=openai_api_key,
        base_url=base_url,
        temperature=temperature
    )
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    embeddings_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese"
    )
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response

def polish_paper(text, polish_type, model_name, openai_api_key, base_url, 
                temperature=0.7, max_length=1000, language="中文", custom_requirements=None):
    model = ChatOpenAI(
        model=model_name,
        openai_api_key=openai_api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_length
    )
    
    # 根据语言选择设置提示词
    language_prompts = {
        "中文": "请用中文回复。",
        "英文": "Please respond in English.",
        "中英对照": "请同时用中文和英文回复，先中文后英文。"
    }
    
    if polish_type == "自定义润色" and custom_requirements:
        # 构建自定义润色提示词
        custom_prompt = "请对以下文本进行润色，要求：\n"
        for i, req in enumerate(custom_requirements, 1):
            custom_prompt += f"{i}. {req}\n"
        custom_prompt += f"{len(custom_requirements) + 1}. {language_prompts[language]}\n\n文本内容："
        prompt = custom_prompt
    else:
        prompts = {
            "学术润色": f"""请作为一位资深学术论文编辑，对以下文本进行学术性润色。要求：
1. 提升学术性表达
2. 确保专业术语使用准确
3. 保持学术写作风格
4. 改善句子结构使其更符合学术论文标准
5. 保持原意的同时提升表达的专业性
6. {language_prompts[language]}

文本内容：""",
            "语法修改": f"""请对以下文本进行语法优化和改进。要求：
1. 纠正语法错误
2. 改善句子结构
3. 确保时态一致性
4. 优化标点符号使用
5. 保持原意不变
6. {language_prompts[language]}

文本内容：""",
            "简单润色": f"""请对以下文本进行基础润色。要求：
1. 改善表达流畅度
2. 优化用词
3. 提升可读性
4. 保持语言简洁
5. 不改变原始含义
6. {language_prompts[language]}

文本内容："""
        }
        prompt = prompts[polish_type]
    
    # 添加字数限制提示
    length_prompt = f"\n\n请确保润色后的文本不超过{max_length}字。"
    
    response = model.invoke(prompt + text + length_prompt)
    return response.content
