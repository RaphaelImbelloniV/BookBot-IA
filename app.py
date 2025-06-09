from flask import Flask, render_template, request, jsonify
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import GPT4All as LangChainGPT4All
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import json


modelo_path = Path("models/zephyr-7b-beta.Q8_0.gguf").resolve()
llm = LangChainGPT4All(model=str(modelo_path), allow_download=False, verbose=False)

with open("prompt_template.json", encoding="utf-8") as f:
    template_data = json.load(f)
template = template_data["template"]

prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(return_messages=True)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    output_parser=StrOutputParser()
)

app = Flask(__name__)
app.secret_key = "minha_chave_secreta"  # Obrigatório para session

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'response': 'Erro: Nenhuma mensagem recebida.'}), 400

    user_message = data['message'].strip()

    resposta = chain.run(input=user_message)

    return jsonify({"response": resposta.strip()})

if __name__ == "__main__":
    app.run(debug=True)
