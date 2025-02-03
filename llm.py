import transformers
import torch
import os
import logging

from transformers import BitsAndBytesConfig
from copy import deepcopy
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

load_dotenv()

logging.basicConfig(filename="./log/llm.log", level=logging.INFO)
logger = logging.getLogger("rag.llm")

class LlamaModel:
    def __init__(self, args):
        self.args = args
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" #"meta-llama/Llama-2-13b-chat-hf"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"use_cache":True, "max_length": 10},
            device_map="cuda",
            token=os.environ["HF_API_KEY"],
        )

        self.messages = [
            {"role": "system", "content": "Sei un assistente che estrae i valori numerici ambientali. Se non sai la risposta alla domanda, per favore non scrivere informazioni false."},
            {"role": "user", "content": "Estrai dal testo il valore numerico associato alla descrizione fornita. Se il valore non è presente, scrivi \"Non presente\". Non scrivere nient'altro.\n\nTesto: \n\"{}\"\n\n Descrizione: \"{}\""},
        ]

    def run(self, contents, query):
        results = []
        batch = [self.format_message(content.page_content, query) for content in contents]
        outputs = self.pipeline(batch, max_new_tokens=10, temperature=0.01)
        results = [0 if "assente" in output[0]["generated_text"][-1]["content"].lower() or "absent" in output[0]["generated_text"][-1]["content"].lower() else 1 for output in outputs]
        return results

    def format_message(self, descr, query):
        messages = deepcopy(self.messages)
        messages[1]["content"] = messages[1]["content"].format(descr, query)
        return messages


class OpenAIModel:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model_name = model_name
        self.temperature = temperature

        self.messages = [
            [
                "system",
                "Sei un assistente che estrae i valori numerici ambientali. Fornisci informazioni esclusivamente in base al contesto fornito. Non fornire informazioni di cui non sei sicuro. Le tue risposte dovranno essere solo codice, senza spiegazioni o formatting Markdown"
            ],
            [
                "human",
                "# TESTO \n\n{}\n\n # QUERY \n\n{}\n\n{} # ISTRUZIONE\n\nIn base al TESTO, rispondi alla QUERY fornita. Se il campo TABELLE ESTRATTE è presente, usa i dati in TABELLE ESTRATTE per estrarre la risposta dalle tabelle. I valori estratti devono essere massimo 5. Se il valore numerico che cerchi è segnato come numero mancante (e.g. n.a. oppure \ oppure - etc.) ritorna la stringa richiesta così come è (e.g. n.a. oppure \ oppure - etc.). Fornisci l'output come lista python con all'interno i valori numerici estratti come stringhe. Se la risposta non è presente, fornisci una lista vuota."
            ] # Estrai dal TESTO i valori numerici associati alla DESCRIZIONE fornita
        ]

    def _get_llm(self,):
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key
        )

        return self.llm

    def format_message(self, descr, query, tables_html):
        messages = deepcopy(self.messages)

        if len(tables_html) == 0:
            table_txt = ""
        else:
            table_txt = "# VALORI ESTRATTI DA TABELLE\n\n"
            for i, table in enumerate(tables_html):
                table_txt += f"## TABELLA NUMERO {i}\n\n{table[0]}\n\n"

        messages[1][1] = messages[1][1].format(descr, query, table_txt)
        messages = [tuple(messages[i]) for i in range(len(messages))]
        return messages

    def invoke(self, prompt):
        if not hasattr(self, "llm"):
            self._get_llm()

        result = self.llm.invoke(prompt)
        return result

    def run(self, contents, query, tables_html):
        batch = [self.format_message(content, query, table_html) for content, table_html in zip(contents, tables_html)]
        print(batch)
        with get_openai_callback() as cb:
            results = [self.invoke(prompt).content for prompt in batch]
            logger.info(cb)
        
        return results
