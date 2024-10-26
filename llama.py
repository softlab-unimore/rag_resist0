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

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.openai")

class LlamaModel:
    def __init__(self, args):
        self.args = args
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" #"meta-llama/Llama-2-13b-chat-hf" #"meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"use_cache":True, "max_length": 10}, #"torch_dtype": torch.bfloat16,
            device_map="cuda",
            token=os.environ["HF_API_KEY"],
        )

        self.messages = [
            #{"role": "system", "content": "You are an helpful assistant that numerical green values. You are only allowed to reply \"yes\" or \"no\". If you don't know the answer to a question, please do not spread false information."},
            #{"role": "user", "content": "Consider the following page: \n\n\"{}\"\n\n And consider the following description: \"{}\" \n\n Inside the page, look for a numerical value associated with the description. If the value is inside the page, reply \"yes\", otherwise reply \"no\"."}
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

        self.preprocess_tables_message = [
            [
                "system",
                "Sei un assistente in grado di fare il preprocessing di tabelle estratte da PDF. Le tue risposte dovranno essere esclusivamente in formato HTML, senza spiegazioni o formato Markdown"
            ],
            [
                "human",
                "# TESTO \n\n{}\n\n{}\n\n Le tabelle in TABELLE HTML sono strutturalmente corrette, ma possono avere degli errori sui valori numerici. Al contrario, il TESTO può avere testo normale e tabelle senza struttura, ma con i valori numerici corretti. Correggi i valori forniti nelle TABELLE HTML con i rispettivi valori che si trovano in TESTO. Se nessuna tabella è presente, fornisci una lista Python vuota. Se molteplici tabelle sono presenti, ritorna una lista Python avente le tabelle HTML."
            ]
        ]

        self.table_message = [
            [
                "system",
                "Sei un assistente che estrae i valori numerici ambientali. Fornisci informazioni esclusivamente in base al contesto fornito. Non fornire informazioni di cui non sei sicuro. Le tue risposte dovranno essere solo codice, senza spiegazioni o formatting Markdown"
            ],
            [
                "human",
                "{}\n\n# DESCRIZIONE \n\n{}\n\n # ISTRUZIONE\n\nEstrai dalle TABELLE ESTRATTE i valori numerici associati alla DESCRIZIONE fornita. Se il valore numerico che cerchi è segnato come numero mancante (e.g. n.a. oppure \ oppure - etc.) ritorna la stringa richiesta così come è (e.g. n.a. oppure \ oppure - etc.). Fornisci l'output come lista python con all'interno i valori numerici estratti come stringhe. Se nessun valore è presente o nessuna tabella è presente, fornisci una lista vuota. "
            ]
        ]

        self.messages = [
            [
                "system",
                "Sei un assistente che estrae i valori numerici ambientali. Fornisci informazioni esclusivamente in base al contesto fornito. Non fornire informazioni di cui non sei sicuro. Le tue risposte dovranno essere solo codice, senza spiegazioni o formatting Markdown"
            ],
            [
                "human",
                "# TESTO \n\n{}\n\n # DESCRIZIONE \n\n{}\n\n # ISTRUZIONE\n\nEstrai dal TESTO i valori numerici associati alla DESCRIZIONE fornita. Se il valore numerico che cerchi è segnato come numero mancante (e.g. n.a. oppure \ oppure - etc.) ritorna la stringa richiesta così come è (e.g. n.a. oppure \ oppure - etc.). Fornisci l'output come lista python con all'interno i valori numerici estratti come stringhe. Se nessun valore è presente, fornisci una lista vuota. "
            ]
        ]

    def _get_llm(self,):
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key
        )

        return self.llm

    def format_message(self, descr, query): #, tables_html):
        messages = deepcopy(self.messages)

        """table_txt = "# TABELLE ESTRATTE\n\n"
        if len(tables_html) == 0:
            table_txt += f"{[]}\n\n"
        else:
            for i, table in enumerate(tables_html):
                table_txt += f"## TABELLA NUMERO {i}\n\n{table}\n\n" """

        messages[1][1] = messages[1][1].format(descr, query) #, table_txt)
        messages = [tuple(messages[i]) for i in range(len(messages))]
        return messages

    def format_multistep_message1(self, descr, tables_html):
        preprocess_tables_message = deepcopy(self.preprocess_tables_message)

        table_txt = "# TABELLE ESTRATTE\n\n"
        if len(tables_html) == 0:
            table_txt += f"{[]}\n\n"
        else:
            for i, table in enumerate(tables_html):
                table_txt += f"## TABELLA NUMERO {i}\n\n{table}\n\n"

        preprocess_tables_message[1][1] = preprocess_tables_message[1][1].format(descr, table_txt)
        preprocess_tables_message = [tuple(preprocess_tables_message[i]) for i in range(len(preprocess_tables_message))]
        return preprocess_tables_message

    def format_multistep_message2(self, tables_html, query):
        table_message = deepcopy(self.table_message)
        table_txt = "# TABELLE ESTRATTE"
        for i in range(len(tables_html)):
            table_txt += f"\n\n ## TABELLA NUMERO {i+1}\n\n{tables_html[i]}"

        table_message[1][1] = table_message[1][1].format(table_txt, query)
        table_message = [tuple(table_message[i]) for i in range(len(table_message))]
        return table_message

    def invoke(self, prompt):
        if not hasattr(self, "llm"):
            self._get_llm()

        result = self.llm.invoke(prompt)
        return result

    def run(self, contents, query): #, tables_html):
        #batch = [self.format_multistep_message1(content, table_html) for content, table_html in zip(contents, tables_html)]
        batch = [self.format_message(content, query) for content in contents]

        with get_openai_callback() as cb:
            results = [self.invoke(prompt).content for prompt in batch]
            #batch2 = [self.format_multistep_message2(table_html, query) for table_html in zip(results)]
            #results_final = [self.invoke(prompt).content for prompt in batch2]
            #results_txt = "\n".join(["- "+res for res in results])

            logger.info(cb)
        
        return results #results_final