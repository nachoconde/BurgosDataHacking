import os
import jsonlines
import yaml
import constants as constants
from langchain.schema import Document
import json
import re
import requests
from bs4 import BeautifulSoup as bs4
import tiktoken


class DocsJSONLLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        with jsonlines.open(self.file_path) as reader:
            documents = []
            for obj in reader:
                page_content = obj.get("texto", "")
                metadata = {
                    "src": obj.get("src", ""),
                    "titulo": obj.get("titulo", ""),
                }
                documents.append(Document(page_content=page_content, metadata=metadata))
        return documents


def load_config():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root_dir, "config.yaml")) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_openai_api_key():
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

    return


def get_file_path():
    config = load_config()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(root_dir, "..")

    return os.path.join(parent_dir, config["jsonl_database_path"])


def get_query_from_user() -> str:
    try:
        query = input()
        return query
    except EOFError:
        print("Error: Input no esperado. Por favor intenta de nuevo.")
        return get_query_from_user()


def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def remove_existing_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def dump_jsonl(file_name: str, data: dict) -> None:
    """
    Dumps a dictionary to a jsonl file.

    Args:
        file_name (str): Name of the jsonl file.
        data (dict): Data to dump.
    """
    with open(file_name, "a") as jsonl_file:
        jsonl_file.write(json.dumps(data) + "\n")


def preprocess_text(text: str) -> str:
    """
    Preprocesses the text by removing certain patterns and characters.

    Args:
        text (str): Text to preprocess.

    Returns:
        The preprocessed text.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"Copyright.*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r":[a-z_&+-]+:", "", text)
    text = text.replace("«", "")
    text = text.replace("»", "")
    text = text.replace("“", "")
    text = (
        text.replace("á", "a")
        .replace("Á", "A")
        .replace("é", "e")
        .replace("É", "E")
        .replace("í", "i")
        .replace("Í", "I")
        .replace("ó", "o")
        .replace("Ó", "O")
        .replace("ú", "u")
        .replace("Ú", "U")
        .replace("ñ", "n")
        .replace("Ñ", "N")
    )
    text = text.replace(
        "Servicio Publico de Empleo Estatal 2023 \u2013 Informe del Mercado de Trabajo de la provincia de Burgos \u2013 Datos ",
        "",
    )
    return text


def fetch_page(url):
    try:
        page = requests.get(url, headers=constants.AGENT)
        page.raise_for_status()  # Raise an exception if the request was unsuccessful
        return page
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return None


def parse_html(page):
    try:
        soup = bs4(page.text, "lxml")
        return soup
    except (AttributeError, TypeError) as e:
        print(f"Error parsing HTML: {e}")
        return None


def extract_link_and_text(item):
    h2 = item.find("h2")
    if h2 is None:
        print("Error: No h2 tag found in item")
        return None, None
    href = h2.find("a")
    if href is None:
        print("Error: No a tag found in h2")
        return None, None
    return href.get("href"), href.text


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
