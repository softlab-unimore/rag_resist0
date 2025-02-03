import os
from unstructured.partition.pdf import partition_pdf
from tqdm import tqdm
from PyPDF2 import PdfReader, PdfWriter
from functools import lru_cache
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytesseract
import pandas as pd
import tabula

from PIL import Image
from bs4 import BeautifulSoup
import torch

from pathlib import Path
import deepdoctection as dd

class TabulaTableExtractor:
    def __init__(self):
        pass

    def extract_page(self, pdf_path, page_num):
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        writer.add_page(reader.pages[page_num])  # Page number adjustment

        output_pdf_path = f"temp_page_{page_num}_tab.pdf"
        with open(output_pdf_path, "wb") as f:
            writer.write(f)

        return output_pdf_path

    @lru_cache
    def extract_tables(self, temp_pdf_path):
        return tabula.read_pdf(temp_pdf_path, pages='all', multiple_tables=True)

    @lru_cache
    def linearize(self, html_table):
        soup = BeautifulSoup(html_table, "html.parser")

        # Get table headers
        headers = []
        header_row = soup.find("thead").find_all("th")
        last_header = "Column"
        for header in header_row:
            curr_header = header.get_text(strip=True)
            if curr_header:
                headers.append(curr_header)
                last_header = header.get_text(strip=True)
            else:
                headers.append(last_header)

        # Get table rows
        rows = soup.find("tbody").find_all("tr")
        linearized_rows = []

        # Process each row in the table
        for row_idx, row in enumerate(rows):
            cells = row.find_all(["td", "th"])
            for col_idx, cell in enumerate(cells):
                # Determine column and row indicators
                row_indicator = cells[0].get_text(strip=True) #f"Row {row_idx + 1}"
                column_indicator = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"

                # Get the cell value
                cell_text = cell.get_text(strip=True)

                # If cell has colspan, adjust accordingly
                colspan = int(cell.get("colspan", 1))
                for span in range(colspan):
                    col_name = headers[col_idx + span] if (col_idx + span) < len(headers) else f"Column {col_idx + span + 1}"
                    # Create sentence in the specified format
                    sentence = f"{row_indicator}, {col_name}, {cell_text}"
                    linearized_rows.append(sentence)
        
        return '\n'.join(linearized_rows)

    def extract_table(self, documents, linearize=True):
        tables = [[] for _ in range(len(documents))]

        for doc_index, doc in tqdm(enumerate(documents)):
            pdf_name = doc.metadata["source"]

            page = doc.metadata["page"]

            try:
                temp_pdf_path = self.extract_page(f"{pdf_name}", int(page))
            except:
                print(f"Error extracting page {page} from {pdf_name}")
                continue

            html_tables = self.extract_tables(temp_pdf_path)

            tables_in_page = 0
            for html_table in html_tables:
                if linearize:
                    tables[doc_index].append((self.linearize(html_table.to_html(index=False, border=0)), pdf_name, page, tables_in_page))
                else:
                    tables[doc_index].append((html_table.to_html(index=False, border=0), pdf_name, page, tables_in_page))
                tables_in_page += 1

            os.remove(temp_pdf_path)

        return tables

class UnstructuredTableExtractor:
    def __init__(self, model_name, strategy):
        self.model_name = model_name #"yolox"
        self.strategy = strategy # "hi_res"

    @lru_cache
    def cached_partition_pdf(self, filename, strategy, model_name):
        return partition_pdf(
            filename=filename,
            strategy=strategy,
            infer_table_structure=True,
            model_name=model_name,
            languages=["ita"],
        )

    @lru_cache
    def linearize(self, html_table):
        soup = BeautifulSoup(html_table, "html.parser")

        # Get table headers
        headers = []
        try:
            header_row = soup.find("thead").find_all("th")
        except:
            header_row = []

        last_header = "Column"
        for header in header_row:
            curr_header = header.get_text(strip=True)
            if curr_header:
                headers.append(curr_header)
                last_header = header.get_text(strip=True)
            else:
                headers.append(last_header)

        # Get table rows
        try:
            rows = soup.find("tbody").find_all("tr")
        except:
            rows = soup.find_all("tr")

        linearized_rows = []

        # Process each row in the table
        for row_idx, row in enumerate(rows):
            cells = row.find_all(["td", "th"])
            for col_idx, cell in enumerate(cells):
                # Determine column and row indicators
                row_indicator = cells[0].get_text(strip=True) #f"Row {row_idx + 1}"
                column_indicator = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"

                # Get the cell value
                cell_text = cell.get_text(strip=True)

                # If cell has colspan, adjust accordingly
                colspan = int(cell.get("colspan", 1))
                for span in range(colspan):
                    col_name = headers[col_idx + span] if (col_idx + span) < len(headers) else f"Column {col_idx + span + 1}"
                    # Create sentence in the specified format
                    sentence = f"{row_indicator}, {col_name}, {cell_text}"
                    linearized_rows.append(sentence)
        
        return '\n'.join(linearized_rows)

    def extract_page(self, pdf_path, page_num):
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        writer.add_page(reader.pages[page_num])  # Page number adjustment

        output_pdf_path = f"temp_page_{page_num}_unst.pdf"
        with open(output_pdf_path, "wb") as f:
            writer.write(f)

        return output_pdf_path

    def extract_table(self, documents, linearize=True):
        tables = [[] for _ in range(len(documents))]

        for doc_index, doc in tqdm(enumerate(documents)):
            pdf_name = doc.metadata["source"]

            page = doc.metadata["page"]

            try:
                temp_pdf_path = self.extract_page(f"{pdf_name}", int(page))
            except:
                print(f"Error extracting page {page} from {pdf_name}")
                continue

            elements = self.cached_partition_pdf(
                filename=temp_pdf_path,
                strategy=self.strategy,
                model_name=self.model_name
            )

            tables_in_page = 0
            for element in elements:
                if element.category == "Table":
                    if linearize:
                        tables[doc_index].append((self.linearize(element.metadata.text_as_html), pdf_name, page, tables_in_page))
                    else:
                        tables[doc_index].append((element.metadata.text_as_html, pdf_name, page, tables_in_page))
                    tables_in_page+=1

            os.remove(temp_pdf_path)

        return tables


class CombinedTableExtractor:
    def __init__(self, model_name, strategy):
        self.unstructured_extractor = UnstructuredTableExtractor(model_name, strategy)
        self.tabula_extractor = TabulaTableExtractor()

    def get_column_count(self, soup, section="header"):
        """
        Get the number of columns in either the header or body of an HTML table.
        
        Parameters:
        - html_table: str, HTML table as a string
        - section: str, "header" to count columns in <thead> or "body" for <tbody>
        
        Returns:
        - int, column count
        """
        #soup = BeautifulSoup(html_table, "html.parser")
        
        if section == "header":
            header_row = soup.find("thead")
            if header_row:
                th_elements = header_row.find_all("th")
                return len(th_elements)
        elif section == "body":
            body_row = soup.find("tbody")
            if body_row:
                first_body_row = body_row.find("tr")
                if first_body_row:
                    td_elements = first_body_row.find_all("td")
                    return len(td_elements)
        
        return 0

    def extract_table(self, documents):
        #unstructured_tables = self.unstructured_extractor.extract_table(documents, linearize=False)
        #tabula_tables = self.tabula_extractor.extract_table(documents, linearize=False)

        with ThreadPoolExecutor() as executor:
            future_unstructured = executor.submit(self.unstructured_extractor.extract_table, documents, linearize=False)
            future_tabula = executor.submit(self.tabula_extractor.extract_table, documents, linearize=False)

            unstructured_tables = future_unstructured.result()
            tabula_tables = future_tabula.result()

        combined_tables = []
        counter = 0
        for doc_index, (unstructured_doc_tables, tabula_doc_tables) in enumerate(zip(unstructured_tables, tabula_tables)):
            doc_combined_tables = []
            if len(unstructured_doc_tables) != len(tabula_doc_tables):
                doc_combined_tables.extend(tabula_doc_tables)
            else:
                for table_index, (unstructured_table, tabula_table) in enumerate(zip(unstructured_doc_tables, tabula_doc_tables)):
                    tabula_html = BeautifulSoup(tabula_table[0], "html.parser")
                    unstructured_html = BeautifulSoup(unstructured_table[0], "html.parser")

                    tabula_headers = tabula_html.find("thead")
                    unstructured_body = unstructured_html.find("tbody")

                    tabula_len = self.get_column_count(tabula_html, section="header")
                    unstructured_len = self.get_column_count(unstructured_html, section="body")
                    if tabula_len != unstructured_len:
                        #print(f"Error: mismatch in number of rows for tabula and unstructured: {tabula_len}, {unstructured_len}")
                        doc_combined_tables.append(tabula_table)

                    if tabula_headers and unstructured_body:
                        unstructured_html.insert(0, tabula_headers)
                    else:
                        #print(f"Error: Mismatch in table structure for document {doc_index}, table {table_index}")
                        continue

                    combined_table = self.unstructured_extractor.linearize(str(unstructured_html))
                    doc_combined_tables.append([combined_table, unstructured_table[1], unstructured_table[2], unstructured_table[3]])
                    counter+=1
            combined_tables.append(doc_combined_tables)
        return combined_tables


class Docdetection:
    def __init__(self):
        self.analyzer = dd.get_dd_analyzer(config_overwrite=["LANGUAGE='ita'"])

    def extract_page(self, pdf_path, page_num):
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        writer.add_page(reader.pages[page_num])  # Page number adjustment

        output_pdf_path = f"temp_page_{page_num}_unst.pdf"
        with open(output_pdf_path, "wb") as f:
            writer.write(f)

        return output_pdf_path

    @lru_cache
    def linearize(self, html_table):
        soup = BeautifulSoup(html_table, "html.parser")

        # Get table headers
        headers = []
        try:
            header_row = soup.find("thead").find_all("th")
        except:
            header_row = []

        last_header = "Column"
        for header in header_row:
            curr_header = header.get_text(strip=True)
            if curr_header:
                headers.append(curr_header)
                last_header = header.get_text(strip=True)
            else:
                headers.append(last_header)

        # Get table rows
        try:
            rows = soup.find("tbody").find_all("tr")
        except:
            rows = soup.find_all("tr")

        linearized_rows = []

        # Process each row in the table
        for row_idx, row in enumerate(rows):
            cells = row.find_all(["td", "th"])
            for col_idx, cell in enumerate(cells):
                # Determine column and row indicators
                row_indicator = cells[0].get_text(strip=True) #f"Row {row_idx + 1}"
                column_indicator = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"

                # Get the cell value
                cell_text = cell.get_text(strip=True)

                # If cell has colspan, adjust accordingly
                colspan = int(cell.get("colspan", 1))
                for span in range(colspan):
                    col_name = headers[col_idx + span] if (col_idx + span) < len(headers) else f"Column {col_idx + span + 1}"
                    # Create sentence in the specified format
                    sentence = f"{row_indicator}, {col_name}, {cell_text}"
                    linearized_rows.append(sentence)
        
        return '\n'.join(linearized_rows)

    @lru_cache
    def get_tables(self, path):
        df = self.analyzer.analyze(path=path)
        df.reset_state()

        doc=iter(df)
        page = next(doc)

        return [page.tables[i].html for i in range(len(page.tables))]

    def extract_table(self, documents, linearize=True):
        tables = [[] for _ in range(len(documents))]

        for doc_index, doc in tqdm(enumerate(documents)):
            pdf_name = doc.metadata["source"]

            page = doc.metadata["page"]

            try:
                temp_pdf_path = self.extract_page(f"{pdf_name}", int(page))
            except:
                print(f"Error extracting page {page} from {pdf_name}")
                continue

            elements = self.get_tables(temp_pdf_path)

            tables_in_page = 0
            for element in elements:
                if linearize:
                    tables[doc_index].append((self.linearize(element), pdf_name, page, tables_in_page))
                else:
                    tables[doc_index].append((element, pdf_name, page, tables_in_page))
                tables_in_page+=1

            os.remove(temp_pdf_path)

        return tables
