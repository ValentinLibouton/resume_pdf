import fitz
from transformers import pipeline
import math

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_chapters_from_pdf(pdf_path, chapter_starts):
    doc = fitz.open(pdf_path)
    chapter_dict = {}
    num_chapters = len(chapter_starts) - 1  # On suppose que la dernière entrée est la fin du dernier chapitre

    # Extraire le texte pour chaque chapitre
    for i in range(num_chapters):
        start_page = chapter_starts[i]
        end_page = chapter_starts[i + 1] - 1  # La page de fin est la page de début du prochain chapitre moins 1
        text = ""

        # Concaténer le texte de chaque page du chapitre
        for page_num in range(start_page, end_page + 1):  # +1 pour inclure la page de fin
            page = doc.load_page(page_num)
            text += page.get_text()

        chapter_dict[i + 1] = text  # Utiliser un nom générique avec numéro de chapitre

    return chapter_dict


def extract_text_from_page(pdf_path, page_number):
    """
    Extract text from a specific page in a PDF file.

    Args:
    pdf_path (str): The path to the PDF file.
    page_number (int): The page number from which to extract text (0-indexed).

    Returns:
    str: The extracted text from the specified page.
    """
    page_number = page_number - 1
    # Ouvrir le fichier PDF
    doc = fitz.open(pdf_path)

    # Vérifier si le numéro de page est valide
    if page_number < 0 or page_number >= len(doc):
        return "Page number is out of range."

    # Sélectionner la page spécifique
    page = doc.load_page(page_number)

    # Extraire le texte de la page
    text = page.get_text()

    # Fermer le document PDF
    doc.close()

    return text


def summarize_chapters(chapters):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = {}
    for chapter, text in chapters.items():
        summaries[chapter] = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    return summaries


def summarize_text(text):
    """
    Summarizes the provided text using a pre-initialized summarizer.

    Args:
    text (str): The text to summarize.

    Returns:
    str: The summarized text.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    input_length = len(text.split())  # Nombre de mots dans le texte d'entrée
    max_length = max(30, math.ceil(
        input_length / 4))  # Ne pas descendre en dessous d'une longueur minimale, par exemple 30 mots
    min_length = max(20, math.ceil(input_length / 6))
    # Generate summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    # Return only the summary text
    return summary[0]['summary_text']