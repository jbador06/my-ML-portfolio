import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def train_tokenizer(data_dir: str, models_dir: str, vocab_size: int = 30000):
    """
    Entraîne un tokenizer WordPiece et sauvegarde le modèle avec HuggingFace.

    Args:
        data_dir (str): Chemin du répertoire contenant les fichiers de données.
        models_dir (str): Chemin du répertoire pour sauvegarder le tokenizer entraîné.
        vocab_size (int): Taille du vocabulaire. Par défaut : 30 000.
    """

    all_files_path = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train(
        files=all_files_path,
        vocab_size=vocab_size,
        min_frequency=5,
        limit_alphabet=1000,
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    tokenizer.save_model(models_dir, 'bert_tokenizer')
    print(f"Tokenizer entraîné et sauvegardé dans : {models_dir}")

def load_tokenizer(models_dir: str):
    """
    Charge un tokenizer WordPiece à partir du vocabulaire sauvegardé.

    Args:
        models_dir (str): Chemin du répertoire contenant le vocabulaire du tokenizer.

    Returns:
        transformers.BertTokenizer: Tokenizer chargé.
    """
    vocab_file = os.path.join(models_dir, 'bert_tokenizer-vocab.txt')
    return BertTokenizer.from_pretrained(vocab_file, local_files_only=True)

if __name__ == "__main__":
    DATA_DIR = "./data"
    MODELS_DIR = "./models/tokenizer"

    train_tokenizer(DATA_DIR, MODELS_DIR)

    # tokenizer = load_tokenizer(MODELS_DIR)
    # print("Tokenizer chargé avec succès.")
