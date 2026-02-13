import torch
import fasttext
import sys
import os
import warnings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =====================================================
# ENV & WARNINGS
# =====================================================
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================
# PROJECT ROOT
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# =====================================================
# INDIC TRANS TOOLKIT PATH
# =====================================================
toolkit_path = os.path.join(PROJECT_ROOT, "IndicTransToolkit")
if toolkit_path not in sys.path:
    sys.path.append(toolkit_path)

from IndicTransToolkit.processor import IndicProcessor

# =====================================================
# HYBRID TRANSLATOR
# =====================================================
class HybridTranslator:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        print(f"✅ HybridTranslator initialized | Device = {self.device}")

        # -----------------------------
        # FastText Language ID
        # -----------------------------
        lid_path = os.path.join(
            PROJECT_ROOT, "models", "fasttext", "lid.176.bin"
        )
        self.lid_model = fasttext.load_model(lid_path)

        # =====================================================
        # IndicTrans2 (Indic ↔ English)
        # =====================================================
        self.it2_model_name = "ai4bharat/indictrans2-indic-en-1B"

        self.it2_tokenizer = AutoTokenizer.from_pretrained(
            self.it2_model_name,
            trust_remote_code=True
        )

        self.it2_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.it2_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.use_cuda else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        self.ip = IndicProcessor(inference=True)

        # =====================================================
        # NLLB-200 (Other ↔ English)
        # =====================================================
        self.nllb_model_name = "facebook/nllb-200-distilled-600M"

        self.nllb_tokenizer = AutoTokenizer.from_pretrained(
            self.nllb_model_name
        )

        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.nllb_model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        # =====================================================
        # LANGUAGE MAPS
        # =====================================================
        self.indic_map = {
            "hi": "hin_Deva",
            "ta": "tam_Taml",
            "te": "tel_Telu",
            "kn": "kan_Knda",
            "ml": "mal_Mlym",
            "gu": "guj_Gujr",
            "pa": "pan_Guru",
            "bn": "ben_Beng",
            "mr": "mar_Deva",
        }

        self.nllb_map = {
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "ja": "jpn_Jpan",
            "zh": "zho_Hans",
            "ar": "ara_Arab",
        }

    # =====================================================
    # PUBLIC APIs
    # =====================================================
    def detect_language(self, text: str) -> str:
        labels, _ = self.lid_model.predict(text.replace("\n", " "), k=1)
        return labels[0].replace("__label__", "")

    def translate_to_english(self, text: str) -> str:
        return self.translate_with_scores(text)["final"]

    def translate_from_english(self, text: str, target_lang: str) -> str:
        if not text or target_lang == "en":
            return text

        if target_lang in self.indic_map:
            return self._translate_en_to_indic(
                text, self.indic_map[target_lang]
            )

        if target_lang in self.nllb_map:
            return self._translate_en_to_nllb(
                text, self.nllb_map[target_lang]
            )

        return text

    # =====================================================
    # ENSEMBLE TRANSLATION (→ ENGLISH)
    # =====================================================
    def translate_with_scores(self, text: str) -> dict:
        if not text or not text.strip():
            return self._empty_result()

        labels, probs = self.lid_model.predict(text.replace("\n", " "), k=1)
        lang = labels[0].replace("__label__", "")
        lang_conf = float(probs[0])

        if lang == "en":
            return {
                "t1": text,
                "t2": text,
                "score1": 1.0,
                "score2": 1.0,
                "final": text
            }

        # IndicTrans2
        t1 = ""
        if lang in self.indic_map:
            t1 = self._translate_indic(text, self.indic_map[lang])

        # NLLB
        t2 = self._translate_nllb(
            text, self.nllb_map.get(lang, "eng_Latn")
        )

        score1 = self._score_translation(t1, lang_conf)
        score2 = self._score_translation(t2, lang_conf)

        final = t1 if score1 >= score2 else t2

        if self.use_cuda:
            torch.cuda.empty_cache()

        return {
            "t1": t1,
            "t2": t2,
            "score1": round(score1, 3),
            "score2": round(score2, 3),
            "final": final
        }

    # =====================================================
    # INDIC ↔ EN (IndicTrans2)
    # =====================================================
    def _translate_indic(self, text, src_lang):
        batch = self.ip.preprocess_batch(
            [text], src_lang=src_lang, tgt_lang="eng_Latn"
        )

        inputs = self.it2_tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.it2_model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            tokens = self.it2_model.generate(
                **inputs, max_length=256, num_beams=5
            )

        decoded = self.it2_tokenizer.batch_decode(
            tokens, skip_special_tokens=True
        )
        return self.ip.postprocess_batch(decoded, lang="eng_Latn")[0]

    def _translate_en_to_indic(self, text, tgt_lang):
        batch = self.ip.preprocess_batch(
            [text], src_lang="eng_Latn", tgt_lang=tgt_lang
        )

        inputs = self.it2_tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.it2_model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            tokens = self.it2_model.generate(
                **inputs, max_length=256, num_beams=5
            )

        decoded = self.it2_tokenizer.batch_decode(
            tokens, skip_special_tokens=True
        )
        return self.ip.postprocess_batch(decoded, lang=tgt_lang)[0]

    # =====================================================
    # NLLB (BOTH DIRECTIONS)
    # =====================================================
    def _translate_nllb(self, text, src_lang):
        self.nllb_tokenizer.src_lang = src_lang
        inputs = self.nllb_tokenizer(text, return_tensors="pt")

        with torch.inference_mode():
            tokens = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(
                    "eng_Latn"
                ),
                max_length=256
            )

        return self.nllb_tokenizer.batch_decode(
            tokens, skip_special_tokens=True
        )[0]

    def _translate_en_to_nllb(self, text, tgt_lang):
        self.nllb_tokenizer.src_lang = "eng_Latn"
        inputs = self.nllb_tokenizer(text, return_tensors="pt")

        with torch.inference_mode():
            tokens = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(
                    tgt_lang
                ),
                max_length=256
            )

        return self.nllb_tokenizer.batch_decode(
            tokens, skip_special_tokens=True
        )[0]

    # =====================================================
    # SCORING
    # =====================================================
    def _score_translation(self, text: str, lang_conf: float) -> float:
        if not text:
            return 0.0

        length_score = min(len(text.split()) / 8.0, 1.0)
        fluency_score = 1.0 if text.strip().endswith(".") else 0.9

        return (
            0.5 * length_score +
            0.3 * fluency_score +
            0.2 * lang_conf
        )

    def _empty_result(self):
        return {
            "t1": "",
            "t2": "",
            "score1": 0.0,
            "score2": 0.0,
            "final": ""
        }


# =====================================================
# LOCAL TEST
# =====================================================
if __name__ == "__main__":
    translator = HybridTranslator()

    text = "मुझे बुखार है"
    detected = translator.detect_language(text)

    en = translator.translate_to_english(text)
    back = translator.translate_from_english(en, detected)

    print("Detected:", detected)
    print("English :", en)
    print("Back    :", back)
