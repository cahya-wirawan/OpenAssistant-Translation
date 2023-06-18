from transformers import MarianMTModel, MarianTokenizer
import torch
import re
from pathlib import Path
import nltk
from collections import deque


class OpenTranslation:
    """
    Translator class for the MarianMTModel
    """
    def __init__(self, model_name, cache_dir=None, cache_name="default", cache_enabled=True,
                 cuda_number=None, use_multiple_cache=False,
                 max_sentence_length=400, max_translation_length=4000, max_sentence_array=50,
                 max_translation_cache=1000, use_auth_token=False):
        nltk.download('punkt')
        self.max_sentence_length = max_sentence_length
        self.max_translation_length = max_translation_length
        self.max_sentence_array = max_sentence_array
        self.max_translation_cache = max_translation_cache
        self.translation_cache = {}
        self.translation_cache_queue = deque()
        self.cache_enabled = cache_enabled
        if not cache_dir:
            cache_dir = Path.home()/".cache/translator"
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.transcriptions = {}
        self.cache_filename = cache_dir/f"transcription_{cache_name}.csv"
        if self.cache_enabled:
            if use_multiple_cache:
                for cache_filename in cache_dir.glob("**/*.csv"):
                    with open(cache_filename, "r") as f:
                        for line in f:
                            transcript_en, transcript_id = line.split("\t")
                            self.transcriptions[transcript_en.lower()] = transcript_id.strip()
            else:
                if self.cache_filename.exists():
                    with open(self.cache_filename, "r") as f:
                        for line in f:
                            transcript_en, transcript_id = line.split("\t")
                            self.transcriptions[transcript_en.lower()] = transcript_id.strip()
                else:
                    self.cache_filename.touch()
        self.tokenizer = MarianTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
        try:
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
        except ModuleNotFoundError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda" and cuda_number:
                self.device = f"{self.device}:{cuda_number}"
        print(f"Device: {self.device}")
        self.model = MarianMTModel.from_pretrained(model_name, use_auth_token=use_auth_token).to(self.device)

    def translate(self, source_text):
        """
        translate multiple sentences
        :param source_text:
        :return:
        """
        source = []
        for text in source_text:
            text = self.clean(text)
            if text == "":
                sentences = [""]
            else:
                sentences = self.split(text)
            for i, sentence in enumerate(sentences):
                sentence = sentence[:self.max_sentence_length]
                if sentence.lower() in self.transcriptions:
                    source.append([sentence, 0])
                else:
                    if self.is_alphabet(sentence):

                        source.append([sentence, 1])
                    else:
                        # print("Non Alphabet: ", sentence, flush=True)
                        source.append([sentence, 2])
                if i == 0:
                    source[-1] = source[-1] + [len(sentences)]
                else:
                    source[-1] = source[-1] + [0]
        translation = []
        result = []
        if len(source) != 0:
            inputs = [row[0] for row in source if row[1] == 1]
            if len(inputs) != 0:
                inputs_current = []
                inputs_length = 0
                for text in inputs:
                    text = re.sub(r"[^a-zA-Z ]{20,}", " ", text)
                    if len(inputs_current) > self.max_sentence_array or inputs_length + len(text) > self.max_translation_length:
                        translation += self._translate(inputs_current)
                        inputs_current = [text]
                        inputs_length = len(text)
                        continue
                    inputs_current.append(text)
                    inputs_length += len(text)
                if len(inputs_current) > 0:
                    translation += self._translate(inputs_current)
            translation_index = 0
            with open(self.cache_filename, "a") as f:
                i = 0
                while True:
                    transcription = ""
                    for k in range(source[i][2]):
                        if source[i+k][1] == 0:
                            transcription = transcription + " " + self.transcriptions[source[i+k][0].lower()]
                        elif source[i+k][1] == 1:
                            transcription_current = re.sub(r"[\r\n\t]+", " ", translation[translation_index])
                            transcription = transcription + " " + transcription_current
                            if self.cache_enabled:
                                self.transcriptions[source[i+k][0].lower()] = transcription_current
                                f.write(f"{source[i+k][0]}\t{transcription_current}\n")
                            translation_index += 1
                        else:
                            transcription = transcription + " " + source[i+k][0]
                    transcription = transcription.strip()
                    result.append(transcription)
                    i += source[i][2]
                    if i >= len(source):
                        break
        return result

    def _translate(self, inputs):
        """
        This is the real translation of multiple sentences
        :param inputs:
        :return:
        """
        inputs_current = []
        for text in inputs:
            if text not in self.translation_cache and text != "":
                inputs_current.append(text)
                self.translation_cache[text] = ""
        if len(inputs_current) != 0:
            inputs_tokens = self.tokenizer(inputs_current, return_tensors="pt", padding=True).to(self.device)
            translation_current = [self.tokenizer.decode(t, skip_special_tokens=True)
                                   for t in self.model.generate(**inputs_tokens)]
        else:
            translation_current = []
        translation = []
        counter = 0
        for text in inputs:
            if text == "":
                translation.append("")
            else:
                if text in self.translation_cache and self.translation_cache[text] != "":
                    translation.append(self.translation_cache[text])
                else:
                    translation.append(translation_current[counter])
                    self.translation_cache[text] = translation_current[counter]
                    self.translation_cache_queue.append(text)
                    counter += 1
        while len(self.translation_cache_queue) > self.max_translation_cache:
            self.translation_cache.pop(self.translation_cache_queue.popleft())
        return translation

    def split(self, text, max_sentences=1):
        sentences = nltk.sent_tokenize(text)
        sentence_list = []
        for i in range(0, len(sentences), max_sentences):
            for j in range(max_sentences):
                if len(sentences[i+j]) > self.max_sentence_length:
                    for sentence in sentences[i+j].split(","):
                        if not str(sentence).endswith("."):
                            sentence += ","
                        sentence_list.append(sentence.strip())
                else:
                    sentence_list += [sentences[i+j]]
        return sentence_list

    @staticmethod
    def clean(text):
        text = text.strip()
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"[\r\n\t]+", " ", text)
        return text

    @staticmethod
    def is_alphabet(text):
        try:
            text_encoded = text.encode().decode("unicode-escape")
            return not re.search(r'^[^a-zA-Y]+$', text_encoded)
        except UnicodeDecodeError as error:
            print(f"{error}: {text}")
            return False

