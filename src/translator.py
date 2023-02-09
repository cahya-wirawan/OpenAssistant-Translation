from transformers import MarianMTModel, MarianTokenizer
import torch
import re
from pathlib import Path
import nltk
from collections import deque


class Translator:
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


def main():
    model_name = "Wikidepia/marian-nmt-enid"
    translator = Translator(model_name)

    sentences = [
        "I am hungry, but I don't have money to buy food",
        "Jakarta is the capital city of Indonesia",
        "Vienna is known for its high quality of life. In a 2005 study of 127 world cities, the Economist Intelligence Unit ranked the city first (in a tie with Vancouver and San Francisco) for the world's most livable cities. Between 2011 and 2015, Vienna was ranked second, behind Melbourne.[18] Monocle's 2015 'Quality of Life Survey' ranked Vienna second on a list of the top 25 cities in the world to 'make a base within'.[19] Monocle's 2012 'Quality of Life Survey' ranked Vienna fourth on a list of the top 25 cities in the world 'to make a base within' (up from sixth in 2011 and eighth in 2010).[20] The UN-Habitat classified Vienna as the most prosperous city in the world in 2012–2013.[21] The city was ranked 1st globally for its culture of innovation in 2007 and 2008, and sixth globally (out of 256 cities) in the 2014 Innovation Cities Index, which analyzed 162 indicators in covering three areas: culture, infrastructure, and markets.[22][23][24] Vienna regularly hosts urban planning conferences and is often used as a case study by urban planners.[25] Between 2005 and 2010, Vienna was the world's number-one destination for international congresses and conventions. It attracts over 6.8 million tourists a year",
        "My friend goes to school every morning with bus",
        "Due to his popularity as Army chief of staff, he was widely expected to run in the 1948 election.",
        "The dog is chasing my cat around the house",
        "0-\\031]+(?:(?:(?:\\r\\n)?[ \\t])+|\\Z",
        ":[^() < > @,;:\".[] \x00-\x19]+(?:(?:(?:\r\n)?",
        "Until the beginning of the 20th century, Vienna was the largest German-speaking city in the world, and before the splitting of the Austro-Hungarian Empire in World War I, the city had 2 million inhabitants. Today, it is the second-largest German-speaking city after Berlin. Vienna is host to many major international organizations, including the United Nations, OPEC and the OSCE. The city is located in the eastern part of Austria and is close to the borders of the Czech Republic, Slovakia and Hungary. These regions work together in a European Centrope border region. Along with nearby Bratislava, Vienna forms a metropolitan region with 3 million inhabitants. In 2001, the city center was designated a UNESCO World Heritage Site. In July 2017 it was moved to the list of World Heritage in Danger. Additionally, Vienna is known as the 'City of Music' due to its musical legacy, as many famous classical musicians such as Beethoven and Mozart called Vienna home. Vienna is also said to be the 'City of Dreams' because it was home to the world's first psychoanalyst, Sigmund Freud. Vienna's ancestral roots lie in early Celtic and Roman settlements that transformed into a Medieval and Baroque city. It is well known for having played a pivotal role as a leading European music center, from the age of Viennese Classicism through the early part of the 20th century. The historic center of Vienna is rich in architectural ensembles, including Baroque palaces and gardens, and the late-19th-century Ringstraße lined with grand buildings, monuments and parks."
    ]
    transcriptions = translator.translate(sentences)
    print(transcriptions)


if __name__ == "__main__":
    main()
