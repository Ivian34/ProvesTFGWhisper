import os
import json
import torch
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from typing import Dict, List, Union
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
import evaluate
import numpy as np

### Tokenizer
class MyCustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, special_tokens_on_vocab_file=False, **kwargs):
        """
        Args:
            vocab_file: the path to the vocabulary file.
            **kwargs: additional arguments for the tokenizer.
        """
        self.bos_token = "<|startoftranscript|>"
        self.eos_token = "<|endoftext|>" #Se utiliza tambien como padding. Creo que tambien como bos_token pero de momento no lo uso como tal
        self.transcribe_token = "<|transcribe|>"
        self.lang_token = "<|ca|>"
        self.notimestamps_token = "<|notimestamps|>"
        self.start_of_prev_token = "<|startofprev|>"

        self.special_tokens = [
            self.bos_token,
            self.eos_token,
            self.transcribe_token,
            self.lang_token,
            self.notimestamps_token,
            self.start_of_prev_token
        ]
        
        self.model_input_names = ["input_ids"]

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)

        if not special_tokens_on_vocab_file:
            offset = len(self.vocab)

            for j, token in enumerate(self.special_tokens):
                self.vocab[token] = offset + j

            
        self.inv_vocab = {i: word for word, i in self.vocab.items()}
        
        self.do_lower_case = True   

        super().__init__(
            errors="replace",
            unk_token=self.eos_token ,
            bos_token=self.eos_token ,
            eos_token=self.eos_token ,
            pad_token=None,
            add_prefix_space=False,
            **kwargs,
        )

    def _tokenize(self, text: str) -> list:
        """
        Tokenizes text.(the tags from the audios)
        """
        if self.do_lower_case:
            text = text.lower()
        tokens = text.split(',')
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        """
        Converts a string in a list of IDs.
        If add_special_tokens True, adds all the appropiate special tokens.
        """
        tokens = self._tokenize(text)
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        if add_special_tokens:
            token_ids = (
                [self.vocab[self.bos_token], self.vocab[self.lang_token], self.vocab[self.transcribe_token], self.vocab[self.notimestamps_token]]
                + token_ids
                + [self.vocab[self.eos_token]]
            )
        return token_ids

    def decode(self, token_ids: list, skip_special_tokens: bool = False) -> str:
        """
        Convert a list of IDs into a string. If skip_special_tokens is True, omit the special tokens in the resulting string.
        """
        tokens = [self._convert_id_to_token(i) for i in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return self.convert_tokens_to_string(tokens)

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token to its ID.
        """
        return self.vocab.get(token)

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts an ID to its corresponding token.
        """
        if isinstance(index, np.ndarray):
            index = int(index.item())
        return self.inv_vocab.get(index)

    def convert_tokens_to_string(self, tokens: list) -> str:
        """
        Converts a list of tokens(strings) in a string.
        """
        return " ".join(tokens)

    def get_vocab(self) -> dict:
        """
        Returns the whole vocab.
        """
        return self.vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> tuple:
        """
        Saves the vocabulary in a JSON file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        return (vocab_file,)

    def pad(self, lists_input_ids, return_tensors: bool = False):
        """
        Pads a list of lists of IDs to the maximum length in the list.
        Uses -100 as the padding value, which is ignored in the loss calculation.
        """
        max_length = max(len(ids) for ids in lists_input_ids)
        padded_input_ids = [ids + [-100] * (max_length - len(ids)) for ids in lists_input_ids]
        attention_mask = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in lists_input_ids]
        if return_tensors:
            padded_input_ids = torch.tensor(padded_input_ids)
            attention_mask = torch.tensor(attention_mask)
        return padded_input_ids, attention_mask

    def batch_encode(self, texts: list) -> list:
        """
        Applies the encode function to a list of texts.
        """
        return [self.encode(text) for text in texts]

    def batch_decode(self, batch_ids: list, skip_special_tokens: bool = False) -> list:
        """
        Decodes a list (batch) of ID sequences into texts.
        Allows skipping special tokens if skip_special_tokens is True.
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
    
    def __len__(self):
        return len(self.vocab)
    
    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary.
        """
        return len(self.vocab)
    
### Data collator

@dataclass
class MyDataCollator:
    feature_extractor: object
    tokenizer: MyCustomTokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [feature["labels"] for feature in features]
        labels_batch, attention_mask = self.tokenizer.pad(label_features, return_tensors=True)

        if (labels_batch[:, 0] == self.tokenizer.vocab[self.tokenizer.bos_token]).all().cpu().item():
            labels_batch = labels_batch[:, 1:]

        batch["labels"] = labels_batch
        batch["attention_mask"] = attention_mask
        
        return batch

### Modifiying WhisperForConditionalGeneration configs and embedding layers to adjust to the new vocab and tokenizer 

def create_my_whisper_model(myTokenizer: MyCustomTokenizer) -> WhisperForConditionalGeneration:
        model_id = "openai/whisper-small" # or large maybe

        model = WhisperForConditionalGeneration.from_pretrained(model_id)

        max_output_length = myTokenizer.vocab_size * 3
        model.config.bos_token_id = myTokenizer.vocab[myTokenizer.eos_token]
        model.config.eos_token_id = myTokenizer.vocab[myTokenizer.eos_token]
        model.config.pad_token_id = myTokenizer.vocab[myTokenizer.eos_token]
        model.config.decoder_start_token_id = myTokenizer.vocab[myTokenizer.bos_token]
        model.config.forced_decoder_ids = None
        model.config.begin_suppress_tokens = [myTokenizer.vocab[myTokenizer.eos_token]]
        model.config.suppress_tokens = []
        model.config.max_length = max_output_length #importante
        model.config.max_target_positions = max_output_length 

        model.generation_config.language = "ca"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids=None


        model.generation_config.bos_token_id = model.config.bos_token_id
        model.generation_config.eos_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = model.config.pad_token_id
        model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
        model.generation_config.begin_suppress_tokens = model.config.begin_suppress_tokens
        model.generation_config.suppress_tokens = model.config.suppress_tokens
        model.generation_config.no_timestamps_token_id = myTokenizer.vocab[myTokenizer.notimestamps_token]
        model.generation_config.prev_sot_token_id = myTokenizer.vocab[myTokenizer.start_of_prev_token]
        model.generation_config.lang_to_id[myTokenizer.lang_token] = myTokenizer.vocab[myTokenizer.lang_token]
        model.generation_config.task_to_id["transcribe"] = myTokenizer.vocab[myTokenizer.transcribe_token]
        model.generation_config.max_length = max_output_length #importante

        language = "ca"
        task = "transcribe"

        processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
        
        whisper_tokenizer = processor.tokenizer

        t_ids = []
        t_str = []
        for w in myTokenizer.vocab.keys():
            tokenized = whisper_tokenizer.tokenize(w)
            t_ids.append(whisper_tokenizer.convert_tokens_to_ids(tokenized))
            t_str.append(tokenized)

        ## whisper's embedding
        whisper_embedding = model.model.decoder.embed_tokens.weight

        #changing the embedding layer to the average of the tokenized words
        my_embedding = []
        for i in range(len(t_ids)):
            sum = torch.zeros(whisper_embedding.shape[1])
            for j in t_ids[i]:
                sum += whisper_embedding[j]
            avg = sum / len(t_ids[i])
            my_embedding.append(avg)

        my_embedding = torch.stack(my_embedding)


        #changing the embedding layer
        new_embedding = torch.nn.Embedding(my_embedding.shape[0], my_embedding.shape[1])
        new_embedding.weight.data = my_embedding
        print("embeding tokens layer and projection layer share the same weights:", id(model.model.decoder.embed_tokens.weight) == id(model.proj_out.weight)) #Mantener la misma referencia
        model.model.decoder.embed_tokens = new_embedding


        #changing the projection layer
        linear_projection = torch.nn.Linear(model.config.d_model, len(myTokenizer), bias=False)
        model.proj_out = linear_projection

        #Same weights as the decoder embedding
        model.proj_out.weight = model.model.decoder.embed_tokens.weight
        model.proj_out.weight.requires_grad = False
        model.proj_out.weight.requires_grad = True

        print("shape of embedding tokens layer: ", model.proj_out.weight.shape)

        #Important changing the vocab_size from the model's config
        model.config.vocab_size = len(myTokenizer)

        print("changing model's config vocab_size:", model.model.config.vocab_size)

        ## freezing the encoder
        model.freeze_encoder()

        return model

### Function to compute metrics

def compute_metrics(pred, model: WhisperForConditionalGeneration, myTokenizer: MyCustomTokenizer)-> Dict[str, float]:
    ### define the metrics
    metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    #replace -100 with the pad_token_id
    label_ids[label_ids == -100] = model.config.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = myTokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = myTokenizer.batch_decode(label_ids, skip_special_tokens=True)

    print("predicted: ", pred_str)
    print("labels: ", label_str)

    #wer in percentage
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    print("wer in %: ", wer)

    return {"wer": wer}

### Function to preprocess the dataset for the model training

def prepare_dataset(batch, feature_extractor: WhisperFeatureExtractor, myTokenizer: MyCustomTokenizer) -> Dict[str, Union[torch.Tensor, List[int]]]:
    audio = batch["audio_sentence"]
    sampling_rate = batch["audio_rate"]
    text = batch["tags"]

    batch["input_features"] = feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
    batch["labels"] = myTokenizer(text)
    return batch


# Function to create the CRF and apply viterbi algorithm
def crf(transitions_file, myTokenizer: MyCustomTokenizer, emission_scores: torch.Tensor):

    """
    transitions_file: path to the transitions file, this file must have the format:
    state transition1,transition2,...,transitionN
    where state is the string of the token and transition1,transition2,...,transitionN are the strings of the tokens that can be transitioned from state.
    """

    log_transition = np.full((myTokenizer.vocab_size, myTokenizer.vocab_size), -np.inf) 
    with open(transitions_file, "r") as f:
        for line in f:
            aux = line.strip().split()
            node = aux[0]
            relations = aux[1]
            node_id = myTokenizer.vocab[node]
            if relations != "None":
                relations = relations.split(",")
                for r in relations:
                    relations_id = myTokenizer.vocab[r]
                    log_transition[node_id, relations_id] = 1/len(relations)

    T = emission_scores.shape[0] 
    N = emission_scores.shape[1]  

    
    viterbi_score = np.full((T, N), -np.inf)
    backpointer = np.zeros((T, N), dtype=int)

    viterbi_score[0] = emission_scores[0]

    # Algoritmo Viterbi
    for t in range(1, T):
        for j in range(N):  # estado actual
            max_score = -np.inf
            arg_max = 0
            for i in range(N):  # estado anterior
 
                score = viterbi_score[t-1, i] + log_transition[i, j] + emission_scores[t, j]
                if score > max_score:
                    max_score = score
                    arg_max = i
            viterbi_score[t, j] = max_score
            backpointer[t, j] = arg_max

    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = np.argmax(viterbi_score[T-1])
    for t in range(T-2, -1, -1):
        best_path[t] = backpointer[t+1, best_path[t+1]]

    decoded_sequence = myTokenizer.decode(best_path.tolist(), skip_special_tokens=True)

    return decoded_sequence

