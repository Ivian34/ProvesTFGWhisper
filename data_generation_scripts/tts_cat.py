from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
from datasets import Dataset, DatasetDict

# Cargar modelo y tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-cat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)
model = model.to(device)
model.eval()

rate = int(model.config.sampling_rate)
print("Tasa de muestreo:", rate)

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-cat")

dict_list = []
batch_size = 1  # Puedes ajustar el tamaño del batch
sentences_batch = []
tags_batch = []

with open("./generated_phrases.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        sentence, tags_str = line.split("\t")
        sentences_batch.append(sentence)
        tags_batch.append(tags_str)
        
        # Cuando alcanzamos el batch_size, procesamos el batch
        if len(sentences_batch) == batch_size:
            print("Procesando batch de", batch_size, "frases...")
            inputs = tokenizer(sentences_batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs).waveform 
            
            for i, sentence in enumerate(sentences_batch):
                audio = outputs[i].cpu().numpy()
                #audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
                dict_list.append({
                    'audio_sentence': audio,
                    'audio_rate': rate,
                    'text_sentence': sentence,
                    'tags': tags_batch[i]
                })
            sentences_batch = []
            tags_batch = []

# Procesamos el resto de frases que no forman un batch completo
if sentences_batch:
    inputs = tokenizer(sentences_batch, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs).waveform
    for i, sentence in enumerate(sentences_batch):
        audio = outputs[i].cpu().numpy()
        #audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
        dict_list.append({
            'audio_sentence': audio,
            'audio_rate': rate,
            'text_sentence': sentence,
            'tags': tags_batch[i]
        })

print("Total de ejemplos generados:", len(dict_list))

# Convertir la lista de ejemplos a un Dataset de Hugging Face
dataset = Dataset.from_dict({
    "audio_sentence": [d["audio_sentence"] for d in dict_list],
    "audio_rate": [d["audio_rate"] for d in dict_list],
    "text_sentence": [d["text_sentence"] for d in dict_list],
    "tags": [d["tags"] for d in dict_list],
})

# Crear un DatasetDict (por ejemplo, asignando todos los datos al split "train")
# Puedes dividir el dataset en train, validation y test si lo deseas
split = 0.9
train_size = int(len(dataset) * split)
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

# Guardar el DatasetDict en disco
dataset_dict.save_to_disk("../Audios/dataset1")
print("Dataset guardado localmente en '../Audios/dataset1'")
