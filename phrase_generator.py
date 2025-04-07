import random

# Vocabulari base
accions_base = ["agafar", "llençar", "deixar"]
objectes_base = ["caixa", "pilota"]

# Diccionaris de sinònims per als verbs i per als objectes.
# Les etiquetes sempre seran els valors base, però en la frase s'utilitza un sinònim aleatori.
synonyms_actions = {
    "agafar": ["agafar", "recollir", "prendre"],
    "llençar": ["llençar", "tira", "llança"],
    "deixar": ["deixar", "allibera", "solta"]
}

direct_forms = {
    "agafar": ["agafes", "reculls", "prens"],
    "llençar": ["llences", "tíres"],
    "deixar": ["deixes", "alliberes", "soltes"]
}

synonyms_objects = {
    "caixa": ["caixa", "capsa", "contenidor"],
    "pilota": ["pilota", "bola", "esfera"]
}

# Diccionari per a les formes imperatives (per a declaratives, B, R, etc.)
imperative_synonyms = {
    "agafar": ["agafa", "recull", "pren"],
    "llençar": ["llença", "tira", "llança"],
    "deixar": ["deixa", "allibera", "solta"]
}

# Funció per obtenir sinònims aleatoris
def get_synonym_action(base_action):
    """Retorna un sinònim aleatori per a l'acció base (forma base, no imperativa)."""
    return random.choice(synonyms_actions[base_action])

def get_synonym_object(base_object):
    """Retorna un sinònim aleatori per a l'objecte base."""
    return random.choice(synonyms_objects[base_object])

def get_imperative_synonym(base_action):
    """Retorna una forma imperativa aleatòria per a l'acció base."""
    return random.choice(imperative_synonyms[base_action])

# Funció per afegir el pronom enclític "la" als verbs (ja que la majoria d'objectes són femenins)
def attach_pronoun(verb_form):
    return verb_form + "-la"

# Funció per obtenir l'article correcte en funció del sinònim triat.
def get_object_with_article(base_object, obj_syn):
    """
    Retorna una cadena amb l'article definit correcte.
    Si el sinònim triat és "contenidor" (masculí), retorna "el {obj_syn}", sinó "la {obj_syn}".
    """
    if obj_syn.lower() == "contenidor":
        return f"el {obj_syn}"
    else:
        return f"la {obj_syn}"

# Funció per crear una frase segons l'escenari.
# Retorna també una llista de parells (acció, objecte) en l'ordre d'aparició per construir l'etiqueta.
def create_sentence(scenario):
    if scenario == "S":
        # Escenari Simple: una sola acció amb un sol objecte
        base_verb = random.choice(accions_base)
        base_obj = random.choice(objectes_base)
        verb_syn = get_imperative_synonym(base_verb)
        obj_syn = get_synonym_object(base_obj)
        sentence = f"{verb_syn} {get_object_with_article(base_obj, obj_syn)}"
        if random.random() < 0.3:
            if random.random() < 0.5:
                sentence = "Si us plau, " + sentence
            else:
                sentence += ", si us plau"
        sentence += random.choice([".", "!"])
        pairs = [(base_verb, base_obj)]
        return sentence, pairs

    elif scenario == "A":
        # Escenari A: dues accions amb dos objectes diferents.
        base_verb1 = random.choice(accions_base)
        base_verb2 = random.choice([a for a in accions_base if a != base_verb1])
        base_obj1 = random.choice(objectes_base)
        base_obj2 = random.choice([o for o in objectes_base if o != base_obj1])
        verb_syn1 = get_imperative_synonym(base_verb1)
        verb_syn2 = get_imperative_synonym(base_verb2)
        obj_syn1 = get_synonym_object(base_obj1)
        obj_syn2 = get_synonym_object(base_obj2)
        connector = " i "
        complement1 = ""
        complement2 = random.choice(["", " a terra", " sobre la taula"])
        if random.random() < 0.5:
            sentence = f"{verb_syn1} {get_object_with_article(base_obj1, obj_syn1)}{complement1}{connector}{verb_syn2} {get_object_with_article(base_obj2, obj_syn2)}{complement2}"
            pairs = [(base_verb1, base_obj1), (base_verb2, base_obj2)]
        else:
            sentence = f"{verb_syn2} {get_object_with_article(base_obj2, obj_syn2)}{complement2}{connector}{verb_syn1} {get_object_with_article(base_obj1, obj_syn1)}{complement1}"
            pairs = [(base_verb2, base_obj2), (base_verb1, base_obj1)]
        if random.random() < 0.3:
            if random.random() < 0.5:
                sentence = "Si us plau, " + sentence
            else:
                sentence += ", si us plau"
        sentence += random.choice([".", "!"])
        return sentence, pairs

    elif scenario == "B":
        # Escenari B: dues accions sobre el mateix objecte (la segona amb pronom enclític)
        base_verb1, base_verb2 = random.sample(accions_base, 2)
        base_obj = random.choice(objectes_base)
        verb_syn1 = get_imperative_synonym(base_verb1)
        verb_syn2 = get_imperative_synonym(base_verb2)
        obj_syn = get_synonym_object(base_obj)
        connector = " i "
        complement = random.choice(["", " a terra", " sobre la taula"])
        sentence = f"{verb_syn1} {get_object_with_article(base_obj, obj_syn)}{connector}{attach_pronoun(verb_syn2)}{complement}"
        pairs = [(base_verb1, base_obj), (base_verb2, base_obj)]
        if random.random() < 0.3:
            if random.random() < 0.5:
                sentence = "Si us plau, " + sentence
            else:
                sentence += ", si us plau"
        sentence += random.choice([".", "!"])
        return sentence, pairs

    elif scenario == "R":
        # Escenari R: frase invertida.
        if random.random() < 0.5:
            # Un sol verb aplicat a dos objectes
            base_verb = random.choice(accions_base)
            base_obj1, base_obj2 = random.sample(objectes_base, 2)
            verb_syn = get_imperative_synonym(base_verb)
            obj_syn1 = get_synonym_object(base_obj1)
            obj_syn2 = get_synonym_object(base_obj2)
            part1 = f"La {get_synonym_object(base_obj1)}"  # aquí s'utilitza el sinònim per l'objecte
            part1 = f"La {obj_syn1}, {attach_pronoun(verb_syn)}"  # reconstruïm amb article correcte
            part2 = f"La {obj_syn2}, {attach_pronoun(verb_syn)}"
            sentence = part1 + " i " + part2
            pairs = [(base_verb, base_obj1), (base_verb, base_obj2)]
        else:
            # Dos verbs diferents aplicats a dos objectes
            base_verb1, base_verb2 = random.sample(accions_base, 2)
            base_obj1, base_obj2 = random.sample(objectes_base, 2)
            verb_syn1 = get_imperative_synonym(base_verb1)
            verb_syn2 = get_imperative_synonym(base_verb2)
            obj_syn1 = get_synonym_object(base_obj1)
            obj_syn2 = get_synonym_object(base_obj2)
            part1 = f"La {obj_syn1}, {attach_pronoun(verb_syn1)}"
            part2 = f"La {obj_syn2}, {attach_pronoun(verb_syn2)}"
            sentence = part1 + " i " + part2
            pairs = [(base_verb1, base_obj1), (base_verb2, base_obj2)]
        if random.random() < 0.3:
            if random.random() < 0.5:
                sentence = "Si us plau, " + sentence
            else:
                sentence += ", si us plau"
        sentence += random.choice([".", "!"])
        return sentence, pairs

    elif scenario == "Q":
        # Escenari Q: frase tipus pregunta
        # Variante: s'escull entre modal (amb "Podries" o "Pots") i directa (conjugació en present).
        if random.random() < 0.5:
            # Variante modal
            modal_prefix = random.choice(["Podries", "Pots"])
            if random.random() < 0.5:
                # Pregunta simple modal
                base_verb = random.choice(accions_base)
                base_obj = random.choice(objectes_base)
                verb_text = get_synonym_action(base_verb)
                obj_text = get_synonym_object(base_obj)
                sentence = f"{modal_prefix} {verb_text} {get_object_with_article(base_obj, obj_text)}"
                pairs = [(base_verb, base_obj)]
            else:
                # Pregunta composta modal: dues accions, dos objectes
                base_verb1 = random.choice(accions_base)
                base_verb2 = random.choice([a for a in accions_base if a != base_verb1])
                base_obj1 = random.choice(objectes_base)
                base_obj2 = random.choice([o for o in objectes_base if o != base_obj1])
                verb_text1 = get_synonym_action(base_verb1)
                verb_text2 = get_synonym_action(base_verb2)
                obj_text1 = get_synonym_object(base_obj1)
                obj_text2 = get_synonym_object(base_obj2)
                sentence = f"{modal_prefix} {verb_text1} {get_object_with_article(base_obj1, obj_text1)} i {verb_text2} {get_object_with_article(base_obj2, obj_text2)}"
                pairs = [(base_verb1, base_obj1), (base_verb2, base_obj2)]
        else:
            # Variante directa: s'utilitza la conjugació directa (present indicatiu)
            if random.random() < 0.5:
                # Pregunta simple directa
                base_verb = random.choice(accions_base)
                base_obj = random.choice(objectes_base)
                direct_verb = random.choice(direct_forms[base_verb])
                obj_text = get_synonym_object(base_obj)
                sentence = f"{direct_verb} {get_object_with_article(base_obj, obj_text)}"
                pairs = [(base_verb, base_obj)]
            else:
                # Pregunta composta directa: dues accions i dos objectes
                base_verb1 = random.choice(accions_base)
                base_verb2 = random.choice([a for a in accions_base if a != base_verb1])
                base_obj1 = random.choice(objectes_base)
                base_obj2 = random.choice([o for o in objectes_base if o != base_obj1])
                direct_verb1 = random.choice(direct_forms[base_verb1])
                direct_verb2 = random.choice(direct_forms[base_verb2])
                obj_text1 = get_synonym_object(base_obj1)
                obj_text2 = get_synonym_object(base_obj2)
                sentence = f"{direct_verb1} {get_object_with_article(base_obj1, obj_text1)} i {direct_verb2} {get_object_with_article(base_obj2, obj_text2)}"
                pairs = [(base_verb1, base_obj1), (base_verb2, base_obj2)]
        if random.random() < 0.3:
            if random.random() < 0.5:
                sentence = "Si us plau, " + sentence
            else:
                sentence += ", si us plau"
        sentence += "?"
        return sentence, pairs

    else:
        return "", []

# Generem 1000 frases úniques
sentences_set = set()
max_tries = 30000
# Escenaris disponibles: S (simple), A, B, R, Q (pregunta)
scenarios = ["S", "A", "B", "R", "Q"]
weights = [0.15, 0.25, 0.20, 0.20, 0.20]

while len(sentences_set) < 1000 and max_tries > 0:
    max_tries -= 1
    scenario = random.choices(scenarios, weights=weights)[0]
    sentence, pairs = create_sentence(scenario)
    # Construïm l'etiqueta segons l'ordre dels parells: per cada parell (verb, objecte) afegim el verb i l'objecte base.
    label_parts = []
    for verb, obj in pairs:
        label_parts.append(verb)
        label_parts.append(obj)
    label = ",".join(label_parts)
    sentences_set.add(f"{sentence}\t{label}")

phrases_list = list(sentences_set)
random.shuffle(phrases_list)
phrases_list = phrases_list[:1000]

with open("generated_phrases.txt", "w", encoding="utf-8") as f:
    for line in phrases_list:
        f.write(line + "\n")

print("S'han generat 1000 frases (incloent preguntes) amb sinònims en el text i etiquetes amb els noms base, i s'han guardat a 'generated_phrases.txt'.")
