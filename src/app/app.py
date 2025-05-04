import streamlit as st
from gliner import GLiNER
from gliner.multitask import GLiNERRelationExtractor, GLiNERSummarizer
import re
import threading
from collections import Counter, defaultdict
import os
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

st.title("GLiNER: Fast NER or Multitask (Relations/Summarization)")

task = st.radio("Choose task:", ["Entity Extraction (fast)", "Relations/Summarization (slower)"])

# Preload multitask models in the background after loading fast ER
multitask_models_loaded = threading.Event()
multitask_models = None

def background_load_multitask_models():
    global multitask_models
    multitask_models = load_multitask_models()
    multitask_models_loaded.set()

if task == "Entity Extraction (fast)":
    @st.cache_resource
    def load_model():
        return GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    model = load_model()
    # Start loading multitask models in the background
    if not multitask_models_loaded.is_set():
        threading.Thread(target=background_load_multitask_models, daemon=True).start()
# Comment out multitask model logic
# if task == "Relations/Summarization (slower)":
#     @st.cache_resource
#     def load_multitask_models():
#         model_id = 'knowledgator/gliner-multitask-v1.0'
#         model = GLiNER.from_pretrained(model_id)
#         relation_extractor = GLiNERRelationExtractor(model=model)
#         summarizer = GLiNERSummarizer(model=model)
#         return model, relation_extractor, summarizer
#     model, relation_extractor, summarizer = load_multitask_models()

# Preset options for entity and relation types
presets = {
    "Fiction Story": {
        "entities": "character, location, date, organization, artifact",
        "relations": "friend_of, enemy_of, family_of, member_of, located_in, owns"
    },
    "Historical": {
        "entities": "person, location, date, organization, event, title",
        "relations": "born_in, died_in, leader_of, member_of, occurred_in, part_of"
    }
}

# Try to load default text from dummy.txt
DEFAULT_TEXT = "Paste your text here..."
default_text_path = os.path.join(os.path.dirname(__file__), '../../dummy.txt')
if os.path.exists(default_text_path):
    with open(default_text_path, 'r', encoding='utf-8') as f:
        DEFAULT_TEXT = f.read()

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.text_area("File content", value=text, height=200)
else:
    text = st.text_area("Enter text:", height=200, value=DEFAULT_TEXT)

def smart_split_text(text, max_chunk_len=1024):
    """
    Split text into chunks of up to max_chunk_len characters, avoiding mid-sentence/dialogue cuts.
    Returns a list of text chunks.
    """
    # Split by sentence boundaries (handles dialogue and punctuation)
    sentence_endings = re.compile(r'([.!?]["\']?\s+)')
    sentences = []
    start = 0
    for match in sentence_endings.finditer(text):
        end = match.end()
        sentences.append(text[start:end])
        start = end
    if start < len(text):
        sentences.append(text[start:])
    # Now group sentences into chunks
    chunks = []
    current_chunk = ''
    for sent in sentences:
        if len(current_chunk) + len(sent) <= max_chunk_len:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# In process_in_chunks, comment out multitask model logic
def process_in_chunks(text, func, *args, **kwargs):
    """
    Process text in smart chunks, calling func(chunk, *args, **kwargs) for each chunk.
    Returns a list of results (one per chunk).
    For multitask models (no max length), process the whole text at once.
    """
    # if task == "Relations/Summarization (slower)":
    #     return [func(text, *args, **kwargs)]
    # Otherwise, chunk for fast model
    chunks = smart_split_text(text)
    results = []
    progress = st.progress(0, text="Processing chunks...")
    for i, chunk in enumerate(chunks):
        result = func(chunk, *args, **kwargs)
        results.append(result)
        progress.progress((i+1)/len(chunks), text=f"Processing chunk {i+1}/{len(chunks)} (Fast Gliner has a 1024 token limit so we have to chunk the text)")
    progress.empty()
    return results

def deduplicate_entities_majority_voting(entities):
    """
    For a list of entities (dicts with 'text' and 'label'),
    deduplicate by majority voting on the label for each unique text.
    """
    entity_types = defaultdict(list)
    for ent in entities:
        entity_types[ent['text']].append(ent['label'])
    deduped = []
    for text, labels in entity_types.items():
        most_common = Counter(labels).most_common(1)[0][0]
        deduped.append({'text': text, 'label': most_common})
    return deduped

def filter_relations(relations, entities):
    """
    Remove relations where answer is a substring of either entity or too generic.
    """
    filtered = []
    for rel in relations:
        e1, e2, answer = rel['entity1'], rel['entity2'], rel['relation']
        if answer.lower() in [e1.lower(), e2.lower()]:
            continue
        if len(answer.split()) < 2:  # Too short, likely not meaningful
            continue
        filtered.append(rel)
    return filtered

# Initialize Mistral client
api_key = os.environ["MISTRAL_API_KEY"]
mistral_model = "mistral-large-latest"
mistral_client = Mistral(api_key=api_key)

def mistral_qa(question, context):
    """
    Use Mistral API to answer a question given a context.
    """
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    chat_response = mistral_client.chat.complete(
        model=mistral_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content.strip()

def get_entity_descriptions(entity_contexts, qa_pipeline=None):
    """
    For each entity, use Mistral LLM to extract a detailed, clear, and context-aware description from its context(s).
    """
    descriptions = {}
    for entity, contexts in entity_contexts.items():
        if not contexts:
            continue
        context = ' '.join(contexts)
        question = (
            f"Based on the following text, provide a detailed and clear description of '{entity}'. "
            f"Include their role, characteristics (physical or moral), and any important relationships or actions. "
            f"If '{entity}' is a place or object, describe its significance or function."
        )
        try:
            res = mistral_qa(question, context)
            descriptions[entity] = res
        except Exception as e:
            descriptions[entity] = f"[Error: {e}]"
    return descriptions

def are_entities_equivalent(entity1, entity2, context):
    """
    Use Mistral LLM to determine if two entities are the same in the given context.
    Returns True if they are the same, False otherwise.
    """
    question = (
        f"Are '{entity1}' and '{entity2}' referring to the same entity in the following context? "
        f"If yes, answer 'yes' and explain why. If not, answer 'no' and explain why.\nContext: {context}"
    )
    try:
        res = mistral_qa(question, context)
        if res.lower().startswith('yes'):
            return True
    except Exception:
        pass
    return False

def refine_entity_list(entities, text):
    """
    Refine the entity list by merging entities that refer to the same thing using Mistral LLM.
    """
    merged = []
    used = set()
    for i, ent1 in enumerate(entities):
        if ent1['text'] in used:
            continue
        group = [ent1['text']]
        for j, ent2 in enumerate(entities):
            if i != j and ent2['text'] not in used:
                # Use a small context window for efficiency
                context = text
                if are_entities_equivalent(ent1['text'], ent2['text'], context):
                    group.append(ent2['text'])
                    used.add(ent2['text'])
        merged.append({'text': '/'.join(group), 'label': ent1['label']})
        used.update(group)
    return merged

# Define a set of relation types for historical/fiction settings
RELATION_TYPE_CONTEXT = """
Possible relations between entities (dates, characters, locations, organizations, artifacts):
- was born in, was born on, died in, died on, lived in, traveled to, visited, resided in, originated from, moved to, exiled to, escaped from
- is a friend of, is an enemy of, is a rival of, is an ally of, is a mentor of, is a student of, is a family member of, is a parent of, is a child of, is a sibling of, is a spouse of, is a descendant of, is an ancestor of
- is a member of, is a leader of, commands, rules, governs, founded, established, joined, left, betrayed, protected, served, worked for, employed by, represented, opposed, supported
- owns, possesses, created, invented, discovered, lost, found, used, destroyed, recovered, stole, gifted, inherited, traded, sold, bought, forged, repaired, enchanted
- took place in, occurred on, happened at, was held in, was signed in, was fought in, was won by, was lost by, was attended by, was witnessed by, was organized by, was caused by, was prevented by
- is located in, is near, is part of, is a region of, is a city in, is a country in, is a capital of, is a province of, is a territory of, borders, surrounds, contains, includes, is adjacent to
- is an artifact of, is a relic of, is a symbol of, is a weapon of, is a tool of, is a treasure of, is a document of, is a record of, is a map of, is a book of, is a letter of, is a decree of
"""

def get_sentence_relations(entities, text, qa_pipeline):
    """
    For each pair of entities in the same sentence, use QA to extract their relationship.
    Pass a set of possible relation types as context for the QA model.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relations = []
    for sent in sentences:
        ents_in_sent = [ent for ent in entities if ent['text'] in sent]
        for i in range(len(ents_in_sent)):
            for j in range(i+1, len(ents_in_sent)):
                ent1 = ents_in_sent[i]['text']
                ent2 = ents_in_sent[j]['text']
                # Predefined relationship questions
                questions = [
                    f"What is the relationship between {ent1} and {ent2}?",
                    f"How is {ent1} related to {ent2}?"
                ]
                for question in questions:
                    try:
                        # Add relation type context to the sentence
                        qa_context = RELATION_TYPE_CONTEXT + "\n" + sent
                        res = qa_pipeline({'question': question, 'context': qa_context})
                        if res['answer'] and res['score'] > 0.1 and res['answer'].lower() not in [ent1.lower(), ent2.lower(), 'no', 'none', 'unknown']:
                            relations.append({'entity1': ent1, 'entity2': ent2, 'relation': res['answer'], 'sentence': sent})
                            break
                    except Exception:
                        continue
    return relations

def get_entity_contexts(text, entities):
    """
    For each entity, collect all sentences where it appears.
    Returns a dict: entity_text -> [contexts]
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    entity_contexts = {ent['text']: [] for ent in entities}
    for sent in sentences:
        for ent in entities:
            if ent['text'] in sent:
                entity_contexts[ent['text']].append(sent)
    return entity_contexts

# Preset selection for both modes
preset_choice = st.selectbox(
    "Choose a preset for entity/relation types (optional):",
    ["Custom"] + list(presets.keys())
)
if preset_choice != "Custom":
    default_entities = presets[preset_choice]["entities"]
    default_relations = presets[preset_choice]["relations"]
else:
    default_entities = "person, company, year"
    default_relations = "founded, owns, works for"

if task == "Entity Extraction (fast)":
    labels = st.text_input("Enter entity types (comma-separated):", value=default_entities)
    threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
    if st.button("Extract Entities"):
        if text.strip() and labels.strip():
            label_list = [l.strip() for l in labels.split(",") if l.strip()]
            # Process in chunks
            chunk_results = process_in_chunks(text, model.predict_entities, label_list, threshold=threshold)
            # Flatten all entities
            entities = [ent for chunk in chunk_results for ent in chunk]
            # Deduplicate using majority voting
            unique_entities = deduplicate_entities_majority_voting(entities)
            if unique_entities:
                st.write("### Extracted Entities")
                for entity in unique_entities:
                    st.write(f"**{entity['text']}**  ➔  `{entity['label']}`")
                # Only call Mistral after displaying entities
                # Refine entity list
                unique_entities = refine_entity_list(unique_entities, text)
                entity_contexts = get_entity_contexts(text, unique_entities)
                descriptions = get_entity_descriptions(entity_contexts)
                st.write("### Entity Descriptions (QA-based)")
                for ent in unique_entities:
                    desc = descriptions.get(ent['text'], '')
                    if desc:
                        st.write(f"**{ent['text']}**: {desc}")
                # Extract relations between entities in the same sentence
                relations = get_sentence_relations(unique_entities, text, qa_pipeline)
                # Filter relations before displaying
                relations = filter_relations(relations, unique_entities)
                if relations:
                    st.write("### Relations between Entities (QA-based)")
                    for rel in relations:
                        st.write(f"`{rel['entity1']}` --**{rel['relation']}**--> `{rel['entity2']}` (in: _{rel['sentence']}_)")
            else:
                st.info("No entities found.")
        else:
            st.warning("Please provide both text and entity types.")
# In the UI, comment out multitask mode UI
# else:
#     entity_types = st.text_input("Enter entity types (comma-separated):", value=default_entities)
#     relation_types = st.text_input("Enter relation types (comma-separated):", value=default_relations)
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         extract_entities = st.button("Extract Entities (multitask)")
#     with col2:
#         extract_relations = st.button("Extract Relations")
#     with col3:
#         summarize = st.button("Summarize")
#     if "extracted_entities" not in st.session_state:
#         st.session_state["extracted_entities"] = []
#     if extract_entities:
#         ...existing code...
#     if extract_relations:
#         ...existing code...
#     if summarize:
#         ...existing code...
# ...existing code...