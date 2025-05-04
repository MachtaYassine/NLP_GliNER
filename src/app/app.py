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

import time
def refine_entity_list(entities, text):
    """
    Refine the entity list by merging entities that refer to the same thing using Mistral LLM.
    Shows a progress bar in both the console and the Streamlit app.
    Also computes relations and entity descriptions in the same pass to minimize Mistral queries.
    Times each Mistral query for performance insight.
    """
    merged = []
    used = set()
    total = len(entities)
    st_progress = st.progress(0, text="Refining entity list...")
    relations = []
    entity_descriptions = {}
    for i, ent1 in enumerate(entities):
        if ent1['text'] in used:
            st_progress.progress((i+1)/total, text=f"Refining entity {i+1}/{total}")
            print(f"Skipping {ent1['text']} (already used) [{i+1}/{total}]")
            continue
        group = [ent1['text']]
        # Compute entity description here
        context_sentences = [sent for sent in re.split(r'(?<=[.!?])\s+', text) if ent1['text'] in sent]
        context = ' '.join(context_sentences)
        question = (
            f"Based on the following text, provide a detailed and clear description of '{ent1['text']}'. "
            f"Include their role, characteristics (physical or moral), and any important relationships or actions. "
            f"If '{ent1['text']}' is a place or object, describe its significance or function."
        )
        print(f"Prompting Mistral for entity: {ent1['text']}")  # DEBUG
        try:
            desc = mistral_qa(question, context)
            print(f"{ent1['text']}: {desc}")  # DEBUG
        except Exception as e:
            desc = f"[Error: {e}]"
            print(f"{ent1['text']}: {desc}")  # DEBUG
        entity_descriptions[ent1['text']] = desc
        for j, ent2 in enumerate(entities):
            if i != j and ent2['text'] not in used:
                context = text
                print(f"Comparing '{ent1['text']}' and '{ent2['text']}' [{i+1}/{total}]")
                start_time = time.time()
                equivalent = are_entities_equivalent(ent1['text'], ent2['text'], context)
                elapsed = time.time() - start_time
                print(f"Mistral query for equivalence took {elapsed:.2f} seconds.")
                if equivalent:
                    print(f"  -> Merged '{ent2['text']}' with '{ent1['text']}'")
                    group.append(ent2['text'])
                    used.add(ent2['text'])
                else:
                    # Compute relation right away
                    rel_question = f"What is the relationship between {ent1['text']} and {ent2['text']}?"
                    rel_context = RELATION_TYPE_CONTEXT + "\n" + context
                    rel_start = time.time()
                    rel_answer = mistral_qa(rel_question, rel_context)
                    rel_elapsed = time.time() - rel_start
                    print(f"Mistral query for relation took {rel_elapsed:.2f} seconds.")
                    if rel_answer and rel_answer.lower() not in [ent1['text'].lower(), ent2['text'].lower(), 'no', 'none', 'unknown'] and len(rel_answer.split()) > 1:
                        relations.append({'entity1': ent1['text'], 'entity2': ent2['text'], 'relation': rel_answer, 'sentence': context})
        merged.append({'text': '/'.join(group), 'label': ent1['label']})
        used.update(group)
        st_progress.progress((i+1)/total, text=f"Refining entity {i+1}/{total}")
    st_progress.empty()
    print("Entity refinement complete.")
    print(f"Relations computed during refinement: {relations}")
    return merged, relations, entity_descriptions

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

def batch_mistral_qa(prompts):
    """
    Given a list of (prompt, context) tuples, send them as a batch to Mistral (if supported),
    otherwise process sequentially. Returns a list of responses.
    """
    responses = []
    for prompt, context in prompts:
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
        chat_response = mistral_client.chat.complete(
            model=mistral_model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                },
            ]
        )
        responses.append(chat_response.choices[0].message.content.strip())
    return responses

def efficient_entity_processing(entities, text):
    """
    For a list of entities, deduplicate, get descriptions, and compute relations with minimal Mistral calls.
    """
    entity_texts = [e['text'] for e in entities]
    n = len(entity_texts)
    # 1. Prepare all entity pairs for equivalence and relation checks
    pair_prompts = []
    pair_indices = []
    for i, ent1 in enumerate(entity_texts):
        for j, ent2 in enumerate(entity_texts):
            if i < j:
                eq_prompt = f"Are '{ent1}' and '{ent2}' referring to the same entity in the following context? If yes, answer 'yes' and explain why. If not, answer 'no' and explain why."
                pair_prompts.append((eq_prompt, text))
                pair_indices.append((i, j))
    # 2. Prepare all entity descriptions
    desc_prompts = []
    for ent in entity_texts:
        context_sentences = [sent for sent in re.split(r'(?<=[.!?])\s+', text) if ent in sent]
        context = ' '.join(context_sentences)
        desc_prompt = (
            f"Based on the following text, provide a detailed and clear description of '{ent}'. "
            f"Include their role, characteristics (physical or moral), and any important relationships or actions. "
            f"If '{ent}' is a place or object, describe its significance or function."
        )
        desc_prompts.append((desc_prompt, context))
    # 3. Run all equivalence prompts
    print(f"Running {len(pair_prompts)} equivalence checks...")
    eq_results = batch_mistral_qa(pair_prompts)
    # 4. Deduplicate entities based on equivalence
    groups = []
    used = set()
    eq_matrix = [[False]*n for _ in range(n)]
    for idx, (i, j) in enumerate(pair_indices):
        eq_matrix[i][j] = eq_results[idx].lower().startswith('yes')
        eq_matrix[j][i] = eq_matrix[i][j]
    for i in range(n):
        if entity_texts[i] in used:
            continue
        group = [entity_texts[i]]
        for j in range(i+1, n):
            if eq_matrix[i][j]:
                group.append(entity_texts[j])
                used.add(entity_texts[j])
        groups.append(group)
        used.update(group)
    merged = [{'text': '/'.join(group), 'label': entities[entity_texts.index(group[0])]['label']} for group in groups]
    # 5. Run all descriptions
    print(f"Running {len(desc_prompts)} description queries...")
    desc_results = batch_mistral_qa(desc_prompts)
    entity_descriptions = {ent: desc for ent, desc in zip(entity_texts, desc_results)}
    # 6. Prepare and run relation prompts for non-equivalent pairs
    rel_prompts = []
    rel_indices = []
    for idx, (i, j) in enumerate(pair_indices):
        if not eq_matrix[i][j]:
            rel_question = f"What is the relationship between {entity_texts[i]} and {entity_texts[j]}?"
            rel_context = RELATION_TYPE_CONTEXT + "\n" + text
            rel_prompts.append((rel_question, rel_context))
            rel_indices.append((i, j))
    print(f"Running {len(rel_prompts)} relation queries...")
    rel_results = batch_mistral_qa(rel_prompts)
    relations = []
    for (i, j), rel_answer in zip(rel_indices, rel_results):
        if rel_answer and rel_answer.lower() not in [entity_texts[i].lower(), entity_texts[j].lower(), 'no', 'none', 'unknown'] and len(rel_answer.split()) > 1:
            relations.append({'entity1': entity_texts[i], 'entity2': entity_texts[j], 'relation': rel_answer, 'sentence': text})
    print("Entity refinement and relation extraction complete.")
    print(f"Relations computed: {relations}")
    return merged, relations, entity_descriptions

def mistral_deduplicate_entities(entities, context):
    """
    Call Mistral to deduplicate entities. Returns a list of dicts: {"texts": [aliases], "label": label}
    """
    entity_list = [f"{e['text']} ({e['label']})" for e in entities]
    prompt = (
        "Given the following list of entities and the context, group together entities that refer to the same thing. "
        "Return a JSON list of objects, each with 'texts' (list of synonyms/aliases) and 'label' (entity type).\n"
        "Example output: [{\"texts\": [\"John\", \"Mr. Smith\"], \"label\": \"person\"}, ...]\n"
        f"Entities: {entity_list}\n"
        f"Context: {context}"
    )
    print(f'prompt deduplication: {prompt}')  # DEBUG
    response = mistral_client.chat.complete(
        model=mistral_model,
        messages=[{"role": "user", "content": prompt}]
    )
    import json
    try:
        deduped = json.loads(response.choices[0].message.content)
    except Exception:
        deduped = []
    return deduped

def mistral_describe_entities(deduped_entities, context):
    """
    Call Mistral to generate descriptions for each deduped entity group.
    Returns a list of dicts: {"texts": [...], "description": "..."}
    """
    prompt = (
        "For each entity group below, provide a detailed description based on the context. "
        "Return a JSON list of {\"texts\": [...], \"description\": \"...\"}.\n"
        f"Entities: {deduped_entities}\n"
        f"Context: {context}"
    )
    print(f'prompt description: {prompt}')  # DEBUG
    response = mistral_client.chat.complete(
        model=mistral_model,
        messages=[{"role": "user", "content": prompt}]
    )
    import json
    try:
        descriptions = json.loads(response.choices[0].message.content)
    except Exception:
        descriptions = []
    return descriptions

def mistral_entity_relations(deduped_entities, context):
    """
    Call Mistral to extract relations between deduped entities.
    Returns a list of dicts: {"entity1": "...", "entity2": "...", "relation": "...", "sentence": "..."}
    """
    prompt = (
        "For the following entities, list all meaningful relations between them found in the context. "
        "Return a JSON list of {\"entity1\": \"...\", \"entity2\": \"...\", \"relation\": \"...\", \"sentence\": \"...\"}.\n"
        f"Entities: {deduped_entities}\n"
        f"Context: {context}"
    )
    print(f'prompt relations: {prompt}')  # DEBUG
    response = mistral_client.chat.complete(
        model=mistral_model,
        messages=[{"role": "user", "content": prompt}]
    )
    import json
    try:
        relations = json.loads(response.choices[0].message.content)
    except Exception:
        relations = []
    return relations

if task == "Entity Extraction (fast)":
    print("Entered fast entity extraction task")  # DEBUG
    labels = st.text_input("Enter entity types (comma-separated):", value=default_entities)
    threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
    if st.button("Extract Entities"):
        print("Extract Entities button pressed")  # DEBUG
        if text.strip() and labels.strip():
            print("Text and labels are not empty")  # DEBUG
            label_list = [l.strip() for l in labels.split(",") if l.strip()]
            # Process in chunks
            print("Calling process_in_chunks")  # DEBUG
            chunk_results = process_in_chunks(text, model.predict_entities, label_list, threshold=threshold)
            print("process_in_chunks finished")  # DEBUG
            # Flatten all entities
            entities = [ent for chunk in chunk_results for ent in chunk]
            print(f"Entities found: {entities}")  # DEBUG
            # Deduplicate using majority voting
            unique_entities = deduplicate_entities_majority_voting(entities)
            print(f"Unique entities after deduplication: {unique_entities}")  # DEBUG
            if unique_entities:
                st.write("### Extracted Entities")
                for entity in unique_entities:
                    st.write(f"**{entity['text']}**  ➔  `{entity['label']}`")
                print("[PROGRESS] Printed extracted entities. Proceeding to deduplication...")
                # --- 3 Mistral calls ---
                st.write("\n---\n### Entity Groups (Deduplicated)")
                deduped = mistral_deduplicate_entities(unique_entities, text)
                print(f"Deduped entities: {deduped}")  # DEBUG
                print("[PROGRESS] Deduplication complete. Proceeding to descriptions...")
                for group in deduped:
                    st.write(f"**{' / '.join(group['texts'])}**  ➔  `{group['label']}`")
                st.write("\n---\n### Entity Descriptions")
                descriptions = mistral_describe_entities(deduped, text)
                print(f"Descriptions: {descriptions}")  # DEBUG
                print("[PROGRESS] Descriptions complete. Proceeding to relations...")
                for desc in descriptions:
                    st.write(f"**{' / '.join(desc['texts'])}**: {desc['description']}")
                st.write("\n---\n### Relations Between Entities")
                relations = mistral_entity_relations(deduped, text)
                print(f"Relations: {relations}")  # DEBUG
                print("[PROGRESS] Relations extraction complete. Displaying results.")
                for rel in relations:
                    st.write(f"`{rel['entity1']}` --**{rel['relation']}**--> `{rel['entity2']}` (in: _{rel['sentence']}_)")
            else:
                print("No entities found after deduplication")  # DEBUG
                st.info("No entities found.")
        else:
            print("Text or labels are empty")  # DEBUG
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