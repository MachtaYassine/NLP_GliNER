import streamlit as st
from gliner import GLiNER
import os

st.title("GLiNER Named Entity Recognition App")

# Load model (cache to avoid reloading on every run)
@st.cache_resource
def load_model():
    return GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

model = load_model()

# File uploader for large texts (books/scripts)
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.text_area("File content", value=text, height=200)
else:
    text = st.text_area("Enter text:", height=200, value="Paste your text here...")

labels = st.text_input("Enter entity types (comma-separated):", value="Person, Award, Date, Competitions, Teams")
threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)

entities = []

if st.button("Extract Entities"):
    if text.strip() and labels.strip():
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
        entities = model.predict_entities(text, label_list, threshold=threshold)
        if entities:
            st.write("### Extracted Entities")
            for entity in entities:
                st.write(f"**{entity['text']}**  ➔  `{entity['label']}`")
        else:
            st.info("No entities found.")
    else:
        st.warning("Please provide both text and entity types.")

def generate_wiki_page(entity):
    return f"""# {entity['text']}
**Type:** {entity['label']}

## Description
_Automatically generated entry for {entity['text']} ({entity['label']})._

## Mentions in text
- Context: _Add context extraction here if needed._
"""

if st.button("Generate Wiki Pages") and entities:
    output_dir = "wiki_pages"
    os.makedirs(output_dir, exist_ok=True)
    # Use set to avoid duplicate pages
    unique_entities = { (e['text'], e['label']) : e for e in entities }.values()
    for entity in unique_entities:
        filename = f"{entity['text'].replace(' ', '_')}_{entity['label']}.md"
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(generate_wiki_page(entity))
    st.success(f"Wiki pages generated in ./{output_dir}/")