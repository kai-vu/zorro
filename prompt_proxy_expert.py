from openai import OpenAI
import pandas as pd
import re
from pathlib import Path

# ---------- Config ----------
MODEL = "gpt-5-nano"
OUT_DIR = Path("prompt-extracted-nano")
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=open("openai-key.txt").read().strip())

system_prompt = """
You are an aviation maintenance knowledge engineer. Output TSV only, no prose. 
Reuse exact strings from prior TSVs unless told otherwise. 
All functions use short gerund phrases (eg. “regulating fuel flow”). 
Enforce referential integrity: reject any value not present in the provided lists or prior TSVs.
"""

# ---------- Helpers ----------
def strip_code_fences(text: str) -> str:
    """
    Remove surrounding ``` blocks or stray markdown fences.
    """
    text = text.strip()
    # remove leading/trailing fenced blocks
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
    if text.endswith("```"):
        text = re.sub(r"\s*```$", "", text)
    # remove any interior code fence lines that might sneak in
    text = re.sub(r"\n?```[a-zA-Z0-9]*\n?", "\n", text)
    return text.strip()


def save_tsv(text: str, path: Path) -> None:
    clean = strip_code_fences(text)
    path.write_text(clean, encoding="utf-8")
    print(f"Wrote {path}")


# ---------- Load source data ----------
parts_df = pd.read_csv("pdf-extracted/parts-catalog.csv")
# guard against NaNs and ensure string type
parts_df["Section"] = parts_df["Section"].astype(str).fillna("")
parts_df["Figure"] = parts_df["Figure"].astype(str).fillna("")

description = Path("prompts/description.txt").read_text(encoding="utf-8")

system_list = "\n".join("- " + s for s in parts_df["Section"].dropna().unique())
assembly_list = "\n".join("- " + f for f in parts_df["Figure"].dropna().unique())

prompt1 = f"""
{description}

Systems:
{system_list}

Assemblies:
{assembly_list}

Output one tsv table with two columns: the name of the system/assembly (exactly) and its function. 
The first row should be headers: "Component" and "hasFunction".
Use short verb phrases in gerund form (-ing) for each function. 
""".strip()

prompt2 = """
What are the dependencies of the functions? What is the hierarchy of functions?
Output one tsv with three columns: 
the name of the function (exactly), which functions (exactly) it depends on, and 
which more general parent function of which it is a child (exactly). 
The first row should be headers: "Function", "dependsOn", "parentFunction".

Rules:
- Function must be exactly one from the previous TSV’s hasFunction column.
- dependsOn is a semicolon-separated list of Functions from the same set; use empty cell if none.
- parentFunction is either one Function from the same set or empty if top-level.
- Do not invent new functions. No explanations.
""".strip()

trouble_df = pd.read_csv("pdf-extracted/troubleshooting.csv")
trouble_list = "\n".join(
    "- " + s for s in trouble_df["TROUBLE"].dropna().astype(str).unique()
)

# Extract probable-cause up to first parenthesis, then dedupe
cause_series = (
    trouble_df["PROBABLE CAUSE"]
    .astype(str)
    .str.extract(r"(.*?)(?:$| \()")[0]
    .dropna()
    .unique()
)
cause_list = "\n".join("- " + s for s in cause_series)

prompt3 = f"""
For the following problems, create a tsv with three columns: 
the problem (exactly), which system or assembly they are related to (exactly), 
and which function of that system or assembly is failing due to the problem (exactly).
The first row should be headers: "Problem", "ofComponent" and "failingFunction".

Rules:
- ofComponent must be exactly a Component from Prompt 1.
- failingFunction must be exactly a Function from Prompt 2 that the component has via hasFunction.


{trouble_list}
{cause_list}
""".strip()

# ---------- Conversation ----------
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt1},
]

print('Turn 1: Components → Functions')
resp1 = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    # temperature=0.2,
)
tsv1 = resp1.choices[0].message.content
save_tsv(tsv1, OUT_DIR / "functions.tsv")

# Add assistant content to history so the next prompt can reference it
messages.append({"role": "assistant", "content": strip_code_fences(tsv1)})

print('Turn 2: Function dependencies and hierarchy')
messages.append({"role": "user", "content": prompt2})
resp2 = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    # temperature=0.2,
)
tsv2 = resp2.choices[0].message.content
save_tsv(tsv2, OUT_DIR / "function_dependencies.tsv")

messages.append({"role": "assistant", "content": strip_code_fences(tsv2)})

print('Turn 3: Troubles → Component + failing function')
messages.append({"role": "user", "content": prompt3})
resp3 = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    # temperature=0.2,
)
tsv3 = resp3.choices[0].message.content
save_tsv(tsv3, OUT_DIR / "problem_component_function.tsv")

print("Done.")
