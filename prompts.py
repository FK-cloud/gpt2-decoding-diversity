"""20 fixed English prompts, organized into 5 themes x 4 prompts each"""

PROMPTS = {
    "News": [
        "The government announced a new economic policy today, stating that",
        "Local authorities issued a warning after heavy rainfall caused",
        "The company's stock price fell sharply this morning following",
        "Officials confirmed that the investigation into the incident",
    ],
    "Narrative": [
        "Once upon a time in a small village, there lived an old man who",
        "She opened the door slowly and saw something she never expected:",
        "The two travelers had been walking for hours when suddenly",
        "It was a cold winter night when the stranger knocked on the door and",
    ],
    "Science": [
        "Researchers at the university discovered a new method for",
        "The experiment showed that when the temperature increased,",
        "Scientists have long debated whether the theory of",
        "A recent study published in the journal found that",
    ],
    "History": [
        "During the medieval period, castles were built primarily to",
        "The fall of the Roman Empire was caused by a combination of",
        "In the early twentieth century, many countries began to",
        "The invention of the printing press changed society by",
    ],
    "Opinion": [
        "Many people believe that technology has made our lives better because",
        "One of the biggest challenges facing modern education is",
        "Critics argue that the new policy will fail to address",
        "In my view, the most important skill for the future is",
    ],
}


def flat_prompts():
    """Return a flat list of dicts: theme, theme_idx, prompt, global_idx (0-19)."""
    flat = []
    gi = 0
    for theme, plist in PROMPTS.items():
        for j, p in enumerate(plist):
            flat.append({"theme": theme, "theme_idx": j, "prompt": p, "global_idx": gi})
            gi += 1
    return flat


if __name__ == "__main__":
    fp = flat_prompts()
    assert len(fp) == 20, f"Expected 20 prompts, got {len(fp)}"
    for item in fp:
        print(item["global_idx"], item["theme"], "-", item["prompt"])
