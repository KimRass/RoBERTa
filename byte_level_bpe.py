# "We consider training BERT with a larger byte-level BPE vocabulary containing 50K subword units, without any additional preprocessing or tokenization of the input. This adds approximately 15M and 20M additional parameters for BERTBASE and BERTLARGE, respectively."


def parse(epubtxt_dir, with_document=False):
    print("Parsing BookCorpus...")
    lines = list()
    for doc_path in tqdm(list(Path(epubtxt_dir).glob("*.txt"))):
        with open(doc_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (not line) or (re.search(pattern=REGEX, string=line)) or (line.count(" ") < 1):
                    continue
                if not with_document:
                    lines.append(line)
                else:
                    lines.append((doc_path.name, line))
    print("Completed.")
    print(f"Number of paragraphs: {len(lines):,}")
    return lines