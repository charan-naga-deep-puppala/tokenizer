from flask import Flask, request, render_template

app = Flask(__name__)

##############################################################################
# 1. Tokenizer Code (same logic, but reorganized for user text)
##############################################################################

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# We'll store the merges and vocab in these global variables (initially None).
MERGES = None
VOCAB = None

def train_tokenizer_with_text(user_text):
    """
    Given some user-provided text, build the merges and vocab.
    Return (merges, vocab).
    """
    tokens = list(user_text.encode("utf-8"))
    vocab_size = 276  # example from your original code
    num_merges = vocab_size - 256
    
    ids = tokens[:]
    merges = {}

    for i in range(num_merges):
        stats = get_stats(ids)
        # if stats is empty, break out early (tiny text, or no pairs)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    # Build vocab dict
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    return merges, vocab

def encode_text(user_input):
    """
    Encode user_input (string) with the already-trained MERGES.
    """
    global MERGES
    if MERGES is None:
        return None  # We haven't trained yet!

    tokens = list(user_input.encode("utf-8"))
    
    # Repeatedly merge pairs if they're in MERGES
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # find the pair in stats that has the *lowest index in MERGES*
        # or break if it doesn't exist in MERGES
        # (the approach from your code: pair = min(stats, key= lambda p: merges.get(p, inf)))
        pair = min(stats, key=lambda p: MERGES.get(p, float('inf')))
        if pair not in MERGES:
            break
        idx = MERGES[pair]
        tokens = merge(tokens, pair, idx)

    return tokens

def decode_tokens(token_ids):
    """
    Decode a list of token IDs to text, using VOCAB.
    """
    global VOCAB
    if VOCAB is None:
        return None  # We haven't trained yet!

    tokens = b"".join(VOCAB[idx] for idx in token_ids)
    text = tokens.decode('utf-8', errors='replace')
    return text

##############################################################################
# 2. Flask Routes
##############################################################################

@app.route("/", methods=["GET"])
def index():
    """
    Displays a page with 3 main sections:
      1) Train the tokenizer with user text
      2) Encode user text
      3) Decode token IDs
    """
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    """
    Gets text from user, trains the tokenizer, stores (MERGES, VOCAB) globally.
    """
    global MERGES, VOCAB
    user_text = request.form.get("train_text", "")
    if not user_text.strip():
        return "<h1>No text provided!</h1><a href='/'>Back</a>"

    # Train
    MERGES, VOCAB = train_tokenizer_with_text(user_text)

    return (
        "<h1>Tokenizer Trained Successfully!</h1>"
        "<p>Now you can encode or decode using the forms below.</p>"
        "<a href='/'>Go Back</a>"
    )


@app.route("/encode", methods=["POST"])
def encode_route():
    """
    Encode user text (POST) and return the resulting token list
    """
    user_input = request.form.get("text_to_encode", "")
    if MERGES is None or VOCAB is None:
        return "Tokenizer not trained yet. Please train first."

    token_ids = encode_text(user_input)
    if token_ids is None:
        return "Tokenizer not trained or something went wrong."

    return f"<h2>Encoded Tokens:</h2><p>{token_ids}</p><br><a href='/'>Back</a>"


@app.route("/decode", methods=["POST"])
def decode_route():
    """
    Decode user token IDs (POST) and return the resulting string
    """
    if MERGES is None or VOCAB is None:
        return "Tokenizer not trained yet. Please train first."

    user_input = request.form.get("text_to_decode", "")
    # We expect something like: [256, 72, 100, ...]
    try:
        token_ids = eval(user_input)  # In production, prefer safer parsing (like json.loads).
        if not isinstance(token_ids, list):
            return "Please provide a Python-style list of integers, e.g. [256, 72, 100]."
    except:
        return "Invalid token list format. Example: [256, 72, 100]"

    decoded = decode_tokens(token_ids)
    if decoded is None:
        return "Tokenizer not trained or something went wrong."

    return f"<h2>Decoded Text:</h2><p>{decoded}</p><br><a href='/'>Back</a>"


if __name__ == "__main__":
    app.run(debug=True)
