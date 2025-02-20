from src.metrics import BLEU, BLEURT, COMET, ROUGE, VIdScore


def test_compute_bleu():
    predictions = ["Hello, World!", "Hello, World!"]
    references = ["Hello, World!", "Hello, World!"]
    bleu = BLEU()
    result = bleu(predictions, references)
    assert isinstance(result["score"], float)


def test_compute_bleurt():
    predictions = ["Hello, World!"]
    references = ["Hello, World!"]
    bleurt = BLEURT()
    result = bleurt(predictions, references)
    assert isinstance(result["mean_score"], float)


def test_compute_rouge():
    predictions = ["Hello, World!"]
    references = ["Hello, World!"]
    rouge = ROUGE()
    result = rouge(predictions, references)
    assert result["rouge1"] == 1.0


def test_compute_comet():
    sources = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
    predictions = ["The fire could be stopped", "Schools and kindergartens were open"]
    references = ["They were able to control the fire.", "Schools and kindergartens opened"]
    comet = COMET()
    result = comet(predictions, references, sources)
    assert isinstance(result["mean_score"], float)


def test_compute_vid():
    predictions = ["Ônibus!", "O cara é louco!"]
    references = ["Autocarro!", "O indivíduo é doido!"]
    vid = VIdScore()
    result = vid(predictions, references)
    assert result["agreement"] == 0.5
