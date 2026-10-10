import os
import subprocess
import sys
from pathlib import Path

import pytest

from zeyrek.attributes import PhoneticAttribute, calculate_phonetic_attributes
from zeyrek.morphology import MorphAnalyzer

FIRST_10K = Path(__file__).parent.parent / "zeyrek" / "resources" / "tr" / "first-10K"


@pytest.fixture(scope="module")
def analyzer():
    return MorphAnalyzer()


def lemmas(analyzer, word):
    return sorted({parse.lemma for result in analyzer.analyze(word) for parse in result})


def morphemes(analyzer, word):
    return [parse.formatted for result in analyzer.analyze(word) for parse in result]


def test_cached_phonetic_attributes_are_not_shared():
    first = calculate_phonetic_attributes("ev")
    first.add(PhoneticAttribute.ExpectsConsonant)
    assert PhoneticAttribute.ExpectsConsonant not in calculate_phonetic_attributes("ev")


def test_analysis_does_not_change_after_many_words():
    analyzer = MorphAnalyzer()
    probes = ["gelip", "gerekiyor", "kazanarak", "söylendi"]
    before = {word: morphemes(analyzer, word) for word in probes}
    for line in FIRST_10K.read_text(encoding="utf8").splitlines():
        if line.strip():
            analyzer.analyze(line.split()[0])
    assert {word: morphemes(analyzer, word) for word in probes} == before


@pytest.mark.parametrize("seed", ["0", "1", "2", "3"])
def test_voicing_and_doubling_do_not_depend_on_hash_seed(seed):
    code = "from zeyrek import MorphAnalyzer; print(sorted({p.lemma for r in MorphAnalyzer().analyze('reddi') for p in r}))"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONHASHSEED": seed}).stdout
    assert "'ret'" in output


@pytest.mark.parametrize("word, morpheme", [
    ("olmalı", "Neces"),
    ("gülmeli", "Neces"),
    ("okumalıyız", "Neces"),
    ("olmaksızın", "WithoutHavingDoneSo"),
    ("okumaksızın", "WithoutHavingDoneSo"),
    ("olurcasına", "AsIf"),
    ("koşarcasına", "AsIf"),
])
def test_suffix_vowels_follow_the_vowel_before_them(analyzer, word, morpheme):
    assert any(f":{morpheme}" in formatted for formatted in morphemes(analyzer, word))


@pytest.mark.parametrize("word", ["olmalu", "gülmelü", "okumaksuzun"])
def test_wrong_vowel_harmony_is_rejected(analyzer, word):
    assert lemmas(analyzer, word) == ["Unk"]


@pytest.mark.parametrize("word, lemma", [("beraatı", "beraat"), ("beraata", "beraat"), ("beraati", "beraat"),
                                         ("emlakı", "emlak"), ("istirahatı", "istirahat")])
def test_everyday_spelling_without_voicing(analyzer, word, lemma):
    assert lemma in lemmas(analyzer, word)


def test_analyze_word_needs_no_tokenizer(analyzer, monkeypatch):
    monkeypatch.setitem(sys.modules, "nltk", None)
    assert "dava" in {parse.lemma for parse in analyzer.analyze_word("davacının")}
    assert [parse.lemma for parse in analyzer.analyze_word("xqzt")] == ["Unk"]


def test_unknown_word_gives_one_entry(analyzer):
    assert len(analyzer.analyze("xqzt ev")) == 2


@pytest.mark.parametrize("word, lemma", [("kitapla", "kitap"), ("bıçakla", "bıçak"), ("ayakla", "ayak"),
                                         ("amaçla", "amaç"), ("panikle", "panik")])
def test_instrumental_after_a_hard_consonant(analyzer, word, lemma):
    assert any(parse.lemma == lemma and ":Ins" in parse.formatted for parse in analyzer.analyze_word(word))


@pytest.mark.parametrize("word, lemma", [("kitabla", "kitap"), ("ayağla", "ayak"), ("bıçağla", "bıçak")])
def test_instrumental_after_a_softened_consonant_is_rejected(analyzer, word, lemma):
    assert lemma not in {parse.lemma for parse in analyzer.analyze_word(word)}


@pytest.mark.parametrize("word, lemma", [("müteahhitin", "müteahhit"), ("müteahhite", "müteahhit"),
                                         ("keşife", "keşif"), ("müdafiinin", "müdafii")])
def test_everyday_spelling_of_court_words(analyzer, word, lemma):
    assert lemma in {parse.lemma for parse in analyzer.analyze_word(word) if parse.pos != "Unk"}
