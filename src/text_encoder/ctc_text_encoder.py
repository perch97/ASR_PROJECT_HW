import os
import re
from string import ascii_lowercase

import kenlm
import numpy as np
import torch
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

from ..utils.load_utils import download_lm, download_vocab


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, vocab_type=None, use_lm=False, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if vocab_type is None or vocab_type == "":
            alphabet = list(ascii_lowercase + " ")
        else:
            vocab_path = download_vocab(vocab_type)
            assert os.path.exists(vocab_path), "Vocab path does not exist."
            with open(vocab_path) as f:
                alphabet = [t.lower() for t in f.read().strip().split("\n")]
            if " " not in alphabet:
                alphabet.append(" ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        if use_lm:
            lm_path = download_lm()
            vocab_path = download_vocab("libri")
            with open(vocab_path) as f:
                unigrams = [t.lower() for t in f.read().strip().split("\n")]
            assert os.path.exists(lm_path), "LM path does not exist."
            kenlm_model = kenlm.Model(lm_path)
            lm = LanguageModel(kenlm_model, unigrams=unigrams)
            self.decoder_lm = BeamSearchDecoderCTC(Alphabet(self.vocab, False), lm)

        self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        result = []
        empty_token_index = self.char2ind[self.EMPTY_TOK]
        previous_index = empty_token_index

        for current_index in inds:
            if current_index != previous_index:
                if current_index != empty_token_index:
                    result.append(self.ind2char[current_index])
                previous_index = current_index

        return "".join(result)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def ctc_beamsearch(
        self, probs: torch.Tensor, type="lm", beam_size=10
    ) -> list[dict[str, float]]:
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        probabilities = np.exp(probs)

        if type == "lm":
            lm_hypothesis = self.decoder_lm.decode(probabilities, beam_size)
            return [
                {
                    "hypothesis": lm_hypothesis,
                    "probability": 1.0,
                }
            ]

        elif type == "nolm":
            nolm_hypothesis = self.decoder.decode(probabilities, beam_size)
            return [
                {
                    "hypothesis": nolm_hypothesis,
                    "probability": 1.0,
                }
            ]

        else:
            dynamic_programming = {("", self.EMPTY_TOK): 1.0}
            for step_probs in probabilities:
                updated_dp = {}
                for char_index, char_prob in enumerate(step_probs):
                    current_char = self.ind2char[char_index]
                    for (prefix, last_char), prefix_prob in dynamic_programming.items():
                        if last_char != current_char and current_char != self.EMPTY_TOK:
                            extended_prefix = prefix + current_char
                        else:
                            extended_prefix = prefix

                        updated_dp[(extended_prefix, current_char)] = (
                            updated_dp.get((extended_prefix, current_char), 0.0)
                            + prefix_prob * char_prob
                        )
                dynamic_programming = dict(
                    sorted(updated_dp.items(), key=lambda item: -item[1])[:beam_size]
                )

            sorted_hypotheses = sorted(
                dynamic_programming.items(), key=lambda item: -item[1]
            )

            result = [
                {"hypothesis": prefix, "probability": prob}
                for (prefix, _), prob in sorted_hypotheses
            ]
            return result
