from typing import List

import torch
from torchaudio.models.decoder import ctc_decoder

from diplom.data import StringLabelEncoder


class IAMOnLineCTCDecoder:
    def __init__(
        self,
        string_encoder: StringLabelEncoder, 
    ):
        self._string_encoder = string_encoder
        self._tokens = ["<BLANK>", ] + list(string_encoder._encoder.classes_) + ["|", "<unk>"]
        self._ctc_decoder = ctc_decoder(
            lexicon=None,
            tokens=self._tokens,
            blank_token="<BLANK>",
        )

    def decode(self, probs: torch.FloatTensor, lengths: torch.LongTensor) -> List[str]:
        assert len(probs) == len(lengths)

        decoded_strings: List[str] = list()

        for hypotheses in self._ctc_decoder(probs, lengths):
            tokens_ids = hypotheses[0].tokens.tolist()
            tokens_ids = list(filter(lambda a: a != len(self._tokens) - 2, tokens_ids))

            str_decoded = self._string_encoder.inverse_transform(tokens_ids)
            decoded_strings.append(str_decoded)

        assert len(decoded_strings) == len(probs)
        return decoded_strings


class IAMOnLineCTCDecoderMultiprocessed:
    def __init__(
        self,
        string_encoder: StringLabelEncoder,
        num_processes: int = 10,
    ):
        from ctcdecode import CTCBeamDecoder

        self._labels = ["<BLANK>", ] + list(string_encoder._encoder.classes_)
        self._decoder = CTCBeamDecoder(
            labels=self._labels,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=100,
            num_processes=num_processes,
            blank_id=0,
            log_probs_input=False
        )
        self._string_encoder = string_encoder

    def decode(self, probs: torch.FloatTensor, lengths: torch.LongTensor) -> List[str]:
        assert len(probs) == len(lengths)

        beam_results, _, _, out_lens = self._decoder.decode(probs, lengths)

        decoded_strings: List[str] = list()

        assert len(beam_results) == len(out_lens)
        for beam_result, out_len in zip(beam_results, out_lens):
            tokens_ids = beam_result[0][:out_len[0]]
            str_decoded = self._string_encoder.inverse_transform(tokens_ids)
            decoded_strings.append(str_decoded)
        
        assert len(decoded_strings) == len(probs)

        return decoded_strings
