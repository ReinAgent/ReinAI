from src.llm.model import LLModel
from reinagent.src.optimse.optim import ScheduledOptim

import torch
import numpy as np

from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from collections import deque
import random
import copy

PAD = 0

class ReinAgent:
    def __init__(
        self,
        args,
        model: str,
        config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        assert torch.cuda.is_available(), "need to use GPUs"

        self.use_cuda = torch.cuda.is_available()
        self.cuda_devices = list(map(int, args.cuda_devices.split(",")))
        self.is_multigpu = len(self.cuda_devices) > 1
        self.device = "cuda"

        self.args = args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)             

        self.llmodel = model or LLModel(args)
        self.config = config or {}
        self.prompt_template = prompt_template
        self.evolution_history: List[Dict[str, Any]] = []
    
    def gen_optim(self, lr):
        optim = torch.optim.Adam(self.llmodel.parameters(), lr=lr, weight_decay=0.1)
        self.optim = ScheduledOptim(optim, lr)    

    def sent2tenosr(self, sentences):
        assert isinstance(sentences, str)

        idx = [self.evolution_history[w] for w in sentences]
        inp = torch.LongTensor(idx).unsqueeze(0)

        return inp,

    def beam_search(self, w_scores, end_seqs, top_seqs):
        max_scores, max_idxs = w_scores.sort(-1, descending=True)
        max_scores = (max_scores[:, :self.beam_size]).tolist()
        max_idxs = (max_idxs[:, :self.beam_size]).tolist()

        all_seqs, seen = [], []
        for index, seq in enumerate(top_seqs):
            seq_idxs, word_index, seq_score = seq
            for score, widx in zip(max_scores[index], max_idxs[index]):
                idx = self.widx2didx(widx)
                seq_idxs, word_index, seq_score = copy.deepcopy(seq)
                seq_score += score
                seq_idxs += [idx]
                word_index += [widx]
                if word_index not in seen:
                    seen.append(word_index)
                    all_seqs += [((seq_idxs, word_index, seq_score),
                                  seq_score,)]

        all_seqs += [((seq[0], seq[1], seq[-1]), seq[-1], True)
                     for seq in end_seqs]
        top_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)[
            :self.beam_size]

        all_done, done_nums = self.check_all_done(top_seqs)
        top_seqs = [seq for seq, _, _ in top_seqs]

        return top_seqs, all_done, self.beam_size-done_nums

    def check_all_done(self, seqs):
        done_nums = len([s for s in seqs if s[-1]])
        return done_nums == self.beam_size, done_nums


    def update_input(self, top_seqs):
        end_seqs, un_end_seqs, input_data = [], [], []
        for seq in top_seqs:
            end_seqs.append(seq)

        return torch.LongTensor(input_data), end_seqs, un_end_seqs

    def update_state(self, step, src_seq, enc_output, un_dones):
        input_pos = torch.arange(1, step+1).unsqueeze(0)
        input_pos = input_pos.repeat(un_dones, 1)
        input_pos = input_pos.long()

        src_seq_beam = src_seq.data.repeat(un_dones, 1)
        enc_output_beam = enc_output.data.repeat(un_dones, 1, 1)

        return input_pos, src_seq_beam, enc_output_beam

    def divine(self, sentences):
        def length_penalty(step, len_penalty_w=1.):
            return (torch.log(torch.FloatTensor([5 + step])) - torch.log(torch.FloatTensor([6])))*len_penalty_w

        with torch.no_grad():
            inp, position, turns = self.sent2tenosr(sentences)

            top_seqs = [([0], [], 0)] * self.beam_size
            enc_output = self.model.encode(inp, position, turns)
            inp_beam = inp.data.repeat(self.beam_size, 1)
            enc_output_beam = enc_output.data.repeat(self.beam_size, 1, 1)
            input_data = self.init_input()
            end_seqs = []
            for step in range(1, self.rewrite_len):
                dec_output = self.model.decode(
                    input_data, inp_beam, enc_output_beam)
                out = dec_output[:, -1, :]
                lp = length_penalty(step)
                top_seqs, all_done, un_dones = self.beam_search(
                    out.data+lp, end_seqs, top_seqs)
                if all_done:
                    break

                input_data, end_seqs, top_seqs = self.update_input(top_seqs)
                input_pos, src_seq_beam, enc_output_beam = self.update_state(
                    step+1, inp, enc_output, un_dones)
                inp_beam = inp.data.repeat(un_dones, 1)

            tgts = []
            for (cor_idxs, word_index, score) in top_seqs:
                cor_idxs = word_index[: -1]
                tgts += [("".join([self.word[idx]
                                   for idx in cor_idxs]), score)]
            return tgts

    
    def update_prompt(self, new_prompt: str) -> None:
        self.evolution_history.append({
            "old_prompt": self.prompt_template,
            "new_prompt": new_prompt,
            "timestamp": datetime.now().isoformat()
        })
        self.prompt_template = new_prompt
    
    def save_state(self, path: str) -> None:
        self.llmodel.save_model(path=path)
    
    def load_state(self, path: str):
        self.llmodel.load_model(path=path, cuda=self.use_cuda)