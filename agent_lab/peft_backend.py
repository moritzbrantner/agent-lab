"""Offline Transformers/PEFT adapter and bounded local LoRA training."""

import asyncio
import gc
import hashlib
import json
import os
import time
from pathlib import Path

from agent_lab.experiments import canonical, digest
from agent_lab.learning import validate_data
from agent_lab.runtime import Reply, ResponseFailure, RuntimeFailure
from agent_lab.telemetry import EnergyCounter, Telemetry


def file_hashes(root):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(root).iterdir())
        if p.is_file() and p.name != "acquisition.json"
    }


def environment(seed):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch


def load_local(path, plan):
    torch = environment(plan["seed"])
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Pinned learning experiment requires BF16-capable CUDA")
    acquisition = json.loads((Path(path) / "acquisition.json").read_text())
    if (
        acquisition["revision"] != plan["revision"]
        or acquisition["weights_sha256"] != plan["weights_sha256"]
        or file_hashes(path)["model.safetensors"] != plan["weights_sha256"]
    ):
        raise ValueError("Local base model differs from pinned acquisition")
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation=plan["attention"],
    ).to(plan["device"])
    return model, tokenizer


class LocalPeftAdapter:
    def __init__(self, base, plan, *, adapter=None):
        self.base, self.plan, self.adapter = (
            Path(base),
            plan,
            Path(adapter) if adapter else None,
        )
        self.identity = digest(
            {
                "base": file_hashes(base),
                "adapter": file_hashes(adapter) if adapter else {},
            }
        )
        self.model = self.tokenizer = None

    def _complete(self, messages, config):
        if config.model_digest != self.identity or config.temperature != 0:
            raise ResponseFailure(
                "Local model identity or sampling drift",
                model_calls=0,
                input_tokens=0,
                output_tokens=0,
            )
        torch = environment(config.seed)
        load_duration = 0
        if self.model is None:
            started = time.perf_counter()
            self.model, self.tokenizer = load_local(self.base, self.plan)
            if self.adapter:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(
                    self.model, self.adapter, local_files_only=True
                )
            self.model.eval()
            load_duration = time.perf_counter() - started
        formatted = []
        for message in messages:
            if message["role"] == "tool":
                formatted.append(
                    {
                        "role": "user",
                        "content": "Tool "
                        + message["name"]
                        + " returned: "
                        + canonical(message["content"])
                        + ". Continue with a JSON reply.",
                    }
                )
            elif message.get("calls"):
                formatted.append(
                    {
                        "role": message["role"],
                        "content": canonical(
                            {"content": message["content"], "calls": message["calls"]}
                        ),
                    }
                )
            else:
                formatted.append(
                    {"role": message["role"], "content": message["content"]}
                )
        inputs = self.tokenizer.apply_chat_template(
            formatted,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.plan["device"])
        input_count = inputs["input_ids"].shape[1]
        if input_count > config.context_size:
            raise ResponseFailure(
                "Context exceeds declared bound",
                model_calls=0,
                input_tokens=0,
                output_tokens=0,
            )
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.max_output_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        generated = output[0, input_count:]
        usage = {
            "input_tokens": input_count,
            "output_tokens": len(generated),
            "model_calls": 1,
            "measurements": {
                "load_duration": load_duration,
                "prompt_eval_duration": 0,
                "eval_duration": time.perf_counter() - started,
            },
        }
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("Reply must be an object")
            return Reply.parse({**value, **usage})
        except (ValueError, TypeError, RuntimeFailure) as error:
            raise ResponseFailure("Malformed local model reply", **usage) from error

    async def complete(self, messages, config):
        return await asyncio.to_thread(self._complete, messages, config)

    def close(self):
        self.model = self.tokenizer = None
        gc.collect()
        import torch

        torch.cuda.empty_cache()


def train(base, data, plan, output):
    validate_data(data)
    if plan["epochs"] != 2 or plan["batch_size"] != 1 or plan["lora_rank"] != 4:
        raise ValueError("Training recipe exceeds reviewed bounds")
    from peft import LoraConfig, get_peft_model

    torch = environment(plan["seed"])
    meter = Telemetry(energy=EnergyCounter.discover())
    meter.start()
    losses, tokens = [], 0
    started = time.perf_counter()
    try:
        model, tokenizer = load_local(base, plan)
        model = get_peft_model(
            model,
            LoraConfig(
                r=plan["lora_rank"],
                lora_alpha=plan["lora_alpha"],
                lora_dropout=0,
                target_modules=plan["target_modules"],
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        model.train()
        load_seconds = time.perf_counter() - started
        trainable = [p for p in model.parameters() if p.requires_grad]
        parameter_count = sum(p.numel() for p in trainable)
        optimizer = torch.optim.AdamW(trainable, lr=plan["learning_rate"])
        gpu_start, gpu_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        gpu_start.record()
        optimization_started = time.perf_counter()
        for _ in range(plan["epochs"]):
            for example in data["examples"]:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": example["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                answer = tokenizer(example["response"], add_special_tokens=False)[
                    "input_ids"
                ] + [tokenizer.eos_token_id]
                sequence = prefix + answer
                if len(sequence) > plan["sequence_limit"]:
                    raise ValueError("Training example exceeds explicit token limit")
                inputs = torch.tensor([sequence], device=plan["device"])
                labels = torch.tensor(
                    [[-100] * len(prefix) + answer], device=plan["device"]
                )
                optimizer.zero_grad(set_to_none=True)
                loss = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    labels=labels,
                ).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                tokens += len(sequence)
        gpu_end.record()
        torch.cuda.synchronize()
        optimization_seconds = time.perf_counter() - optimization_started
        model.save_pretrained(output, safe_serialization=True)
        report = {
            "optimizer_steps": len(losses),
            "training_tokens": tokens,
            "losses": losses,
            "trainable_parameters": parameter_count,
            "model_load_seconds": load_seconds,
            "optimization_seconds": optimization_seconds,
            "gpu_interval_seconds": gpu_start.elapsed_time(gpu_end) / 1000,
            "gpu_scope": "CUDA event interval including host dispatch gaps",
            "weights": file_hashes(output),
        }
    finally:
        measurements, sources = meter.stop()
    report["measurements"], report["measurement_sources"] = measurements, sources
    del optimizer, model, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return report
