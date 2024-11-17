import re

from typing import List, Union
from loguru import logger

import torch.nn as nn


def get_module(model, name):
    """
    Get a specific module from the model by its name path.

    Args:
        model: PyTorch model
        name: Dot-separated path to the module, e.g., "model.layers.0"

    Returns:
        The requested PyTorch module

    Raises:
        AttributeError: If the specified module is not found
    """
    try:
        names = name.split(".")
        module = model
        for n in names:
            module = getattr(module, n)
        return module
    except AttributeError:
        raise AttributeError(f"Module {name} not found in model")


class LearnableICVInterventionLMM(nn.Module):
    def __init__(
        self,
        lmm: nn.Module,
        enable_intervention=True,
        intervention_layer: Union[int, List[int]] = None,
        layer_format: str = None,
        total_layers: int = None,
    ):
        super().__init__()
        self.lmm = lmm
        self.hooks = []

        if enable_intervention:
            self.total_layers = total_layers
            self.intervention_layers = self._prepare_layers(intervention_layer)
            self.intervention_layer_names = [
                layer_format.replace("<LAYER_NUM>", str(layer))
                for layer in self.intervention_layers
            ]
            self.layer_to_icv_index = {
                int(layer_id): int(icv_idx)
                for icv_idx, layer_id in enumerate(self.intervention_layers)
            }
            self.intervention_enabled = True
            logger.info(f"Intervention layers: {self.intervention_layer_names}")

    def _prepare_layers(self, layers):
        if layers == -1:
            return list(range(self.total_layers))
        return [layers] if isinstance(layers, int) else layers

    def _add_intervention_hooks(self, icv):
        def get_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states, *rest = output
                    shift = icv[:, self.layer_to_icv_index[layer_idx]].unsqueeze(dim=1)
                    shifted_states = hidden_states + shift
                    normalized_states = (
                        shifted_states
                        / shifted_states.norm(dim=-1, keepdim=True)
                        * hidden_states.norm(dim=-1, keepdim=True)
                    )
                    return (normalized_states,) + tuple(rest)
                else:
                    shift = icv[:, self.layer_to_icv_index[layer_idx]].unsqueeze(dim=1)
                    shifted_states = output + shift
                    normalized_states = (
                        shifted_states
                        / shifted_states.norm(dim=-1, keepdim=True)
                        * output.norm(dim=-1, keepdim=True)
                    )
                    return normalized_states

            return hook

        # Remove any existing hooks
        self.remove_hooks()

        # Add new hooks
        for layer_name in self.intervention_layer_names:
            module = get_module(self.lmm, layer_name)
            layer_idx = int(re.findall(r"\d+", layer_name)[0])
            hook = module.register_forward_hook(get_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @property
    def device(self):
        return next(self.lmm.parameters()).device

    def toggle_intervention(self, enable: bool):
        self.intervention_enabled = enable
        if not enable:
            self.remove_hooks()

    def forward(self, icv=None, *args, **kwargs):
        if icv is not None and self.intervention_enabled:
            self._add_intervention_hooks(icv)
            output = self.lmm(*args, **kwargs)
            # Optionally remove hooks immediately after forward pass
            # self.remove_hooks()
            return output
        return self.lmm(*args, **kwargs)

    def generate(self, icv=None, *args, **kwargs):
        if icv is not None and self.intervention_enabled:
            self._add_intervention_hooks(icv)
            output = self.lmm.generate(*args, **kwargs)
            # Optionally remove hooks immediately after generation
            # self.remove_hooks()
            return output
        return self.lmm.generate(*args, **kwargs)

    def __del__(self):
        self.remove_hooks()
