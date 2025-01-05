import re
from typing import List, Union
from loguru import logger

import torch
import torch.nn as nn


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
        self.intervention_enabled = enable_intervention
        self.current_icv = None
        self.hooks = []

        if enable_intervention:
            self.total_layers = total_layers
            self.intervention_layers = self._prepare_layers(intervention_layer)
            self.intervention_layer_names = [
                layer_format.replace("<LAYER_NUM>", str(layer))
                for layer in self.intervention_layers
            ]
            logger.info(
                f"The intervention_layer_names is {self.intervention_layer_names}"
            )
            self.layer_to_icv_index = {
                int(layer_id): int(icv_idx)
                for icv_idx, layer_id in enumerate(self.intervention_layers)
            }
            logger.info(f"The layer_to_icv_index is {self.layer_to_icv_index}")
            self._register_hooks()

    def _prepare_layers(self, layers):
        if layers == -1:
            return list(range(self.total_layers))
        return [layers] if isinstance(layers, int) else layers

    @property
    def device(self):
        return next(self.lmm.parameters()).device

    @property
    def intervention_status(self) -> bool:
        return self.intervention_enabled

    @intervention_status.setter
    def intervention_status(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("Intervention status must be a boolean value.")
        self.intervention_enabled = value

    def toggle_intervention(self, enable: bool):
        self.intervention_status = enable

    def _apply_intervention(self, hidden_states, layer_idx):
        """Apply ICV intervention to hidden states"""
        if not self.intervention_enabled or self.current_icv is None:
            return hidden_states

        shift = self.current_icv[:, self.layer_to_icv_index[layer_idx]].unsqueeze(dim=1)
        shifted_states = hidden_states + shift
        normalized_states = (
            shifted_states
            / shifted_states.norm(dim=-1, keepdim=True)
            * hidden_states.norm(dim=-1, keepdim=True)
        )
        return normalized_states

    def _get_layer_by_name(self, model, layer_name):
        """Get layer from model by name"""
        parts = layer_name.split('.')
        current = model
        for part in parts:
            current = getattr(current, part)
        return current

    def _intervention_hook(self, layer_idx):
        def hook(module, input_tensor, output):
            if isinstance(output, tuple):
                hidden_states, *rest = output
                hidden_states = self._apply_intervention(hidden_states, layer_idx)
                return (hidden_states,) + tuple(rest)
            return self._apply_intervention(output, layer_idx)
        return hook

    def _register_hooks(self):
        """Register forward hooks for intervention"""
        # Remove existing hooks if any
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        # Register new hooks
        for layer_name in self.intervention_layer_names:
            layer_idx = int(re.findall(r"\d+", layer_name)[0])
            mlp = self._get_layer_by_name(self.lmm, layer_name)
            hook = mlp.register_forward_hook(self._intervention_hook(layer_idx))
            self.hooks.append(hook)

    def forward(self, icv=None, *args, **kwargs):
        """
        Forward pass of the model with optional ICV intervention.

        Args:
            icv: Input control variable.
            *args: Variable length argument list of lmm.forward().
            **kwargs: Arbitrary keyword arguments of lmm.forward().

        Returns:
            The output of the model's forward pass.
        """
        self.current_icv = icv
        return self.lmm(*args, **kwargs)

    def generate(self, icv=None, *args, **kwargs):
        """
        Generate output using the specified ICV model with optional ICV intervention.

        Parameters:
            icv (ICVModel): The ICV model to use for generation.
            *args: Variable length argument list of lmm.generate().
            **kwargs: Arbitrary keyword arguments of lmm.generate().

        Returns:
            The generated output.
        """
        self.current_icv = icv
        return self.lmm.generate(*args, **kwargs)

    def __del__(self):
        # Clean up hooks when the module is deleted
        for hook in self.hooks:
            hook.remove() 