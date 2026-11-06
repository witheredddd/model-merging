import abc
import copy
from typing import OrderedDict, Union

import torch

from src.models.modeling import ImageEncoder
from src.utils.variables_and_paths import MODELS

_Checkpoint = Union[str, dict, torch.nn.Module]


def symmetric_difference(A, B):
    """Returns the symmetric difference between two lists."""
    return list(set(A) ^ set(B))


class _TaskVector(abc.ABC):
    def __init__(
        self,
        model_name,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        assert (
            model_name in MODELS
        ), f"Invalid model name: {model_name}. Valid models are: {MODELS}"
        self.model_name = model_name
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                # parse pretrained_checkpoint
                pretrained_state_dict = self._safe_load(pretrained_checkpoint)

                # parse finetuned_checkpoint
                finetuned_state_dict = self._safe_load(finetuned_checkpoint)

                if "model_name" in finetuned_state_dict.keys():
                    finetuned_state_dict.pop("model_name")

                # the final task vector is the difference between finetuned and pretrained vectors.
                assert (
                    pretrained_state_dict.keys() == finetuned_state_dict.keys()
                ), f"State dicts have different keys: {symmetric_difference(pretrained_state_dict.keys(), finetuned_state_dict.keys())}."
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype == torch.int64:
                        continue
                    if pretrained_state_dict[key].dtype == torch.uint8:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    def _safe_load(self, checkpoint):
        if isinstance(checkpoint, str):
            return self._load_checkpoint(checkpoint).state_dict()
        elif isinstance(checkpoint, dict):
            return checkpoint
        elif isinstance(checkpoint, torch.nn.Module):
            return checkpoint.state_dict()
        else:
            raise ValueError(f"Invalid type for checkpoint: {type(checkpoint)}")

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint) -> torch.nn.Module:
        """Load a checkpoint into a model."""
        raise NotImplementedError

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return self.__class__(vector=new_vector)

    def dot(self, other):
        """Dot product of two task vectors."""
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))


    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, args=None, clone_model=True):
        """
        Apply a task vector to a pretrained model.

        参数:
        pretrained_checkpoint:
            可以是
            - str: checkpoint 路径
            - dict: state_dict
            - torch.nn.Module: 已经预加载好的模型
        scaling_coef:
            全局缩放系数
        args:
            评估配置，若包含 device，则会把返回模型搬到该设备
        clone_model:
            当 pretrained_checkpoint 是预加载模型时，是否先 deepcopy 再加 task vector。
            对 MOEA 场景应保持 True，避免污染基座模型。
        """
        with torch.no_grad():
            # 1) 支持三种输入：路径 / state_dict / 预加载模型
            if isinstance(pretrained_checkpoint, str):
                pretrained_model = self._load_checkpoint(pretrained_checkpoint)
            elif isinstance(pretrained_checkpoint, torch.nn.Module):
                pretrained_model = (
                    copy.deepcopy(pretrained_checkpoint) if clone_model else pretrained_checkpoint
                )
            elif isinstance(pretrained_checkpoint, dict):
                pretrained_model = ImageEncoder(self.model_name)
                pretrained_model.load_state_dict(pretrained_checkpoint, strict=True)
            else:
                raise ValueError(
                    f"Invalid type for pretrained_checkpoint in apply_to: {type(pretrained_checkpoint)}"
                )
            # 2) 放到目标设备
            if args is not None and hasattr(args, "device"):
                pretrained_model = pretrained_model.to(args.device)

            pretrained_state_dict = pretrained_model.state_dict()
            new_state_dict = {}

            # 3) 执行 theta_new = theta_0 + scaling_coef * delta
            for key in pretrained_state_dict:
                base_param = pretrained_state_dict[key]

                # 如果 task vector 里没有这个 key，就保持原值
                if key not in self.vector:
                    new_state_dict[key] = base_param
                    continue

                task_delta = self.vector[key].to(
                    device=base_param.device,
                    dtype=base_param.dtype,
                )
                new_state_dict[key] = base_param + scaling_coef * task_delta

            pretrained_model.load_state_dict(new_state_dict, strict=True)
            pretrained_model.eval()
        return pretrained_model


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return ImageEncoder.load(self.model_name, checkpoint)

    def apply_to_nonlinear(self, pretrained_nonlinear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a nonlinear pretrained model."""
        # 对每一层参数执行： (\theta_{\text{new}} = \theta_0 + \text{scaling_coef} \cdot \Delta\theta)
        return self.apply_to(pretrained_nonlinear_checkpoint, scaling_coef)
