# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose


class LoraaaLayer(BaseTunerLayer):
    # lora层加载方式
    # All names of layers that may contain (trainable) adapter weights
    
    #########  这地方要改！！！ #########
    #########  "lora_A_3", "lora_B_3", "lora_A_4", "lora_B_4",
    adapter_layer_names = ("lora_A_1", "lora_B_1", "lora_A_2", "lora_B_2", "lora_A_3", "lora_B_3", "lora_A_4", "lora_B_4",
                            "lora_embedding_A_1", "lora_embedding_B_1", "lora_embedding_A_2", "lora_embedding_B_2")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        # base_layer是要peft的model
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A_1 = nn.ModuleDict({}) # lora_encoder  self.lora_A_1[adapter_name]，找到对应的名字
        self.lora_B_1 = nn.ModuleDict({}) # lora_decoder
        self.lora_A_2 = nn.ModuleDict({}) # lora_encoder
        self.lora_B_2 = nn.ModuleDict({}) # lora_decoder
        self.lora_A_3 = nn.ModuleDict({}) # lora_encoder  self.lora_A_1[adapter_name]，找到对应的名字
        self.lora_B_3 = nn.ModuleDict({}) # lora_decoder
        self.lora_A_4 = nn.ModuleDict({}) # lora_encoder
        self.lora_B_4 = nn.ModuleDict({}) # lora_decoder

        # For Embedding layer
        self.lora_embedding_A_1 = nn.ParameterDict({})
        self.lora_embedding_B_1 = nn.ParameterDict({})
        self.lora_embedding_A_2 = nn.ParameterDict({})
        self.lora_embedding_B_2 = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        # 这个方法是一个递归的过程，会一直迭代获取 base_layer 的基本层
        #  base_layer的基本层是nn.Module中的一种，会不停的找
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

        ### moe的全局变量
        self.num_expert = 4
        self.router_img = nn.Linear(in_features=self.in_features, out_features=self.num_expert)
        self.router_text = nn.Linear(in_features=self.in_features, out_features=self.num_expert)

        nn.init.normal_(self.router_img.weight, mean=0, std=1e-5)
        nn.init.normal_(self.router_text.weight, mean=0, std=1e-5)

        # self.router_img = nn.Linear(in_features=self.in_features, out_features=self.num_expert, bias=False)
        # self.router_text = nn.Linear(in_features=self.in_features, out_features=self.num_expert, bias=False)

        # self.router_img = nn.Sequential(
        #     nn.Linear(in_features=self.in_features, out_features=self.num_expert),
        #     nn.LayerNorm(self.num_expert)
        # )
        # self.router_text = nn.Sequential(
        #     nn.Linear(in_features=self.in_features, out_features=self.num_expert),
        #     nn.LayerNorm(self.num_expert)
        # )



    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        # self.r 是一个字典，用于存储各个适配器的中间层大小，
        # self.lora_alpha 是一个字典，用于存储各个适配器的缩放系数。
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity() # 它实现了恒等映射，无参数

        # lora_dropout_layer的代码在上面
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        # 字典
        if r > 0:
            self.lora_A_1[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B_1[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            self.lora_A_2[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B_2[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            self.lora_A_3[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B_3[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            self.lora_A_4[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B_4[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name) # 此reset_lora_parameters函数在下面定义了

        weight = getattr(self.get_base_layer(), "weight", None)
        # 获取self.get_base_layer()基本层的，权重
        # weight = 权重

        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        # 作用：只是训练active_adapters所提到的adapter层
        self.set_adapter(self.active_adapters) # 在basetuner的class当中；
        ########
        ### active_adapters 要修改，不然识别不到可训练的层
        ########

    # 同上，只是针对conv2d
    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        if r > 0:
            kernel_size = base_layer.kernel_size
            stride = base_layer.stride
            padding = base_layer.padding
            self.lora_A_1[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
            self.lora_B_1[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
            self.lora_A_2[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
            self.lora_B_2[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
            # self.lora_A_3[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
            # self.lora_B_3[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
            # self.lora_A_4[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
            # self.lora_B_4[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)
        self.set_adapter(self.active_adapters)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        if r > 0:
            weight_A = torch.randn((r, self.in_features))
            weight_B = torch.randn((self.out_features, r))
            self.lora_embedding_A_1[adapter_name] = nn.Parameter(weight_A)
            self.lora_embedding_B_1[adapter_name] = nn.Parameter(weight_B)
            self.lora_embedding_A_2[adapter_name] = nn.Parameter(weight_A)
            self.lora_embedding_B_2[adapter_name] = nn.Parameter(weight_B)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)
        self.set_adapter(self.active_adapters)

    # 初始化loraaa的参数
    def reset_lora_parameters(self, adapter_name):
        ## 自己加的
        # nn.init.kaiming_uniform_(self.router_img.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.router_text.weight, a=math.sqrt(5))

        # or adapter_name in self.lora_A_3.keys() or adapter_name in self.lora_A_4.keys()
        if adapter_name in self.lora_A_1.keys() or adapter_name in self.lora_A_2.keys() or adapter_name in self.lora_A_3.keys() or adapter_name in self.lora_A_4.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_1[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_1[adapter_name].weight)

            nn.init.kaiming_uniform_(self.lora_A_2[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_2[adapter_name].weight)

            nn.init.kaiming_uniform_(self.lora_A_3[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_3[adapter_name].weight)

            nn.init.kaiming_uniform_(self.lora_A_4[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_4[adapter_name].weight)

        if adapter_name in self.lora_embedding_A_1.keys() or adapter_name in self.lora_embedding_A_2.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A_1[adapter_name])
            nn.init.normal_(self.lora_embedding_B_1[adapter_name])

            nn.init.zeros_(self.lora_embedding_A_2[adapter_name])
            nn.init.normal_(self.lora_embedding_B_2[adapter_name])

    # set_scale参数，影响 self.scaling
    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            # if active_adapter not in self.lora_A.keys():
            if active_adapter not in self.lora_A_1.keys() or active_adapter not in self.lora_A_2.keys() or active_adapter not in self.lora_A_3.keys() or active_adapter not in self.lora_A_4.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            # if active_adapter not in self.lora_A.keys():
            if  active_adapter not in self.lora_A_1.keys() or active_adapter not in self.lora_A_2.keys() or active_adapter not in self.lora_A_3.keys() or active_adapter not in self.lora_A_4.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraaaLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer, # 原模型的参数
        adapter_name: str, # 需要拟合的模型的层
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraaaLayer.__init__(self, base_layer)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        # 针对线性层
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

        # ### moe的全局变量
        # self.num_expert = 4
        # self.router_img = nn.Linear(in_features=self.in_features, out_features=self.num_expert)
        # self.router_text = nn.Linear(in_features=self.in_features, out_features=self.num_expert)


    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            # 已经融合的adapters(层)=merged_adapters
            # 为融合的adapters(层)=active_adapters
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            # if active_adapter in self.lora_A.keys():
            ## or active_adapter in self.lora_A_3.keys() or active_adapter in self.lora_A_4.keys()
            if active_adapter in self.lora_A_1.keys() or active_adapter in self.lora_A_2.keys() or active_adapter in self.lora_A_3.keys() or active_adapter in self.lora_A_4.keys():
                base_layer = self.get_base_layer() # 获得基础层
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    # orig_weights += self.get_delta_weight(active_adapter)

                    orig_weights += self.get_delta_weight(active_adapter, lora_A=self.lora_A_1, lora_B=self.lora_B_1)
                    orig_weights += self.get_delta_weight(active_adapter, lora_A=self.lora_A_2, lora_B=self.lora_B_2)
                    orig_weights += self.get_delta_weight(active_adapter, lora_A=self.lora_A_3, lora_B=self.lora_B_3)
                    orig_weights += self.get_delta_weight(active_adapter, lora_A=self.lora_A_4, lora_B=self.lora_B_4)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    # get_delta_weight函数代码在下面
                    # 得到的是lora的权重
                    # base_layer.weight.data += self.get_delta_weight(active_adapter) 

                    if active_adapter in self.lora_A_1.keys():
                        base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_1, lora_B=self.lora_B_1)
                    elif active_adapter in self.lora_A_2.keys():
                        base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_2, lora_B=self.lora_B_2)
                    elif active_adapter in self.lora_A_3.keys():
                        base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_3, lora_B=self.lora_B_3)
                    elif active_adapter in self.lora_A_4.keys():
                        base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_4, lora_B=self.lora_B_4)
                    
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            # 如果没有需要融合的东西，bool->从list中有无东西来判断
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop() # 删去已经融合的层
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A_1.keys():
                base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_1, lora_B=self.lora_B_1)
            elif active_adapter in self.lora_A_2.keys():
                base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_2, lora_B=self.lora_B_2)
            elif active_adapter in self.lora_A_3.keys():
                base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_3, lora_B=self.lora_B_3)
            elif active_adapter in self.lora_A_4.keys():
                base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_4, lora_B=self.lora_B_4)

            ## original:
            # if active_adapter in self.lora_A.keys():
                # self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    # def get_delta_weight(self, adapter) -> torch.Tensor:
    #     """
    #     Compute the delta weight for the given adapter.

    #     Args:
    #         adapter (str):
    #             The name of the adapter for which the delta weight should be computed.
    #     """
    #     device = self.lora_B[adapter].weight.device
    #     dtype = self.lora_B[adapter].weight.dtype

    #     # In case users wants to merge the adapter weights that are in
    #     # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    #     # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    #     cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    #     weight_A = self.lora_A[adapter].weight
    #     weight_B = self.lora_B[adapter].weight

    #     if cast_to_fp32:
    #         weight_A = weight_A.float()
    #         weight_B = weight_B.float()

    #     output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

    #     if cast_to_fp32:
    #         output_tensor = output_tensor.to(dtype=dtype)

    #         # cast back the weights
    #         self.lora_A[adapter].weight.data = weight_A.to(dtype)
    #         self.lora_B[adapter].weight.data = weight_B.to(dtype)

    #     return output_tensor

    def get_delta_weight(self, adapter, lora_A, lora_B) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
            lora_A (nn.ModuleDict):
                Dictionary containing lora_A weights for the specified adapter.
            lora_B (nn.ModuleDict):
                Dictionary containing lora_B weights for the specified adapter.
        """
        device = lora_B[adapter].weight.device
        dtype = lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = lora_A[adapter].weight
        weight_B = lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            lora_A[adapter].weight.data = weight_A.to(dtype)
            lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor
    

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        # print("raw: ",x.size())
        
        # Split x based on sequence length
        # seq_length = x.shape[1]

        # x_img = x[:, :32, :]
        # x_text = x[:, 32:, :]

        x_img = x[:, :4, :]
        x_text = x[:, 4:, :]

        # print(f"image token:{x_img.size()}")
        # print(f"text token:{x_text.size()}")
        
        
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            
            # for active_adapter in self.active_adapters:
            #     if active_adapter not in self.lora_A.keys():
            #         continue
                # lora_A = self.lora_A[active_adapter]
                # lora_B = self.lora_B[active_adapter]
                # dropout = self.lora_dropout[active_adapter]
                # scaling = self.scaling[active_adapter]
                # x = x.to(lora_A.weight.dtype)
                # result += lora_B(lora_A(dropout(x))) * scaling
            
            lora_outputs = []
            

            # 计算第一个 Lora 层的输出
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A_1.keys():
                    continue
                lora_A_1 = self.lora_A_1[active_adapter]
                lora_B_1 = self.lora_B_1[active_adapter]
                dropout_1 = self.lora_dropout[active_adapter]
                scaling_1 = self.scaling[active_adapter]
                x = x.to(lora_A_1.weight.dtype)
                lora_outputs.append(lora_B_1(lora_A_1(dropout_1(x))) * scaling_1)
            
            # 计算第二个 Lora 层的输出
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A_2.keys():
                    continue
                lora_A_2 = self.lora_A_2[active_adapter]
                lora_B_2 = self.lora_B_2[active_adapter]
                dropout_2 = self.lora_dropout[active_adapter]
                scaling_2 = self.scaling[active_adapter]
                x = x.to(lora_A_2.weight.dtype)
                lora_outputs.append(lora_B_2(lora_A_2(dropout_2(x))) * scaling_2)

            ## 计算第三个 Lora 层的输出
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A_3.keys():
                    continue
                lora_A_3 = self.lora_A_3[active_adapter]
                lora_B_3 = self.lora_B_3[active_adapter]
                dropout_3 = self.lora_dropout[active_adapter]
                scaling_3 = self.scaling[active_adapter]
                x = x.to(lora_A_3.weight.dtype)
                lora_outputs.append(lora_B_3(lora_A_3(dropout_3(x))) * scaling_3)

            # 计算第四个 Lora 层的输出
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A_4.keys():
                    continue
                lora_A_4 = self.lora_A_4[active_adapter]
                lora_B_4 = self.lora_B_4[active_adapter]
                dropout_4 = self.lora_dropout[active_adapter]
                scaling_4 = self.scaling[active_adapter]
                x= x.to(lora_A_4.weight.dtype)
                lora_outputs.append(lora_B_4(lora_A_4(dropout_4(x))) * scaling_4)

            # 取两个 Lora 层的输出的均值
            # if lora_outputs:
            #     result += torch.stack(lora_outputs).mean(dim=0)
            
            # LORA 向量进行整合
            # [batch, seq_len, num_experts, hid_dim]
            expert_tensors = [torch.unsqueeze(tensor, dim=2) for tensor in lora_outputs]
            expert_tensor = torch.cat(expert_tensors, dim=2)

            # mean掉seq_len
            # Average along the first dimension (seq_len)
            # [batch, hid_dim]
            
            """
            从如何得到两个router
            """
            ### 这块
            ### x_img_avg 和 x_text_avg 分别做cross attention += 得到的attention加到自己
            ### new img 和 new text 获得彼此的信息
            x_img_avg = torch.mean(x_img, dim=1, keepdim=False)
            x_text_avg = torch.mean(x_text, dim=1, keepdim=False)

            # 将第二个维度变成只有2个数，并且每个数等于0.5
            # router_img = torch.stack([torch.ones_like(x_img_avg[:, 0]) * 0.5] * 2, dim=1)
            # router_text = torch.stack([torch.ones_like(x_text_avg[:, 0]) * 0.5] * 2, dim=1)
                
            # 新建router_img和router_text
            # [batch, num_experts]
            router_img = self.router_img(x_img_avg)
            router_text = self.router_text(x_text_avg)
            # router_img = nn.functional.softmax(self.router_img(x_img_avg), dim=-1)  ## softmax
            # router_text = nn.functional.softmax(self.router_text(x_text_avg), dim=-1)  ## softmax

            # 计算img_tensor和text_tensor
            ## 变成[batch, seq_len, hid_dim]
            img_tensor = torch.einsum('be,bsed->bsd', router_img, expert_tensor[:, :32, :, :])
            text_tensor = torch.einsum('be,bsed->bsd', router_text, expert_tensor[:, 32:, :, :])

            # mix_router = mean一下router_img＋router_text 
            """
            20 epochs or 30 epochs
            """

            """
            两个router如何结合的
            """

            # 这个20 vs 16 lora*epoch
            # result_tensor = mix_router * expert_tensor

            """
            router 只有一个, img和text的router: average一下;
            手动切开；前后一定有关系（说不清楚）
            img1 和 text2
            """

            # img_tensor = torch.einsum('be,bsed->bsd', router_img, expert_tensor[:, :4, :, :])
            # text_tensor = torch.einsum('be,bsed->bsd', router_text, expert_tensor[:, 4:, :, :])

            result += torch.cat([img_tensor, text_tensor], dim=1) 
            
            # result += result_tensor
            
            ## seq_length mean一下
            ## 不消去专家数量，router = [batch, num_experts]， 每个sample在experts上的分配向量
            ## router [batch, num_experts] * lora [batch, seq_len, num_experts, hid_dim]
            ## img和text router分别负责不同的部分，和lora相乘

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loraaa." + rep



class Embedding(nn.Module, LoraaaLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraaaLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A_1.keys() or active_adapter in self.lora_embedding_A_2.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.copy()
                    # orig_weights += self.get_delta_weight(active_adapter)

                    orig_weights += self.get_delta_weight(active_adapter, lora_embedding_A=self.lora_embedding_A_1, lora_B=self.lora_embedding_B_1)
                    orig_weights += self.get_delta_weight(active_adapter, lora_embedding_A=self.lora_embedding_A_2, lora_B=self.lora_embedding_B_2)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    # base_layer.weight.data += self.get_delta_weight(active_adapter)
                    
                    if active_adapter in self.lora_embedding_A_1.keys():
                        base_layer.weight.data += self.get_delta_weight(active_adapter, lora_embedding_A=self.lora_embedding_A_1, lora_B=self.lora_embedding_B_1)
                    elif active_adapter in self.lora_embedding_A_2.keys():
                        base_layer.weight.data += self.get_delta_weight(active_adapter, lora_embedding_A=self.lora_embedding_A_2, lora_B=self.lora_embedding_B_2)
                    
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A_1.keys() or active_adapter in self.lora_embedding_A_2.keys():
                # self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)
                base_layer = self.get_base_layer()
                base_layer.weight.data -= self.get_delta_weight(active_adapter, lora_embedding_A=self.lora_embedding_A_1, lora_B=self.lora_embedding_B_1)
                base_layer.weight.data -= self.get_delta_weight(active_adapter, lora_embedding_A=self.lora_embedding_A_2, lora_B=self.lora_embedding_B_2)


    # def get_delta_weight(self, adapter) -> torch.Tensor:
    #     """
    #     Compute the delta weight for the given adapter.

    #     Args:
    #         adapter (str):
    #             The name of the adapter for which the delta weight should be computed.
    #     """
    #     device = self.lora_embedding_B[adapter].device
    #     dtype = self.lora_embedding_A[adapter].dtype

    #     # In case users wants to merge the adapter weights that are in
    #     # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    #     # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    #     cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    #     weight_A = self.lora_embedding_A[adapter]
    #     weight_B = self.lora_embedding_B[adapter]

    #     if cast_to_fp32:
    #         weight_A = weight_A.float()
    #         weight_B = weight_B.float()

    #     output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

    #     if cast_to_fp32:
    #         output_tensor = output_tensor.to(dtype=dtype)

    #         # cast back the weights
    #         self.lora_embedding_A[adapter] = weight_A.to(dtype)
    #         self.lora_embedding_B[adapter] = weight_B.to(dtype)

    #     return output_tensor

    def get_delta_weight(self, adapter,lora_embedding_A, lora_embedding_B) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = lora_embedding_B[adapter].device
        dtype = lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = lora_embedding_A[adapter]
        weight_B = lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            lora_embedding_A[adapter] = weight_A.to(dtype)
            lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            # for active_adapter in self.active_adapters:
            #     if active_adapter not in self.lora_embedding_A:
            #         continue
            #     embedding_A = self.lora_embedding_A[active_adapter].T
            #     embedding_B = self.lora_embedding_B[active_adapter].T
            #     scaling = self.scaling[active_adapter]
            #     after_A = self._embed(x, embedding_A)
            #     result += (after_A @ embedding_B) * scaling

            averaged_output = torch.zeros_like(result)
            count = 0
            # 计算第一个 Lora 层的输出
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A_1:
                    continue
                embedding_A_1 = self.lora_embedding_A_1[active_adapter].T
                embedding_B_1 = self.lora_embedding_B_1[active_adapter].T
                scaling_1 = self.scaling_1[active_adapter]
                after_A_1 = self._embed(x, embedding_A_1)
                result_1 = (after_A_1 @ embedding_B_1) * scaling_1

            # 计算第二个 Lora 层的输出
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A_2.keys():
                    continue
                embedding_A_2 = self.lora_embedding_A_2[active_adapter].T
                embedding_B_2 = self.lora_embedding_B_2[active_adapter].T
                scaling_2 = self.scaling_2[active_adapter]
                after_A_2 = self._embed(x, embedding_A_2)
                result_2 = (after_A_2 @ embedding_B_2) * scaling_2

            # 求结果的均值
            averaged_output += (result_1 + result_2) / 2
            count += 1

            # 如果有有效的适配器，则取均值
            if count > 0:
                averaged_output /= count
                result += averaged_output

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loraaa." + rep


class Conv2d(nn.Module, LoraaaLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraaaLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    # def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
    #     """
    #     Merge the active adapter weights inside the base weights

    #     Args:
    #         safe_merge (`bool`, *optional*):
    #             If True, the merge operation will be performed in a copy of the original weights and check for NaNs
    #             before merging the weights. This is useful if you want to check if the merge operation will produce
    #             NaNs. Defaults to `False`.
    #         adapter_names (`List[str]`, *optional*):
    #             The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
    #             to `None`.
    #     """
    #     if self.merged:
    #         warnings.warn(
    #             f"Already following adapters were merged {','.join(self.merged_adapters)}. "
    #             f"You are now additionally merging {','.join(self.active_adapters)}."
    #         )

    #     if adapter_names is None:
    #         adapter_names = self.active_adapters

    #     for active_adapter in adapter_names:
    #         if active_adapter in self.lora_A.keys():
    #             base_layer = self.get_base_layer()
    #             if safe_merge:
    #                 # Note that safe_merge will be slower than the normal merge
    #                 # because of the copy operation.
    #                 orig_weights = base_layer.weight.data.copy()
    #                 orig_weights += self.get_delta_weight(active_adapter)

    #                 if not torch.isfinite(orig_weights).all():
    #                     raise ValueError(
    #                         f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
    #                     )
    #                 base_layer.weight.data = orig_weights
    #             else:
    #                 base_layer.weight.data += self.get_delta_weight(active_adapter)
    #             self.merged_adapters.append(active_adapter)
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A_1.keys() or active_adapter in self.lora_A_2.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter, lora_A_1=self.lora_A_1, lora_B_1=self.lora_B_1)
                    orig_weights += self.get_delta_weight(active_adapter, lora_A_2=self.lora_A_2, lora_B_2=self.lora_B_2)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A_1=self.lora_A_1, lora_B_1=self.lora_B_1)
                    base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A_2=self.lora_A_2, lora_B_2=self.lora_B_2)
                self.merged_adapters.append(active_adapter)

    # def unmerge(self) -> None:
    #     if not self.merged:
    #         warnings.warn("Already unmerged. Nothing to do.")
    #         return
    #     while len(self.merged_adapters) > 0:
    #         active_adapter = self.merged_adapters.pop()
    #         if active_adapter in self.lora_A.keys():
    #             self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop() # 删去以及融合的层
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A_1.keys():
                base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_1, lora_B=self.lora_B_1)
            elif active_adapter in self.lora_A_2.keys():
                base_layer.weight.data += self.get_delta_weight(active_adapter, lora_A=self.lora_A_2, lora_B=self.lora_B_2)

    # def get_delta_weight(self, adapter) -> torch.Tensor:
    #     """
    #     Compute the delta weight for the given adapter.

    #     Args:
    #         adapter (str):
    #             The name of the adapter for which the delta weight should be computed.
    #     """
    #     device = self.lora_B[adapter].weight.device
    #     dtype = self.lora_A[adapter].weight.dtype

    #     # In case users wants to merge the adapter weights that are in
    #     # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    #     # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    #     cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    #     weight_A = self.lora_A[adapter].weight
    #     weight_B = self.lora_B[adapter].weight

    #     if cast_to_fp32:
    #         weight_A = weight_A.float()
    #         weight_B = weight_B.float()

    #     # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
    #     if self.get_base_layer().weight.size()[2:4] == (1, 1):
    #         # conv2d 1x1
    #         output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
    #             3
    #         ) * self.scaling[adapter]
    #     else:
    #         # conv2d 3x3
    #         output_tensor = (
    #             F.conv2d(
    #                 weight_A.permute(1, 0, 2, 3),
    #                 weight_B,
    #             ).permute(1, 0, 2, 3)
    #             * self.scaling[adapter]
    #         )

    #     if cast_to_fp32:
    #         output_tensor = output_tensor.to(dtype=dtype)

    #         # cast back the weights
    #         self.lora_A[adapter].weight.data = weight_A.to(dtype)
    #         self.lora_B[adapter].weight.data = weight_B.to(dtype)

    #     return output_tensor

    def get_delta_weight(self, adapter, lora_A, lora_B) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = lora_B[adapter].weight.device
        dtype = lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = lora_A[adapter].weight
        weight_B = lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            lora_A[adapter].weight.data = weight_A.to(dtype)
            lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor
    

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

        #     for active_adapter in self.active_adapters:
        #         if active_adapter not in self.lora_A.keys():
        #             continue
        #         lora_A = self.lora_A[active_adapter]
        #         lora_B = self.lora_B[active_adapter]
        #         dropout = self.lora_dropout[active_adapter]
        #         scaling = self.scaling[active_adapter]
        #         x = x.to(lora_A.weight.dtype)
        #         result += lora_B(lora_A(dropout(x))) * scaling

        # result = result.to(previous_dtype)
        # return result

            lora_outputs = []

            # 计算第一个 Lora 层的输出
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A_1.keys():
                    continue
                lora_A_1 = self.lora_A_1[active_adapter]
                lora_B_1 = self.lora_B_1[active_adapter]
                dropout_1 = self.lora_dropout[active_adapter]
                scaling_1 = self.scaling[active_adapter]
                x = x.to(lora_A_1.weight.dtype)
                lora_outputs.append(lora_B_1(lora_A_1(dropout_1(x))) * scaling_1)

            # 计算第二个 Lora 层的输出
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A_2.keys():
                    continue
                lora_A_2 = self.lora_A_2[active_adapter]
                lora_B_2 = self.lora_B_2[active_adapter]
                dropout_2 = self.lora_dropout[active_adapter]
                scaling_2 = self.scaling[active_adapter]
                x = x.to(lora_A_2.weight.dtype)
                lora_outputs.append(lora_B_2(lora_A_2(dropout_2(x))) * scaling_2)

            # 取两个 Lora 层的输出的均值
            if lora_outputs:
                result += torch.stack(lora_outputs).mean(dim=0)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loraaa." + rep
