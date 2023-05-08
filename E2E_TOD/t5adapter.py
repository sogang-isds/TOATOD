import torch
from torch import nn
from torch.nn import Parameter
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm
from logging import getLogger
logger = getLogger(__name__)

class AdapterLayer(nn.Module):
    def __init__(self, dim, down_dim, norm=None):
        super().__init__()

        self.dim = dim
        self.down_dim = down_dim
        self.dropout = nn.Dropout(0.2)
        self.down = nn.Linear(dim, down_dim)
        self.relu = nn.ReLU()
        self.up = nn.Linear(down_dim, dim)
        if norm is not None:
            self.layer_norm = T5LayerNorm(dim)
            self.layer_norm.weight = Parameter(norm.clone())

    def forward(self, inputs):
        x = self.dropout(inputs)
        x = self.down(inputs)
        x = self.relu(x)
        x = self.up(x)
        x += inputs
        x = self.layer_norm(x)
        return x


class TaskOptimizedAdapter(nn.Module):
    def __init__(self, adapter_type, adapter_config, task: list, norm=None):
        super().__init__()
        self.toa = nn.ModuleDict({i: adapter_type(**adapter_config, norm=norm) for i in task})
        self.task = 'nlu'

    def forward(self, inputs):
        return self.toa[self.task](inputs)


class TaskOptimizedModuleList(nn.ModuleList):
    def __init__(self, modules):
        super().__init__()
        self += nn.ModuleList(modules)
        self.task = 'nlu'
    # freeze pretrained parameter and other task adapters of all blocks
    def freeze_pretrained(self, task):
        self.task = task
        for module in self:
            module.task = task
            module.freeze_pretrained(task)


class T5AdapterBlock(T5Block):
    def __init__(self, block, config, adapter_type, adapter_config, task: list):
        super().__init__(config)
        self.layer = block.layer
        self.is_decoder = block.is_decoder
        for layer in self.layer:
            if 'layer_norm' in layer._modules:
                norm = layer.layer_norm.weight.clone()
                del layer.layer_norm
            else:
                norm = None
            layer.layer_norm = TaskOptimizedAdapter(adapter_type, adapter_config, task, norm)
            layer._modules.move_to_end('dropout')
        self.task = 'nlu'

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs

    # freeze pretrained params and other task adapters in the block
    def freeze_pretrained(self, task):
        assert task in ['all', 'nlu', 'dst', 'policy', 'nlg']

        if task == 'all':
            for layer in self.layer:
                for params in layer.layer_norm.parameters():
                    params.requires_grad = True
        else:
            for layer in self.layer:
                layer.layer_norm.task = task
                for params in layer.layer_norm.toa[task].parameters():
                    params.requires_grad = True

    def set_layer_task(self, task):
        for layer in self.layer:
            layer.layer_norm.task = task

# add adapter to the Transformer blocks in the model
def add_adapter(model, adapter_type, adapter_config, tasks):
    model.encoder.block = TaskOptimizedModuleList(
        [T5AdapterBlock(block, model.encoder.config, adapter_type, adapter_config, tasks) for block in
         model.encoder.block])
    model.decoder.block = TaskOptimizedModuleList(
        [T5AdapterBlock(block, model.decoder.config, adapter_type, adapter_config, tasks) for block in
         model.decoder.block])
    return model


# freeze pretrained parameter & other task adapters
def set_task_for_train(model, task):
    for params in model.parameters():
        params.requires_grad = False
    model.encoder.block.freeze_pretrained(task)
    model.decoder.block.freeze_pretrained(task)
    return model


def set_task_for_inference(model, task):
    for block in model.encoder.block:
        block.task = task
        block.set_layer_task(task)
    for block in model.decoder.block:
        block.task = task
        block.set_layer_task(task)
    return model

def copy_weight(target_model, reference_model, task):
    reference_encoder, reference_decoder = reference_model.encoder, reference_model.decoder

    for block, ref_block in zip(target_model.encoder.block,reference_encoder.block):
        for layer, ref_layer in zip(block.layer, ref_block.layer):
            for params, ref_params in zip(layer.layer_norm.toa[task].parameters(), ref_layer.layer_norm.toa[task].parameters()):
                params.data.copy_(ref_params.data)
    for block, ref_block in zip(target_model.decoder.block, reference_decoder.block):
        for layer, ref_layer in zip(block.layer, ref_block.layer):
            for params, ref_params in zip(layer.layer_norm.toa[task].parameters(), ref_layer.layer_norm.toa[task].parameters()):
                params.data.copy_(ref_params.data)

    return target_model

if __name__ == '__main__':
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    # add adapter to the model
    model = add_adapter(model, AdapterLayer, {'dim':1024, 'down_dim':256},['nlu', 'dst', 'policy', 'nlg'])
    print(model)