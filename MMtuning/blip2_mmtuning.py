import torch
import torch.nn as nn
from torchvision import models
from transformers import BertTokenizer
import contextlib
import string
import random
import copy
from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5TokenizerFast, T5Config, T5ForConditionalGeneration
from transformers import ViTModel
from .qformer import BertLMHeadModel


def device(a):
    return list(a.parameters())[0].device

def init_tokenizer(truncation_side="right"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",truncation_side=truncation_side)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"}) 
    return tokenizer

def disabled_train(a, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return a

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
    model.apply(_convert_weights_to_fp16)

class Blip2MMtuning(nn.Module):
    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        vit_model="google/vit-base-patch16-224",
        freeze_vit=True,
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        num_query_token=32,
        cross_attention_freq=2,
        num_few_shot_examples=0,
        few_shot_prob=0,
        num_features=1408,
        t5_model="t5-3b",
        freeze_qformer = True,
        llm_peft = True,
        dtype_llm = torch.bfloat16,
        dtype_vis = torch.float32,
        dtype_qf = torch.float32,
        truncation_side="left"
                ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer(truncation_side="left") 
        new_vocab = self.tokenizer.get_vocab()
        
        # vit model
        self.visual_encoder = ViTModel.from_pretrained(vit_model, add_pooling_layer=False)
        
        self.vit_config = self.visual_encoder.config
        self.fc1 = nn.Linear(self.vit_config.hidden_size, num_features)

        self.precision = vit_precision
        if self.precision== "fp16":
            convert_weights_to_fp16(self.visual_encoder)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval() 
            self.visual_encoder.train = disabled_train(self.visual_encoder.train)

        self.ln_vision = torch.nn.LayerNorm(num_features)

        

        

        ## qformer setting
        self.encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        self.encoder_config.encoder_width = num_features  
        self.encoder_config.add_cross_attention = True 
        self.encoder_config.cross_attention_freq = cross_attention_freq
        self.encoder_config.query_length = num_query_token 
        self.encoder_config.vocab_size = len(new_vocab)

        self.Qformer = BertLMHeadModel(config=self.encoder_config)

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer)) 
        self.Qformer.cls = None
        
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval() 

        query_tokens_em = nn.Parameter(
            torch.zeros(1, num_query_token, self.encoder_config.hidden_size) 
        ) 
        self.query_tokens = query_tokens_em.data.normal_(mean=0.0, std=self.encoder_config.initializer_range)
        

        


        ## google/T5
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config, torch_dtype=torch.bfloat16
        )
        


        if llm_peft:
            for name, param in self.t5_model.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16()
                
            # peft
            from peft import get_peft_model, MMtuningConfig
            # target_modules = ['q','k','v']

            num_layers = 24
            ## loraaa
            encoder_attention_target_modules = [
                (f'encoder.block.{block_idx}.layer.0.SelfAttention.q',
                 f'encoder.block.{block_idx}.layer.0.SelfAttention.k',
                 f'encoder.block.{block_idx}.layer.0.SelfAttention.v')
                for block_idx in range(num_layers)
            ]
            target_modules = [item for sublist in encoder_attention_target_modules for item in sublist]

            lora_config = MMtuningConfig(r=4, lora_alpha=8, target_modules=target_modules, bias="none")

            model_cp = copy.deepcopy(self.t5_model)
            self.t5_model = get_peft_model(model_cp, lora_config)


        else:
             for name, param in self.t5_model.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16()

            
       
        self.t5_proj = nn.Linear(
           self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.t5_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob
        
        
        self.qformer_text_input = qformer_text_input
        
        self.dtype_llm = dtype_llm
        self.dtype_vis = dtype_vis
        self.dtype_qf = dtype_qf
    
        
        
    def device(self):
        return list(self.parameters())[0].device
    
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype) 
        else:
            return contextlib.nullcontext() 
    
    def init_tokenizer(self, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer


    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')
        
        image = samples["image"]
        
        if image.device.type != 'cuda':
            image = image.to('cuda')
            

        with self.maybe_autocast(dtype=self.dtype_vis):
            vis_ouput = self.visual_encoder(image)
            vis_ouput_f = self.fc1(vis_ouput.last_hidden_state)
            image_embeds = self.ln_vision(vis_ouput_f)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(image.device) #展平向量
        
        
        
        with self.maybe_autocast(dtype=self.dtype_qf):
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    samples["text_input"], 
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device) 

                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device) 
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1) 
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        


        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        fs_embeds, fs_atts = None, None
        if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
            fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples['few_shot_samples'])

        with self.maybe_autocast(dtype=self.dtype_llm):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
                ).to(image.device)
            output_tokens = self.t5_output_tokenizer(
                    samples["text_output"],
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                
                
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
                )
                
            # encoder
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids) 
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

                
            if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)
                
            # decoder
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )

            loss = outputs.loss
            
            return outputs, loss
    


    def prepare_few_shot_embeds(self,samples):
        this_n_fs = random.choices(
            list(range(self.num_few_shot_examples + 1)),
            weights=[1 - self.few_shot_prob] + [self.few_shot_prob / self.num_few_shot_examples] * self.num_few_shot_examples
        )[0]

        if this_n_fs == 0:
            return None, None

        images = []
        text_input = []
        for sample in samples:
            for n in range(this_n_fs):
                images.append(sample['image'][n])
                text_input.append(sample['text_input'][n])
        images = torch.stack(images, dim=0)

        image = images

        with self.maybe_autocast(dtype=self.dtype_vis):
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        
        with self.maybe_autocast(dtype=self.dtype_qf):
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(image.device)
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    text_input,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask = Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

       
        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=self.dtype_llm):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        if this_n_fs > 1:
            encoder_atts = encoder_atts.reshape(encoder_atts.size(0) // this_n_fs, encoder_atts.size(1) * this_n_fs)
            inputs_embeds = inputs_embeds.reshape(inputs_embeds.size(0) // this_n_fs, inputs_embeds.size(1) * this_n_fs, inputs_embeds.size(2))

        return inputs_embeds, encoder_atts

    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        do_sample=False, 
        top_p=0.9,
        repetition_penalty=1.5, 
        length_penalty=1.0, 
        num_captions=1,
        temperature=1, 
    ):
        
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]
        if image.device.type != 'cuda':
            image = image.to('cuda')

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]
        
        query_tokens = self.query_tokens.expand(bs, -1, -1).to(image.device)

        ## vision_encoder

        with self.maybe_autocast(dtype=self.dtype_vis):
            vis_ouput = self.visual_encoder(image)
            vis_ouput_f = self.fc1(vis_ouput.last_hidden_state)
            image_embeds = self.ln_vision(vis_ouput_f)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        
        
        with self.maybe_autocast(dtype=self.dtype_qf):
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    prompt,
                    # samples["text_input"],
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
            
        input_tokens = self.t5_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)
        
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        
        with self.maybe_autocast(dtype=self.dtype_llm):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text