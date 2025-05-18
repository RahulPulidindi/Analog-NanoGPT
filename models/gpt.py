import math
from functools import reduce
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn
from tqdm import trange

from src.model.gpt_language_model.peft.lora import MergedLinear
from src.model.gpt_language_model.transformer_block import LayerNorm, TransformerBlock
from src.utils import log_error

# Import for Llama block
from fms.modules.positions import RotaryEmbedding
from fms.models.llama import LLaMABlock, LLaMAConfig


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_size: int,
        context_size: int,
        head_size: Optional[int],
        num_heads: int,
        feed_forward_scaling: int,
        num_layers: int,
        bias: bool,
        dropout: float,
        weight_tying: bool = True,
        weight_decay: Optional[float] = None,
        block_config: Optional[dict] = None,  # Configuration for different block types
    ) -> None:
        """Create Generative Pre-trained Transformer model (decoder part of transformer architecture).

        Parameters
        ----------
        vocab_size : int
            number of unique tokens in vocabulary
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : Optional[int]
            the size of output of self-attention
        num_heads : int
            how many self-attention heads to use
        feed_forward_scaling : int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `feed_forward_scaling` specifies by how much
        num_layers : int
            how many transformer blocks to use
        bias: bool
            whether to use bias or not: without bias might be a bit better and faster
        dropout : float
            how many connection between tokens are dropped during each forward pass
        weight_tying: bool
           Weight Tying improves the performance of language models by tying (sharing) the weights of the embedding and
           softmax layers. This method also massively reduces the total number of parameters in the language models that
           it is applied to.
           https://paperswithcode.com/method/weight-tying, by default True
        weight_decay: Optional[float]
            if provided will prepare parameters for optimizer
        block_config: Optional[dict]
            Configuration for different types of blocks. If None, uses all transformer blocks.
            Example format:
            {
                "transformer": {"num_blocks": 6},
                "mamba": {"num_blocks": 2},
                "llama": {"num_blocks": 2}
            }
        """
        super().__init__()

        # Validate block configuration
        if block_config is not None:
            total_blocks = sum(config["num_blocks"] for config in block_config.values())
            if total_blocks != num_layers:
                raise ValueError(
                    f"Total number of blocks in block_config ({total_blocks}) "
                    f"does not match num_layers ({num_layers})"
                )
            
            # Validate block types
            valid_block_types = {"transformer", "mamba", "llama"}
            invalid_types = set(block_config.keys()) - valid_block_types
            if invalid_types:
                raise ValueError(f"Invalid block types found: {invalid_types}. Valid types are: {valid_block_types}")
            
            # Validate block counts
            for block_type, config in block_config.items():
                if config["num_blocks"] < 0:
                    raise ValueError(f"Number of {block_type} blocks cannot be negative")
                if config["num_blocks"] == 0:
                    raise ValueError(f"Number of {block_type} blocks cannot be zero")

        self.vocab_size = vocab_size
        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.feed_forward_scaling = feed_forward_scaling
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.weigh_tying = weight_tying
        self.weight_decay = weight_decay
        self.block_config = block_config or {"transformer": {"num_blocks": num_layers}}

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.embeddings_size)
        # since attention doesn't have any notion of space and we want to use spatial information we need to implement
        # positional embeddings (they will encode relative position of each token)
        # positional embeddings knows how to encode position of last N (context_size) tokens
        self.positional_embedding_table = nn.Embedding(self.context_size, self.embeddings_size)
        self.embeddings_dropout = nn.Dropout(self.dropout)

        # Create rotary embeddings for Llama blocks
        self.rotary_emb = RotaryEmbedding(
            dim=self.embeddings_size // self.num_heads,
            max_position_embeddings=self.context_size,
            base=10000,
        )

        # Create blocks based on configuration
        self.blocks = nn.ModuleList()
        current_layer = 0
        
        for block_type, config in self.block_config.items():
            num_blocks = config["num_blocks"]
            for _ in range(num_blocks):
                if block_type == "transformer":
                    block = TransformerBlock(
                        embeddings_size=self.embeddings_size,
                        context_size=self.context_size,
                        head_size=self.head_size,
                        num_heads=self.num_heads,
                        bias=self.bias,
                        dropout=self.dropout,
                        feed_forward_scaling=self.feed_forward_scaling,
                        is_decoder=True,
                        use_causal_self_attention=True,
                    )
                elif block_type == "mamba":
                    from mamba_ssm.modules.block import MambaBlock
                    block = MambaBlock(
                        d_model=self.embeddings_size,
                        d_state=16,  # Default value, can be made configurable
                        d_conv=4,    # Default value, can be made configurable
                        expand=2,    # Default value, can be made configurable
                        bias=self.bias,
                        dropout=self.dropout,
                    )
                elif block_type == "llama":
                    # Create Llama config with our model's parameters
                    llama_config = LLaMAConfig(
                        src_vocab_size=self.vocab_size,
                        emb_dim=self.embeddings_size,
                        nheads=self.num_heads,
                        nlayers=1,  # We're creating one block at a time
                        p_dropout=self.dropout,
                        attn_bias=self.bias,
                        mlp_bias=self.bias,
                    )
                    block = LLaMABlock(
                        config=llama_config,
                        rotary_emb=self.rotary_emb,
                    )
                else:
                    raise ValueError(f"Unknown block type: {block_type}")
                
                self.blocks.append(block)
                current_layer += 1

        self.layer_norm_final = LayerNorm(self.embeddings_size, bias=self.bias)  # final layer norm
        self.language_model_head = nn.Linear(self.embeddings_size, self.vocab_size, bias=False)
        if self.weigh_tying:
            self.token_embedding_table.weight = self.language_model_head.weight

        self.apply(self.__init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith("projection.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.2 / math.sqrt(2 * self.num_layers))

        # configure parameters for optimizer that will be decay and that will not
        if self.weight_decay:
            self.optimizer_parameters = self.__optimizer_parameters(weight_decay=self.weight_decay)

        # report number of parameters
        logger.debug(
            "GPT language model is created with number of parameters: {:.2f} million".format(
                self.__get_parameters_number() / 1e6,
            ),
        )

    def __get_parameters_number(self, exclude_positional_embeddings: bool = True) -> int:
        # TODO: print total number of parameters and number of learnable parameters
        """Return total number of parameters of the model without counting parameters of positional embeddings."""
        params_count = sum(param.numel() for param in self.parameters())
        if exclude_positional_embeddings:
            params_count -= self.positional_embedding_table.weight.numel()
        return params_count

    def __init_weights(self, module: torch.nn.modules) -> None:
        """Initialize Embedding and Linear layers with a smaller std.

        By default weights of Embedding layer are initialized from normal distribution
        with zero mean and unit std ( N(0, 1) ).
        Weights for linear layer are initialized from uniform distribution from
        [-k**0.5, k**0.5], where k = 1 / in_features. Even with 128 features it will be
        [-0.09, 0.09].

        Parameters
        ----------
        module : torch.nn.modules
            module of the network
        """
        if isinstance(module, (nn.Embedding, nn.Linear)):
            # TODO: check different std init works better
            #   0.2 / sqrt ( 2 * number of transformer blocks)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: Tensor,
        *,
        inference: bool = False,
        kv_cache: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Do the whole forward pass for decoder part of transformer.

        Parameters
        ----------
        idx : Tensor
            tensor of size (batch, time-step) consisting of indices of tokens inside vocabulary
            for each time-step for each batch
        inference: bool
            during inference we don't care about all tokens but the very last one, so we can
            apply final language head only on the last token and save some computations
        kv_cache: None | list[Tensor]
            key-value cache, but only if not None; if None - it means that it's disabled;
            contains cache for keys and value from all previous steps

        Returns
        -------
        Tensor
            tensor of size (batch, time-step, vocabulary_size): logits for each token in vocabulary
            for each time-step for each batch, or the last one in case of inference
        """
        # batch, time-step
        B, T = idx.shape  # noqa: N806
        if self.context_size < T:
            msg = f"Cannot do forward pass on sequence of length {T}, "
            f"context size should less or equal to {self.context_size}"
            raise ValueError(msg)

        # obtain token embeddings and add positional information
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        # if kv_cache is provided and it's not an empty tensor
        if kv_cache is not None and kv_cache[0] is not None and kv_cache[0].numel():
            pos_idx = kv_cache[0].shape[-2]  # kv_cache of shape: num_layers * (2, B, nh, T, hs)
            positional_embeddings = self.positional_embedding_table.weight[None, pos_idx]  # (1, C)
        else:
            positional_embeddings = self.positional_embedding_table.weight[:T]  # (T, C)
        x = token_embeddings + positional_embeddings  # (B, T, C)
        x = self.embeddings_dropout(x)  # (B, T, C)

        # apply multiple blocks
        new_kv_cache = []
        kv_cache = kv_cache or [None] * len(self.blocks)
        
        # Create position IDs for Llama blocks
        position_ids = torch.arange(0, T, dtype=torch.long, device=x.device)
        if kv_cache is not None and kv_cache[0] is not None and kv_cache[0].numel():
            position_ids = position_ids + kv_cache[0].shape[-2]

        for i, (block, kv_cache_layer) in enumerate(zip(self.blocks, kv_cache)):
            # Handle different block types
            if isinstance(block, TransformerBlock):
                x, new_kv = block(x, kv_cache_layer)
            elif hasattr(block, 'ssm'):  # Mamba block
                # Mamba uses its own state management
                # The state is maintained internally in the block
                # We pass the current state through kv_cache_layer
                x, new_kv = block(
                    x,
                    past_key_value_state=kv_cache_layer,
                    use_cache=True,
                    cache_position=position_ids,
                    mask=None
                )
            elif isinstance(block, LLaMABlock):  # Llama block
                # Llama uses its own attention state management
                # The state includes both key and value caches
                # We pass the current state through kv_cache_layer
                x, new_kv = block(
                    x,
                    position_ids=position_ids,
                    past_key_value_state=kv_cache_layer,
                    use_cache=True
                )
            else:
                raise ValueError(f"Unknown block type: {type(block)}")
            
            new_kv_cache.append(new_kv)

        # apply final normalization and generate logits for each token in vocabulary
        x = self.layer_norm_final(x)  # (B, T, C)

        # during inference we don't need to encode all token predictions,
        # only the last one (newly generated)
        if inference:
            return (
                self.language_model_head(x[:, -1:, :]),  # (B, 1, vocab_size)
                new_kv_cache,  # num_layers * (2, B, num_heads, T, head_size)
            )
        return self.language_model_head(x)  # (B, T, vocab_size)

    def __optimizer_parameters(self, weight_decay: float) -> Tuple[dict, dict]:
        """Configure optimizer with weight decay for nn.Linear.

        Parameters
        ----------
        weight_decay : float
            weight decay (L2 penalty)

        Returns
        -------
        Tuple[dict, dict]
            list of two dictionaries, containing parameter names that will have weight decay
            and that will not accordingly

        Raises
        ------
        ValueError
            if model's layer are not in (nn.Linear, nn.LayerNorm, LayerNorm, nn.Embedding)
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay, no_decay = set(), set()
        expected_weight_modules = (nn.Linear, nn.LayerNorm, LayerNorm, nn.Embedding, MergedLinear)
        for pn, _ in self.named_parameters():
            # get the parent module by the parameter's name
            module = reduce(lambda module, key: getattr(module, key), pn.split(".")[:-1], self)
            if type(module) not in expected_weight_modules:
                log_error(f"Expected the module to be one of '{expected_weight_modules}', but got {type(module)}")
            if isinstance(module, nn.Linear) and pn.endswith("weight"):
                decay.add(pn)
            else:
                no_decay.add(pn)

        # create the pytorch optimizer object
        param_dict = dict(self.named_parameters())
        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Prepare tensors to be compatible with Pytorch's Cross-Entropy loss and applies it.

        Cross-Entropy expects to have features as the second-dimension, so we first need to
        transform logits in supported shape and in order to align targets tensor with logits tensor,
        we need to transform targets too.

        Parameters
        ----------
        logits : Tensor
            tensor with model's outputs
        targets : Tensor
            tensor with true labels

        Returns
        -------
        Tensor
            tensor with loss value (of how good model's predictions are)
        """
        B, T, C = logits.shape  # noqa: N806
        return F.cross_entropy(
            logits.view(B * T, C),
            targets.view(B * T),
        )

    @classmethod
    def from_pretrained(cls: "GPTLanguageModel", gpt2_type: str) -> "GPTLanguageModel":
        """Create GPT2 model with weights copied from Huggingface pretrained model.

        Parameters
        ----------
        gpt2_type : str
            GPT2 type: gpt2, gpt2-medium, gpt2-large and gpt2-xl are supported

        Returns
        -------
        GPTLanguageModel
            a model with pretrained weights

        Raises
        ------
        ValueError
            provided gpt2 type is not in the list of supported types
        ValueError
            Huggingface GPT2 config has different values for dropout
        ValueError
            mismatch number of keys/parameters between GPT and Huggingface's GPT2
        ValueError
            mismatch shape of a parameter between GPT and Huggingface's GPT2
        """
        # Notation:
        # target* | the model to which the weights are copied (this GPT implementation)
        # source* | the model from which the weight are copied (Huggingface GPT2 implementation)

        # huggingface transformers library is needed only in this method
        from transformers import GPT2Config, GPT2LMHeadModel

        # check that the gpt2 type is supported
        supported_types = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        if gpt2_type not in supported_types:
            log_error(f"Only '{supported_types}' are supported, but '{gpt2_type}' was provided.")

        # prepare config that will be passed into our GPT implementation
        source_config = GPT2Config.get_config_dict(gpt2_type)[0]
        # syncing argument names between our GPT implementation and from Huggingface
        target_config = {
            "vocab_size": source_config["vocab_size"],
            "embeddings_size": source_config["n_embd"],
            "context_size": source_config["n_ctx"],
            "num_layers": source_config["n_layer"],
            "num_heads": source_config["n_head"],
            "head_size": None,
            "feed_forward_scaling": 4,
            "bias": True,
        }
        # Dropouts for embedding, attention, residual, and summary in Huggingface implementation have to be identical
        dropouts = [(name, value) for name, value in source_config.items() if "dropout" in name]
        if any(dropouts[0][1] != x[1] for x in dropouts[1:]):
            log_error(f"All dropouts for GPT2 model should have had the same value, but in fact received '{dropouts}'")
        target_config["dropout"] = dropouts[0][1]

        # Instantiate GPT model and extract params
        logger.debug("Creating GPT model with parameters: {}".format(target_config))
        target_model = GPTLanguageModel(**target_config)
        # extract gpt model parameters into a variable
        target_state_dict = target_model.state_dict()
        # drop tril as it's a buffer (doesn't learn anything)
        target_state_dict_keys = [k for k in target_state_dict if not k.endswith(".self_attention.tril")]

        # create Huggingface pretrained GPT2 model
        logger.debug("Loading pretrained Huggingface model of size '{}' ...".format(gpt2_type))
        source_model = GPT2LMHeadModel.from_pretrained(gpt2_type)
        logger.debug("Huggingface model is loaded.")
        source_state_dict = source_model.state_dict()
        # skip bias as it's not a parameter
        source_state_dict_keys = [
            key for key in source_state_dict if not key.endswith((".attn.bias", ".attn.masked_bias"))
        ]

        if len(target_state_dict_keys) != len(source_state_dict_keys):
            log_error(f"Mismatch number of keys: {len(target_state_dict_keys)} != {len(source_state_dict_keys)}")

        # since names of layers are different for our implementation and the one from Huggingface,
        # we need to map them properly
        param_mapping = {
            "transformer": "",
            "wte": "token_embedding_table",
            "wpe": "positional_embedding_table",
            "h": "blocks",
            "attn": "self_attention",
            "c_attn": "causal_self_attention",
            "ln_1": "layer_norm_1",
            "ln_2": "layer_norm_2",
            "c_proj": "projection",
            "mlp": "feed_forward",
            "ln_f": "layer_norm_final",
            "lm_head": "language_model_head",
        }

        def sync_name(name: str) -> str:
            names_renamed = [param_mapping.get(n, n) for n in name.split(".")]
            return ".".join(filter(None, names_renamed))

        # in Huggingface implementation attention and feed forward use 'Conv1D',
        # while in this implementation - LinearLayer
        # that means that we can use weights for those layers, only we need to transpose them before copying
        to_transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

        # loading weights
        logger.debug("Starting copying weights from pretrained Huggingface model into our implementation ...")
        for source_key in source_state_dict_keys:
            # map param name from Hugginface notation to this implementation's notation
            target_key = sync_name(source_key)
            source_weights = source_state_dict[source_key]
            if source_key.endswith(to_transposed):
                source_weights = source_weights.t()
            if source_weights.shape != target_state_dict[target_key].shape:
                log_error(
                    f"Shape mismatch: shape of source '{source_weights.shape}' and destination - "
                    f"'{target_state_dict[target_key].shape}'",
                )
            with torch.no_grad():
                target_state_dict[target_key].copy_(source_weights)
        logger.debug("Weights are copied.")

        return target_model

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        use_kv_cache: bool,
        temperature: float = 1.0,
        top_k_logits: Optional[int] = None,
    ) -> Tensor:
        """Generate new character after the current one.

        Parameters
        ----------
        idx : Tensor
            index of the current character
        max_new_tokens : int
            number of characters to be generated
        use_kv_cache: bool
            use key-value cache for speed up token generation; if true the number of generated tokens
            should not be larger than context size of the model
        temperature : float, optional
            The temperature determines how greedy the generative model is:
            If the temperature is low, the probabilities to sample other but the class with the highest log probability
            will be small, and the model will probably output the most correct text, but rather boring, with small
            variation.
            If the temperature is high, the model can output, with rather high probability, other words than those with
            the highest probability. The generated text will be more diverse, but there is a higher possibility of
            grammar mistakes and generation of nonsense.
            https://ai.stackexchange.com/questions/32477/what-is-the-temperature-in-the-gpt-models, by default 1.0
        top_k_logits : Optional[int], optional
            only top K logits (with the highest value) will be kept, by default None

        Returns
        -------
        Tensor
            tensor containing indices of the provided characters and newly generated

        Raises
        ------
        ValueError
            if using key-value cache and the number of tokens to generate is larger that context size of the model
        """
        if use_kv_cache and (max_new_tokens + idx.shape[-1] - 1) > self.context_size:
            msg = (
                "With kv-cache the number of new tokens should not be greater than context size of the model "
                f"plus size of initial context, but was requested '{max_new_tokens}' new tokens "
                f"with initial context of size '{idx.shape[-1]}' and '{self.context_size}' context size of the model"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Initialize cache based on block types
        kv_cache = []
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                # Transformer blocks use a simple key-value cache format
                cache = torch.empty(2, 0, device=idx.device, dtype=idx.dtype)
            elif hasattr(block, 'ssm'):  # Mamba block
                # Mamba blocks maintain their own state internally
                # The state includes both the SSM state and the convolution state
                cache = None  # State will be initialized inside the block
            elif isinstance(block, LLaMABlock):  # Llama block
                # Llama blocks maintain their own attention state
                # The state includes both key and value caches
                cache = None  # State will be initialized inside the block
            else:
                raise ValueError(f"Unknown block type: {type(block)}")
            kv_cache.append(cache)

        for iteration in trange(max_new_tokens, ascii=True):
            # with kv-cache - use only last token, without - crop to the last block_size
            # also crop to the last block if idx provided with more than 1 token in the
            # beginning of token generation (start words)
            if not use_kv_cache or (iteration == 0 and idx.shape[-1] > 1):
                context = idx[:, -self.context_size :]
            else:
                context = idx[:, -1:]
            # get the predictions
            logits, kv_cache = self(
                context,
                inference=True,
                kv_cache=kv_cache if use_kv_cache else None,
            )  # (B, T, C), with inference=True -> (1, 1, C)
            # focus only on the last time step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # becomes (B, C)
            if top_k_logits:
                # topk returns rearranged tensor where the first column contains the highest values,
                # the last column - the smallest values from top K logits ...
                values, _ = torch.topk(logits, min(top_k_logits, logits.shape[-1]))
                # ... that's why we need to compare with the last column
                logits[logits < values[:, -1]] = float("-inf")  # `-1:` is to preserve dimensionality
            # apply softmax on the predictions to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx
