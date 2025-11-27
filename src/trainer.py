import os
from dataclasses import dataclass, field
from typing import Union, Any, Optional, Callable

import torch
from datasets import Dataset, IterableDataset
from peft import PeftConfig, PeftModel
from torch import nn
from transformers import TrainingArguments, DataCollator, PreTrainedModel, EvalPrediction, TrainerCallback, \
    PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_peft_available
from transformers.utils import logging
from trl import clone_chat_template
from trl.models import prepare_peft_model
from trl.trainer.base_trainer import BaseTrainer

logger = logging.get_logger(__name__)
logging.set_verbosity_info()
logging.enable_default_handler()
logging.enable_explicit_format()


@dataclass
class CustomTrainingArguments(TrainingArguments):
    r"""
    This class includes only the parameters that are specific to Custom training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        chat_template_path (`str`, *optional*):
            If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory
            or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must
            ensure that any special tokens referenced in the template are added to the tokenizer and that the model's
            embedding layer is resized accordingly.
    """
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If specified, sets the model's chat template. This can either be the path to a tokenizer (local "
                    "directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, "
                    "you must ensure that any special tokens referenced in the template are added to the tokenizer and "
                    "that the model's embedding layer is resized accordingly."
        },
    )


class CustomTrainer(BaseTrainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

            <Tip>

            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.

            </Tip>

        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `processing_class` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or tokenizer.
        train_dataset (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.

            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (Union[`torch.utils.data.Dataset`, dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
            This supersedes the `tokenizer` argument, which is now deprecated.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by [`Trainer`].
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use.
            Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module, None] = None,
            args: Optional[CustomTrainingArguments] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
            eval_dataset: Optional[Union[Dataset, dict[str, Dataset], "datasets.Dataset"]] = None,
            processing_class: Optional[
                Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
            ] = None,
            model_init: Optional[Callable[..., PreTrainedModel]] = None,
            compute_loss_func: Optional[Callable] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional["PeftConfig"] = None,

    ):
        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []
        # PEFT configuration and model wrapping
        if peft_config is not None:
            if added_tokens:
                # Ensure that the added tokens are trainable
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)

                # Ensure that the lm_head is trainable
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's "
                        "`modules_to_save`. As a result, the model may not learn to generate outputs with these new "
                        "tokens, leading to degraded generation quality. To fix this, add "
                        "`modules_to_save=['lm_head']` to your PEFT configuration."
                    )

                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")

        # In Prompt Tuning a small set of trainable virtual tokens (continuous prompt embeddings) is prepended to the
        # input. We store the number of these tokens so we can account for them correctly when calculating accuracy.
        self.num_virtual_tokens = 0

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)
            if model.active_adapter in model.peft_config:
                peft_model_config = model.peft_config[model.active_adapter]
                self.num_virtual_tokens = getattr(peft_model_config, "num_virtual_tokens", 0)

            # Log trainable vs total parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_percent = 100 * trainable_params / total_params
            logger.info(
                f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {trainable_percent:.4f}%"
            )

        super(CustomTrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def compute_loss(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            return_outputs: bool = False,
            num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if self.compute_loss_func is not None:
            raise ValueError(f"compute_loss_func must be set `None` but got {self.compute_loss_func}")

        few_shot_input_ids = inputs.pop("few_shot_input_ids")
        few_shot_attention_mask = inputs.pop("few_shot_attention_mask")
        candidate_indices = inputs.pop("candidate_indices")

        with torch.no_grad():
            with model.disable_adapter():
                icl_outputs = model(
                    **{"input_ids": few_shot_input_ids, "attention_mask": few_shot_attention_mask, "logits_to_keep": 1}
                )
                # Log the original logits shape
                logger.info(f"ICL output logits shape (before squeeze): {icl_outputs.logits.shape}")
                logger.info(f"ICL input_ids shape: {few_shot_input_ids.shape}")

                icl_logits = icl_outputs.logits.squeeze(1)

                logger.info(f"ICL logits shape (after squeeze): {icl_logits.shape}")

                # Filter logits to candidate indices only
                candidate_indices_device = candidate_indices.to(icl_logits.device)
                icl_logits_filtered = icl_logits[:, candidate_indices_device]

                logger.info(f"Candidate indices: {candidate_indices_device}")
                logger.info(f"ICL logits filtered shape: {icl_logits_filtered.shape}")

                icl_prob = nn.functional.softmax(icl_logits_filtered, dim=-1)

        zero_shot_input_ids = inputs.pop("zero_shot_input_ids")
        zero_shot_attention_mask = inputs.pop("zero_shot_attention_mask")

        tvp_outputs = model(
            **{"input_ids": zero_shot_input_ids, "attention_mask": zero_shot_attention_mask, "logits_to_keep": 1}
        )

        # Log TVP logits shape
        logger.info(f"TVP output logits shape (before squeeze): {tvp_outputs.logits.shape}")
        logger.info(f"TVP input_ids shape: {zero_shot_input_ids.shape}")

        tvp_logits = tvp_outputs.logits.squeeze(1)

        logger.info(f"TVP logits shape (after squeeze): {tvp_logits.shape}")

        # Filter TVP logits to candidate indices only
        tvp_logits_filtered = tvp_logits[:, candidate_indices_device]

        logger.info(f"TVP logits filtered shape: {tvp_logits_filtered.shape}")

        kl_loss = nn.functional.kl_div(
            nn.functional.log_softmax(tvp_logits_filtered, dim=-1),
            icl_prob,
            reduction="batchmean",
        )

        return (kl_loss, (icl_outputs, tvp_outputs)) if return_outputs else kl_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss")
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            with self.compute_loss_context_manager():
                num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
                loss, outputs = self.compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                )
            loss = loss.detach().mean()

        if prediction_loss_only:
            return loss, None, None
