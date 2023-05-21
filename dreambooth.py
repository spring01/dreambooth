#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Haichen Li. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

import xformers



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('description', help='Description of the dreambooth instance.')
    parser.add_argument('workspace', nargs='?', default='dreambooth-diffuser',
                        help='The output directory where the model will be written.')
    parser.add_argument('--instance_data', default='instance_data',
                        help='Subdirectory name under workspace containing instance data.')
    parser.add_argument('--class_data', default='class_data',
                        help='Subdirectory name under workspace containing class data.')
    parser.add_argument('--model_dir', default='model',
                        help='Subdirectory name under workspace to write the trained model to.')
    args = parser.parse_args()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    train(args.description, args.workspace, args.instance_data, args.class_data, args.model_dir)


def default_optimizer(parameters, lr):
    return CPUSGD(parameters, lr=lr, momentum=0.9)


def train(description, workspace,
          instance_data='instance_data',
          class_data='class_data',
          model_dir='model',
          pretrained='stabilityai/stable-diffusion-2-base',
          num_class_images=200,
          sample_batch_size=1,
          train_batch_size=2,
          num_train_epochs=4,
          gradient_accumulation_steps=1,
          device='cuda',
          torch_dtype=torch.float16,
          optimizer_class=default_optimizer,
          learning_rate=5e-2):
    logger = logging.getLogger(__name__)
    instance_prompt = f'a photo of sks {description}'
    class_prompt = f'a photo of {description}'

    with open(os.path.join(workspace, 'description.txt'), 'w') as fp:
        fp.write(description)

    # Generate class images.
    class_data_dir = os.path.join(workspace, class_data)
    class_images_dir = Path(class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < num_class_images:
        pipeline = DiffusionPipeline.from_pretrained(pretrained, torch_dtype=torch_dtype, safety_checker=None)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.unet.enable_xformers_memory_efficient_attention()

        num_new_images = num_class_images - cur_class_images
        logger.info(f'Number of class images to sample: {num_new_images}.')

        sample_dataset = PromptDataset(class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

        pipeline.to(device)

        for example in tqdm(sample_dataloader, desc='Generating class images'):
            images = pipeline(example['prompt']).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained, subfolder='tokenizer', use_fast=False)

    # Load scheduler and models
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained, subfolder='scheduler')
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder='vae')
    model = TextEncoderUnet(pretrained)

    vae.requires_grad_(False)

    model.unet.enable_xformers_memory_efficient_attention()

    model.to(device, dtype=torch_dtype)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    # Dataset and DataLoaders creation:
    instance_data_dir = os.path.join(workspace, instance_data)
    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt,
        class_data_root=class_data_dir,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=512,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Move vae and text_encoder to device and cast to torch_dtype
    vae.to(device, dtype=torch_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # Train!
    total_batch_size = train_batch_size * gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num batches each epoch = {len(train_dataloader)}')
    logger.info(f'  Num Epochs = {num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {max_train_steps}')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps))
    progress_bar.set_description('Steps')

    cur_scale = 2**32
    scale_factor = 2

    model.train()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            for _ in range(gradient_accumulation_steps):
                # Convert images to latent space
                latents = vae.encode(batch['pixel_values'].to(device, dtype=torch_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning and predict the noise residual
                model_pred = model(batch['input_ids'].to(device), noisy_latents, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f'Unknown prediction type {noise_scheduler.config.prediction_type}')

                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')

                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction='mean')

                # Add the prior loss to the instance loss.
                total_loss = loss + prior_loss

                scaled_loss = total_loss.float() * cur_scale
                scaled_loss.backward()

            if any(has_inf_or_nan(param.grad.data) for param in model.parameters()):
                cur_scale = max(cur_scale / scale_factor, 1.0)
                logger.info(f'OVERFLOW! Skipping step. Reducing to loss scale: {int(cur_scale)}')
            else:
                for param in model.parameters():
                    param.grad.data.mul_(1.0 / cur_scale)
                optimizer.step()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item())

    # Create the pipeline using using the trained modules and save it.
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained,
        unet=model.unet,
        text_encoder=model.text_encoder,
        safety_checker=None,
    )
    pipeline.save_pretrained(os.path.join(workspace, model_dir))


def import_model_class_from_model_name_or_path(pretrained):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained, subfolder='text_encoder')
    model_class = text_encoder_config.architectures[0]
    models = argparse.Namespace()
    models.CLIPTextModel = CLIPTextModel
    models.RobertaSeriesModelWithTransformation = RobertaSeriesModelWithTransformation
    return getattr(models, model_class)


class DreamBoothDataset(Dataset):
    '''
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    '''

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == 'RGB':
            instance_image = instance_image.convert('RGB')
        example['instance_images'] = self.image_transforms(instance_image)
        example['instance_prompt_ids'] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == 'RGB':
                class_image = class_image.convert('RGB')
            example['class_images'] = self.image_transforms(class_image)
            example['class_prompt_ids'] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt',
            ).input_ids

        return example


def collate_fn(examples):
    input_ids = [example['instance_prompt_ids'] for example in examples]
    pixel_values = [example['instance_images'] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    input_ids += [example['class_prompt_ids'] for example in examples]
    pixel_values += [example['class_images'] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        'input_ids': input_ids,
        'pixel_values': pixel_values,
    }
    return batch


class PromptDataset(Dataset):

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example['prompt'] = self.prompt
        example['index'] = index
        return example


class TextEncoderUnet(torch.nn.Module):

    def __init__(self, pretrained):
        super().__init__()
        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(pretrained)
        self.text_encoder = text_encoder_cls.from_pretrained(pretrained, subfolder='text_encoder')
        self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder='unet')

    def forward(self, input_ids, noisy_latents, timesteps):
        encoder_hidden_states = self.text_encoder(input_ids)[0]
        return self.unet(noisy_latents, timesteps, encoder_hidden_states).sample


def has_inf_or_nan(x):
    try:
        cpu_sum = x.detach().float().sum().item()
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if 'value cannot be converted' not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        return False


class CPUSGD(torch.optim.SGD):
    """
    SGD with states (momentum buffers) offloaded to CPU.
    """

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
            else:
                buf = buf.to(param.device, non_blocking=True)
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            momentum_buffer_list[i] = buf.to('cpu', non_blocking=True)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


def main_sample():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', help='Text prompt for sampling image.')
    parser.add_argument('workspace', nargs='?', default='dreambooth-diffuser',
                        help='The output directory where the model will be written.')
    parser.add_argument('--model_dir', default='model',
                        help='Subdirectory name under workspace to write the trained model to.')
    args = parser.parse_args()
    sample(args.prompt, args.workspace, args.model_dir)


def sample(prompt, workspace,
           model_dir='model',
           image_filenames=None,
           sample_batch_size=1,
           device='cuda',
           torch_dtype=torch.float16,
           num_inference_steps=50,
           guidance_scale=7.5):
    with open(os.path.join(workspace, 'description.txt'), 'r') as fp:
        description = fp.read()
    prompt = prompt.replace(description, f'sks {description}')
    model_id = os.path.join(workspace, model_dir)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch_dtype, safety_checker=None)
    pipe = pipe.to(device)
    prompt = [prompt for _ in range(sample_batch_size)]
    images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
    if image_filenames is None:
        image_filenames = [f'sample{idx}.png' for idx in range(len(images))]
    for image, image_filename in zip(images, image_filenames):
        image.save(os.path.join(workspace, image_filename))


if __name__ == '__main__':
    main()
