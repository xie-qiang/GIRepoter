"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import random

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption,isNorm = True):
        caption = self.prompt + self.pre_caption(caption)
        return caption
        # if isNorm:
        #     caption = self.prompt + self.pre_caption(caption)
        #     return caption
        # else:
        #     part = caption.split("：")[0]
        #     words = caption.lstrip(part+"：").strip("。").split("，")
        #     if len(words) != 1:
        #         idx = random.randint(0, len(words) - 1)
        #         chosen_word = words.pop(idx)
        #         words.insert(0, chosen_word)
        #         caption = part + "："+"，".join(words) + "。"
        #         caption = self.prompt + self.pre_caption(caption)

        #     return caption
                

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("blip_question")
class BlipQuestionProcessor(BaseProcessor):
    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 50)

        return cls(max_words=max_words)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question


@registry.register_processor("blip_image_train")
class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blip_image_eval")
class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform_norm = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,])
        # self.transform_norm = transforms.Compose([
        #         RandomRotationAndCrop(90),
        #         transforms.Resize((image_size, image_size)),
        #         transforms.ToTensor(),
        #         self.normalize,])

        self.transform_abnorm = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                # RandomRotationAndCrop(10),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize,])

    def __call__(self, item, isNorm = True):
        if isNorm:
            return self.transform_norm(item)
        else:
            if random.random() <= 0.2:
                return self.transform_abnorm(item)
            else:
                return self.transform_norm(item)
        # return self.transform_norm(item)
        
        

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )
    
import math
import random
from PIL import Image
import torchvision.transforms as transforms

class RandomRotationAndCrop:
    def __init__(self, degrees, expand=False, center=None, fill=None):
        self.degrees = degrees
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img):
        # Random rotation
        # angle = random.uniform(-self.degrees, self.degrees)
        angle = random.choice([-90-self.degrees,-90+self.degrees,90-self.degrees,90+self.degrees,180-self.degrees,180+self.degrees])
        # angle = random.uniform(-self.degrees, self.degrees)
        rotated_img = img.rotate(angle, expand=self.expand, center=self.center, fillcolor=self.fill)


        # Compute the largest possible square that fits inside the rotated rectangle
        w, h = img.size
        angle_rad = math.radians(abs(angle))
        new_w = w * math.cos(angle_rad) + h * math.sin(angle_rad)
        new_h = h * math.cos(angle_rad) + w * math.sin(angle_rad)
        new_w, new_h = min(w, h), min(w, h)

        left = (rotated_img.width - new_w) / 2
        top = (rotated_img.height - new_h) / 2
        right = (rotated_img.width + new_w) / 2
        bottom = (rotated_img.height + new_h) / 2

        # Crop the image to the largest possible square
        cropped_img = rotated_img.crop((left, top, right, bottom))
        return cropped_img
