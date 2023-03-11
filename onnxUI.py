import argparse
import functools
import pandas as pd
import gc
import os
import re
import time
import cv2
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import torch
from typing import Optional, Tuple
from math import ceil
import lpw_pipe
from lpw_stable_diffusion_onnx import OnnxStableDiffusionLongPromptWeightingPipeline
from pipeline_onnx_stable_diffusion_controlnet import OnnxStableDiffusionControlNetPipeline
from controlnet_aux import OpenposeDetector, HEDdetector
from transformers import pipeline

from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionInpaintPipelineLegacy,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
from diffusers import __version__ as _df_version
import gradio as gr
import numpy as np
from packaging import version
import PIL

    

# gradio function
def run_diffusers(
    prompt: str,
    neg_prompt: Optional[str],
    init_image: Optional[PIL.Image.Image],
    init_mask: Optional[PIL.Image.Image],
    iteration_count: int,
    batch_size: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    eta: float,
    denoise_strength: Optional[float],
    seed: str,
    image_format: str,
    legacy: bool,
    loopback: bool,
    preprocess: bool,
) -> Tuple[list, str]:
    global model_name
    global controlnet_name
    global provider
    global current_pipe
    global pipe
    global controlnet
    
    model_path = os.path.join("model", model_name)

    prompt.strip("\n")
    neg_prompt.strip("\n")

    # generate seeds for iterations
    if seed == "":
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max

    # use given seed for the first iteration
    seeds = np.array([seed], dtype=np.uint32)


    if iteration_count > 1:
        seed_seq = np.random.SeedSequence(seed)
        seeds = np.concatenate((seeds, seed_seq.generate_state(iteration_count - 1)))

    # create and parse output directory
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*")
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0

    sched_name = pipe.scheduler.__class__.__name__
    neg_prompt = None if neg_prompt == "" else neg_prompt
    images = []
    time_taken = 0
    for i in range(iteration_count):
        print(f"iteration {i + 1}/{iteration_count}")
        info = (
            f"{next_index + i:06} | "
            f"prompt: {prompt} "
            f"negative prompt: {neg_prompt} | "
            f"scheduler: {sched_name} "
            f"model: {model_name} "
            f"iteration size: {iteration_count} "
            f"batch size: {batch_size} "
            f"steps: {steps} "
            f"scale: {guidance_scale} "
            f"height: {height} "
            f"width: {width} "
            f"eta: {eta} "
            f"seed: {seeds[i]}"
        )
        if (current_pipe == "img2img"):
            info = info + f" denoise: {denoise_strength}"
        with open(os.path.join(output_path, "history.txt"), "a") as log:
            log.write(info + "\n")

        # create generator object from seed
        #rng = np.random.RandomState(seeds[i])
        rng = torch.Generator()
        rng.manual_seed(int(seeds[i]))
        rng_cnet = np.random.RandomState(seeds[i])

        if current_pipe == "txt2img":
            start = time.time()
            batch_images = pipe(
                prompt,
                negative_prompt=neg_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                eta=eta,
                num_images_per_prompt=batch_size,
                generator=rng,
                ancestral_generator=rng).images
            finish = time.time()
        elif current_pipe == "img2img":
            pipe.vae_encoder = OnnxRuntimeModel.from_pretrained(
                model_path + "/vae_encoder",provider=provider)
            start = time.time()
            if loopback is True:
                try:
                    loopback_image
                except UnboundLocalError:
                    loopback_image = None

                if loopback_image is not None:
                    batch_images = pipe(
                        prompt,
                        negative_prompt=neg_prompt,
                        image=loopback_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        strength=denoise_strength,
                        num_images_per_prompt=batch_size,
                        generator=rng,
                        ancestral_generator=rng,
                    ).images
                elif loopback_image is None:
                    batch_images = pipe(
                        prompt,
                        negative_prompt=neg_prompt,
                        image=init_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        strength=denoise_strength,
                        num_images_per_prompt=batch_size,
                        generator=rng,
                        ancestral_generator=rng,
                    ).images
            elif loopback is False:
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=init_image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    strength=denoise_strength,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                    ancestral_generator=rng,
                ).images
            finish = time.time()
        elif current_pipe == "inpaint":
            pipe.vae_encoder = OnnxRuntimeModel.from_pretrained(
                model_path + "/vae_encoder",provider=provider)
            start = time.time()
            if legacy is True:
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=init_image,
                    mask_image=init_mask,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                    ancestral_generator=rng,
                ).images
            else:
                batch_images = pipe.inpaint_new(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=init_image,
                    mask_image=init_mask,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                    ancestral_generator=rng,
                ).images
            finish = time.time()
        elif current_pipe == "controlnet":
            if preprocess:
                cnet_image=init_image
                print("no preprocessing")
            else:
                if controlnet_type == "canny":
                    image = np.array(init_image)
                    low_threshold = 100
                    high_threshold = 200

                    image = cv2.Canny(image, low_threshold, high_threshold)
                    image = image[:, :, None]
                    image = np.concatenate([image, image, image], axis=2)
                    cnet_image = PIL.Image.fromarray(image)
                elif controlnet_type == "openpose":
                    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
                    cnet_image = openpose(init_image)
                    del openpose
                    gc.collect() 
                elif controlnet_type == "scribble":
                    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
                    cnet_image = hed(init_image, scribble=True)
                    del hed
                    gc.collect()   
                elif controlnet_type == "depth":
                    depth_estimator = pipeline('depth-estimation')
                    image = depth_estimator(image)['depth']
                    image = np.array(image)
                    image = image[:, :, None]
                    image = np.concatenate([image, image, image], axis=2)
                    cnet_image = PIL.Image.fromarray(image)
                cnet_image.save("./tmp.png")
            start = time.time()
            batch_images = pipe(
                prompt,
                negative_prompt=neg_prompt,
                image=cnet_image,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                eta=eta,
                num_images_per_prompt=batch_size,
                generator=rng_cnet).images
            finish = time.time()
        if vaedec_on_cpu:
            pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(
                model_path + "/vae_decoder")
        else:
            pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(
                model_path + "/vae_decoder",provider=provider)
        short_prompt = prompt.strip("<>:\"/\\|?*\n\t")
        short_prompt = re.sub(r'[\\/*?:"<>|\n\t]', "", short_prompt)
        short_prompt = short_prompt[:99] if len(short_prompt) > 100 else short_prompt

        if loopback is True:
            loopback_image = batch_images[0]

            # png output
            if image_format == "png":
                loopback_image.save(
                    os.path.join(
                        output_path,
                        f"{next_index + i:06}-00.{short_prompt}.{image_format}",
                    ),
                    optimize=True,
                )
            # jpg output
            elif image_format == "jpg":
                loopback_image.save(
                    os.path.join(
                        output_path,
                        f"{next_index + i:06}-00.{short_prompt}.{image_format}",
                    ),
                    quality=95,
                    subsampling=0,
                    optimize=True,
                    progressive=True,
                )
        elif loopback is False:
            # png output
            if image_format == "png":
                for j in range(batch_size):
                    batch_images[j].save(
                        os.path.join(
                            output_path,
                            f"{next_index + i:06}-{j:02}.{short_prompt}.{image_format}",
                        ),
                        optimize=True,
                    )
            # jpg output
            elif image_format == "jpg":
                for j in range(batch_size):
                    batch_images[j].save(
                        os.path.join(
                            output_path,
                            f"{next_index + i:06}-{j:02}.{short_prompt}.{image_format}",
                        ),
                        quality=95,
                        subsampling=0,
                        optimize=True,
                        progressive=True,
                    )

        images.extend(batch_images)
        time_taken = time_taken + (finish - start)

    time_taken = time_taken / 60.0
    if iteration_count > 1:
        status = (
            f"Run indexes {next_index:06} "
            f"to {next_index + iteration_count - 1:06} "
            f"took {time_taken:.1f} minutes "
            f"to generate {iteration_count} "
            f"iterations with batch size of {batch_size}. "
            f"seeds: " + np.array2string(seeds, separator=",")
        )
    else:
        status = (
            f"Run index {next_index:06} "
            f"took {time_taken:.1f} minutes "
            f"to generate a batch size of {batch_size}. "
            f"seed: {seeds[0]}"
        )

    return images, status


def resize_and_crop(input_image: PIL.Image.Image, height: int, width: int):
    input_width, input_height = input_image.size
    if height / width > input_height / input_width:
        adjust_width = int(input_width * height / input_height)
        input_image = input_image.resize((adjust_width, height))
        left = (adjust_width - width) // 2
        right = left + width
        input_image = input_image.crop((left, 0, right, height))
    else:
        adjust_height = int(input_height * width / input_width)
        input_image = input_image.resize((width, adjust_height))
        top = (adjust_height - height) // 2
        bottom = top + height
        input_image = input_image.crop((0, top, width, bottom))
    return input_image
    
    
    
def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im
    
def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img
    
def danbooru_click(extras_image):

    tagger_model_path = hf_hub_download(repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2", revision='v2.0', filename="model.onnx")
    tags_path = hf_hub_download(repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2", revision='v2.0', filename="selected_tags.csv")

    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    tagger_model = ort.InferenceSession(tagger_model_path, sess_options=opts, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    modeltags = pd.read_csv(tags_path)
    _, height, _, _ = tagger_model.get_inputs()[0].shape
    
    image = extras_image.convert("RGBA")
    new_image = PIL.Image.new('RGBA', image.size, 'WHITE')
    new_image.paste(image, mask=image)
    image = new_image.convert('RGB')
    image = np.asarray(image)
    
    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]
    
    image = make_square(image, height)
    image = smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)
    
    input_name = tagger_model.get_inputs()[0].name
    label_name = tagger_model.get_outputs()[0].name
    confidents = tagger_model.run([label_name], {input_name: image})[0]
    
    tags = modeltags[:][['name']]
    tags['confidents'] = confidents[0]

    # first 4 items are for rating (general, sensitive, questionable, explicit)
    ratings = dict(tags[:4].values)

    # rest are regular tags
    tags = dict(tags[4:].values)
    
    keys = list(ratings.keys())
    values = list(ratings.values())
    sorted_value_index = np.argsort(values)[::-1]
    sortedratings = {keys[i]: values[i] for i in sorted_value_index}

    print(sortedratings)
    
    keys = list(tags.keys())
    values = list(tags.values())
    sorted_value_index = np.argsort(values)[::-1]
    sortedtags = {keys[i]: values[i] for i in sorted_value_index}
    
    newprompt = ""
    for x in sortedtags:
        if sortedtags[x] >= 0.35:
            newprompt+=(x + ", ")
    newprompt = newprompt.strip(", ")
    repdict = {"_": " ", "\\": "\\\\", "(": "\\(" , ")": "\\)"}
    for key, value in repdict.items():
        newprompt=newprompt.replace(key,value)

    print(newprompt)
    del tagger_model
    gc.collect()
    return {prompt_t3: newprompt}

def clear_click():
    global current_tab
    if current_tab == 0:
        return {
            prompt_t0: "",
            neg_prompt_t0: "",
            sch_t0: "DPMS_ms",
            iter_t0: 1,
            batch_t0: 1,
            steps_t0: 16,
            guid_t0: 7.5,
            height_t0: 512,
            width_t0: 512,
            eta_t0: 0.0,
            seed_t0: "",
            fmt_t0: "png",
        }
    elif current_tab == 1:
        return {
            prompt_t1: "",
            neg_prompt_t1: "",
            sch_t1: "DPMS_ms",
            image_t1: None,
            iter_t1: 1,
            batch_t1: 1,
            steps_t1: 16,
            guid_t1: 7.5,
            height_t1: 512,
            width_t1: 512,
            eta_t1: 0.0,
            denoise_t1: 0.8,
            seed_t1: "",
            fmt_t1: "png",
            loopback_t1: False,
        }
    elif current_tab == 2:
        return {
            prompt_t2: "",
            neg_prompt_t2: "",
            sch_t2: "DPMS_ms",
            legacy_t2: False,
            image_t2: None,
            iter_t2: 1,
            batch_t2: 1,
            steps_t2: 16,
            guid_t2: 7.5,
            height_t2: 512,
            width_t2: 512,
            eta_t2: 0.0,
            seed_t2: "",
            fmt_t2: "png",
        }
    elif current_tab == 4:
        return {
            prompt_t4: "",
            neg_prompt_t4: "",
            sch_t4: "DPMS_ms",
            preprocess_t4: False,
            image_t4: None, 
            iter_t4: 1,
            batch_t4: 1,
            steps_t4: 16,
            guid_t4: 7.5,
            height_t4: 512,
            width_t4: 512,
            eta_t4: 0.0,
            seed_t4: "",
            fmt_t4: "png",
        }


def generate_click(
    model_drop,
    controlnet_drop,
    prompt_t0,
    neg_prompt_t0,
    sch_t0,
    iter_t0,
    batch_t0,
    steps_t0,
    guid_t0,
    height_t0,
    width_t0,
    eta_t0,
    seed_t0,
    fmt_t0,
    prompt_t1,
    neg_prompt_t1,
    image_t1,
    sch_t1,
    iter_t1,
    batch_t1,
    steps_t1,
    guid_t1,
    height_t1,
    width_t1,
    eta_t1,
    denoise_t1,
    seed_t1,
    fmt_t1,
    loopback_t1,
    prompt_t2,
    neg_prompt_t2,
    sch_t2,
    legacy_t2,
    image_t2,
    iter_t2,
    batch_t2,
    steps_t2,
    guid_t2,
    height_t2,
    width_t2,
    eta_t2,
    seed_t2,
    fmt_t2,
    prompt_t3,
    prompt_t4,
    neg_prompt_t4,
    image_t4,
    sch_t4,
    preprocess_t4,
    iter_t4,
    batch_t4,
    steps_t4,
    guid_t4,
    height_t4,
    width_t4,
    eta_t4,
    seed_t4,
    fmt_t4,
):
    global model_name
    global controlnet_name
    global provider
    global current_tab
    global current_pipe
    global current_legacy
    global release_memory_after_generation
    global release_memory_on_change
    global scheduler
    global controlnet_type
    global pipe
    global controlnet

    # reset scheduler and pipeline if model is different
    if model_name != model_drop:
        model_name = model_drop
        scheduler = None
        pipe = None
        gc.collect()
    model_path = os.path.join("model", model_name)
    
    if controlnet_name != controlnet_drop:
        controlnet_name = controlnet_drop
        controlnet = None
        #scheduler = None
        #pipe = None
        gc.collect()
    controlnet_path = os.path.join("controlnet", controlnet_name)
    if "canny" in controlnet_name:
        controlnet_type = "canny"
    elif "openpose" in controlnet_name:
        controlnet_type = "openpose"
    elif "scribble" in controlnet_name:
        controlnet_type = "scribble"

    # select which scheduler depending on current tab
    if current_tab == 0:
        sched_name = sch_t0
    elif current_tab == 1:
        sched_name = sch_t1
    elif current_tab == 2:
        sched_name = sch_t2
    elif current_tab == 4:
        sched_name = sch_t4
    else:
        raise Exception("Unknown tab")

    if sched_name == "PNDM" and type(scheduler) is not PNDMScheduler:
        scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "LMS" and type(scheduler) is not LMSDiscreteScheduler:
        scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "DDIM" and type(scheduler) is not DDIMScheduler:
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "Euler" and type(scheduler) is not EulerDiscreteScheduler:
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "EulerA" and type(scheduler) is not EulerAncestralDiscreteScheduler:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "DPMS_ms" and type(scheduler) is not DPMSolverMultistepScheduler:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "DPMS_ss" and type(scheduler) is not DPMSolverSinglestepScheduler:
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "DEIS" and type(scheduler) is not DEISMultistepScheduler:
        scheduler = DEISMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "HEUN" and type(scheduler) is not HeunDiscreteScheduler:
        scheduler = HeunDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "KDPM2" and type(scheduler) is not KDPM2DiscreteScheduler:
        scheduler = KDPM2DiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "UniPC" and type(scheduler) is not UniPCMultistepScheduler:
        scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")

    # select which pipeline depending on current tab
    if current_tab == 0:
        if pipe is None:
            if textenc_on_cpu and vaedec_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                    vae_decoder=cpuvaedec,
                    vae_encoder=None
                )
            elif textenc_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc)
            elif vaedec_on_cpu:
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    vae_decoder=cpuvaedec,
                    vae_encoder=None
                )
            else:
                pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler)
        current_pipe = "txt2img"
    elif current_tab == 1:
        if pipe is None:
            if textenc_on_cpu and vaedec_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                    vae_decoder=cpuvaedec)
            elif textenc_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc)
            elif vaedec_on_cpu:
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    vae_decoder=cpuvaedec)
            else:
                pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler)
        current_pipe = "img2img"
    elif current_tab == 2:
        if pipe is None:
            if legacy_t2:
                if textenc_on_cpu and vaedec_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                        vae_decoder=cpuvaedec)
                elif textenc_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc)
                elif vaedec_on_cpu:
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        vae_decoder=cpuvaedec)
                else:
                    pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler)
            else:
                if textenc_on_cpu and vaedec_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                        vae_decoder=cpuvaedec)
                elif textenc_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc)
                elif vaedec_on_cpu:
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        vae_decoder=cpuvaedec)
                else:
                    pipe = OnnxStableDiffusionLongPromptWeightingPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler)
        current_pipe = "inpaint"
        current_legacy = legacy_t2
    elif current_tab == 4:
        if current_pipe != "controlnet" or pipe is None:
            if controlnet == None:
                controlnet = OnnxRuntimeModel.from_pretrained(
                    controlnet_path + "/cnet", provider=provider)
            
            if textenc_on_cpu and vaedec_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                    vae_decoder=cpuvaedec,
                    controlnet=controlnet)
            elif textenc_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                    controlnet=controlnet)
            elif vaedec_on_cpu:
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    vae_decoder=cpuvaedec,
                    controlnet=controlnet)
            else:
                pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    controlnet=controlnet)
        else:
            if controlnet == None:
                controlnet = OnnxRuntimeModel.from_pretrained(
                    controlnet_path + "/cnet", provider=provider)
                pipe.controlnet = None
                gc.collect()
                pipe.controlnet = controlnet
        current_pipe = "controlnet"

    # manual garbage collection
    gc.collect()

    # modifying the methods in the pipeline object
    if type(pipe.scheduler) is not type(scheduler):
        pipe.scheduler = scheduler
    if version.parse(_df_version) >= version.parse("0.8.0"):
        safety_checker = None
    else:
        safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe.safety_checker = safety_checker
    if current_pipe == "controlnet":
        pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, pipe)

    # run the pipeline with the correct parameters
    if current_tab == 0:
        images, status = run_diffusers(
            prompt_t0,
            neg_prompt_t0,
            None,
            None,
            iter_t0,
            batch_t0,
            steps_t0,
            guid_t0,
            height_t0,
            width_t0,
            eta_t0,
            0,
            seed_t0,
            fmt_t0,
            None,
            False,
            False,
        )
    elif current_tab == 1:
        # input image resizing
        input_image = image_t1.convert("RGB")
        input_image = resize_and_crop(input_image, height_t1, width_t1)
        
        # adjust steps to account for denoise.
        steps_t1_old = steps_t1
        steps_t1 = ceil(steps_t1 / denoise_t1)
        if steps_t1 > 1000 and (sch_t1 == "DPMS_ms" or "DPMS_ss" or "DEIS"):
            steps_t1_unreduced = steps_t1
            steps_t1 = 1000
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_t1_old} to {steps_t1_unreduced} steps internally."
            )
            print(
                f"Without adjustment the actual step count would be ~{ceil(steps_t1_old * denoise_t1)} steps."
            )
            print()
            print(
                f"INTERNAL STEP COUNT EXCEEDS 1000 MAX FOR DPMS_ms, DPMS_ss, or DEIS. INTERNAL STEPS WILL BE REDUCED TO 1000."
            )
            print()
        else:
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_t1_old} to {steps_t1} steps internally."
            )
            print(
                f"Without adjustment the actual step count would be ~{ceil(steps_t1_old * denoise_t1)} steps."
            )
            print()

        images, status = run_diffusers(
            prompt_t1,
            neg_prompt_t1,
            input_image,
            None,
            iter_t1,
            batch_t1,
            steps_t1,
            guid_t1,
            height_t1,
            width_t1,
            eta_t1,
            denoise_t1,
            seed_t1,
            fmt_t1,
            None,
            loopback_t1,
            False,
        )
        pipe.vae_encoder = OnnxRuntimeModel.from_pretrained(
            model_path + "/vae_encoder",provider=provider)
    elif current_tab == 2:
        input_image = image_t2["image"].convert("RGB")
        input_image = resize_and_crop(input_image, height_t2, width_t2)

        input_mask = image_t2["mask"].convert("RGB")
        input_mask = resize_and_crop(input_mask, height_t2, width_t2)
        
        # adjust steps to account for legacy inpaint only using ~80% of set steps.
        if legacy_t2 is True:
            steps_t2_old = steps_t2
            if steps_t2 < 5:
                steps_t2 = steps_t2 + 1
            elif steps_t2 >= 5:
                steps_t2 = int((steps_t2 / 0.7989) + 1)
            print()
            print(
                f"Adjusting steps for legacy inpaint. From {steps_t2_old} to {steps_t2} internally."
            )
            print(
                f"Without adjustment the actual step count would be ~{int(steps_t2_old * 0.8)} steps."
            )
            print()

        images, status = run_diffusers(
            prompt_t2,
            neg_prompt_t2,
            input_image,
            input_mask,
            iter_t2,
            batch_t2,
            steps_t2,
            guid_t2,
            height_t2,
            width_t2,
            eta_t2,
            0,
            seed_t2,
            fmt_t2,
            legacy_t2,
            False,
            False,
        )
        #pipe.vae_encoder = OnnxRuntimeModel.from_pretrained(
        #    model_path + "/vae_encoder",provider=provider)
    elif current_tab == 4:
        # input image resizing
        input_image = image_t4.convert("RGB")
        input_image = resize_and_crop(input_image, height_t4, width_t4)
        
        if steps_t4 > 1000 and (sch_t4 == "DPMS_ms" or "DPMS_ss" or "DEIS"):
            steps_t4_unreduced = steps_t4
            steps_t4 = 1000
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_t4_old} to {steps_t4_unreduced} steps internally."
            )
            print()
            print(
                f"INTERNAL STEP COUNT EXCEEDS 1000 MAX FOR DPMS_ms, DPMS_ss, or DEIS. INTERNAL STEPS WILL BE REDUCED TO 1000."
            )
            print()

        images, status = run_diffusers(
            prompt_t4,
            neg_prompt_t4,
            input_image,
            None,
            iter_t4,
            batch_t4,
            steps_t4,
            guid_t4,
            height_t4,
            width_t4,
            eta_t4,
            0,
            seed_t4,
            fmt_t4,
            None,
            False,
            preprocess_t4,
        )
        pipe.vae_encoder = OnnxRuntimeModel.from_pretrained(
            model_path + "/vae_encoder",provider=provider)

    if release_memory_after_generation:
        pipe = None
        gc.collect()
    #if vaedec_on_cpu:
    #    pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(
    #        model_path + "/vae_decoder")
    #else:
    #    pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(
     #       model_path + "/vae_decoder",provider=provider)
    return images, status


def select_tab0():
    global current_tab
    current_tab = 0


def select_tab1():
    global current_tab
    current_tab = 1


def select_tab2():
    global current_tab
    current_tab = 2
    
def select_tab3():
    global current_tab
    current_tab = 3
    
def select_tab4():
    global current_tab
    current_tab = 4


def choose_sch(sched_name: str):
    if sched_name == "DDIM":
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gradio interface for ONNX based Stable Diffusion")
    parser.add_argument("--cpu-only", action="store_true", default=False, help="run ONNX with CPU")
    parser.add_argument(
        "--cpu-textenc", action="store_true", default=False,
        help="Run Text Encoder on CPU, saves VRAM by running Text Encoder on CPU")
    parser.add_argument(
        "--cpu-vaedec", action="store_true", default=False,
        help="Run VAE Decoder on CPU, saves VRAM by running VAE Decoder on CPU")
    parser.add_argument(
        "--release-memory-after-generation", action="store_true", default=False,
        help="de-allocate the pipeline and release memory after generation")
    parser.add_argument(
        "--release-memory-on-change", action="store_true", default=False,
        help="de-allocate the pipeline and release memory allocated when changing pipelines",
    )
    args = parser.parse_args()

    # variables for ONNX pipelines
    model_name = None
    controlnet_name = None
    provider = "CPUExecutionProvider" if args.cpu_only else "DmlExecutionProvider"
    current_tab = 0
    current_pipe = "txt2img"
    current_legacy = False
    release_memory_after_generation = args.release_memory_after_generation
    release_memory_on_change = args.release_memory_on_change
    textenc_on_cpu = args.cpu_textenc
    vaedec_on_cpu = args.cpu_vaedec

    # diffusers objects
    scheduler = None
    pipe = None

    # check versions
    is_v_0_12 = version.parse(_df_version) >= version.parse("0.12.0")
    is_v_dev = version.parse(_df_version).is_prerelease

    # prerelease version use warning
    if is_v_dev:
        print(
            "You are using diffusers " + str(version.parse(_df_version)) + " (prerelease)\n" +
            "If you experience unexpected errors please run `pip install diffusers --force-reinstall`.")

    # custom css
    custom_css = """
    #gen_button {height: 90px}
    #image_init {min-height: 400px}
    #image_init [data-testid="image"], #image_init [data-testid="image"] > div {min-height: 400px}
    #image_inpaint {min-height: 400px}
    #image_inpaint [data-testid="image"], #image_inpaint [data-testid="image"] > div {min-height: 400px}
    #image_inpaint .touch-none {display: flex}
    #image_inpaint img {display: block; max-width: 84%}
    #image_inpaint canvas {max-width: 84%; object-fit: contain}
    """

    # search the model folder
    model_dir = "model"
    model_list = []
    with os.scandir(model_dir) as scan_it:
        for entry in scan_it:
            if entry.is_dir():
                model_list.append(entry.name)
    default_model = model_list[0] if len(model_list) > 0 else None
    
    controlnet_dir = "controlnet"
    controlnet_list = []
    with os.scandir(controlnet_dir) as scan_it:
        for entry in scan_it:
            if entry.is_dir():
                controlnet_list.append(entry.name)
    default_controlmodel = controlnet_list[0] if len(controlnet_list) > 0 else None

    if is_v_0_12:
        from diffusers import (
            DPMSolverSinglestepScheduler,
            DEISMultistepScheduler,
            HeunDiscreteScheduler,
            KDPM2DiscreteScheduler
        )
        sched_list = ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC"]
    else:
        sched_list = ["DPMS_ms", "EulerA", "Euler", "DDIM", "LMS", "PNDM"]

    # create gradio block
    title = "Stable Diffusion ONNX"
    with gr.Blocks(title=title, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=13, min_width=650):
                model_drop = gr.Dropdown(model_list, value=default_model, label="model folder", interactive=True)
                controlnet_drop = gr.Dropdown(controlnet_list, value=default_controlmodel, label="controlnet folder", interactive=True)
            with gr.Column(scale=11, min_width=550):
                with gr.Row():
                    gen_btn = gr.Button("Generate", variant="primary", elem_id="gen_button")
                    clear_btn = gr.Button("Clear", elem_id="gen_button")
        with gr.Row():
            with gr.Column(scale=13, min_width=650):
                with gr.Tab(label="txt2img") as tab0:
                    prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t0 = gr.Textbox(value="", lines=2, label="negative prompt")
                    sch_t0 = gr.Radio(sched_list, value="DPMS_ms", label="scheduler")
                    with gr.Row():
                        iter_t0 = gr.Slider(1, 24, value=1, step=1, label="iteration count")
                        batch_t0 = gr.Slider(1, 4, value=1, step=1, label="batch size")
                    steps_t0 = gr.Slider(1, 300, value=16, step=1, label="steps")
                    guid_t0 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                    height_t0 = gr.Slider(192, 1536, value=512, step=64, label="height")
                    width_t0 = gr.Slider(192, 1536, value=512, step=64, label="width")
                    eta_t0 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
                    seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t0 = gr.Radio(["png", "jpg"], value="png", label="image format")
                with gr.Tab(label="img2img") as tab1:
                    prompt_t1 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t1 = gr.Textbox(value="", lines=2, label="negative prompt")
                    sch_t1 = gr.Radio(sched_list, value="DPMS_ms", label="scheduler")
                    image_t1 = gr.Image(label="input image", type="pil", elem_id="image_init")
                    with gr.Row():
                        iter_t1 = gr.Slider(1, 24, value=1, step=1, label="iteration count")
                        batch_t1 = gr.Slider(1, 4, value=1, step=1, label="batch size")
                    with gr.Row():
                        loopback_t1 = gr.Checkbox(value=False, label="loopback (use iteration count)")
                    steps_t1 = gr.Slider(1, 300, value=16, step=1, label="steps")
                    guid_t1 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                    height_t1 = gr.Slider(192, 1536, value=512, step=64, label="height")
                    width_t1 = gr.Slider(192, 1536, value=512, step=64, label="width")
                    eta_t1 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
                    denoise_t1 = gr.Slider(0, 1, value=0.8, step=0.01, label="denoise strength")
                    seed_t1 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t1 = gr.Radio(["png", "jpg"], value="png", label="image format")
                with gr.Tab(label="inpainting") as tab2:
                    prompt_t2 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t2 = gr.Textbox(value="", lines=2, label="negative prompt")
                    sch_t2 = gr.Radio(sched_list, value="DPMS_ms", label="scheduler")
                    legacy_t2 = gr.Checkbox(value=False, label="legacy inpaint")
                    image_t2 = gr.Image(
                        source="upload", tool="sketch", label="input image", type="pil", elem_id="image_inpaint")
                    with gr.Row():
                        iter_t2 = gr.Slider(1, 24, value=1, step=1, label="iteration count")
                        batch_t2 = gr.Slider(1, 4, value=1, step=1, label="batch size")
                    steps_t2 = gr.Slider(1, 300, value=16, step=1, label="steps")
                    guid_t2 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                    height_t2 = gr.Slider(192, 1536, value=512, step=64, label="height")
                    width_t2 = gr.Slider(192, 1536, value=512, step=64, label="width")
                    eta_t2 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
                    seed_t2 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t2 = gr.Radio(["png", "jpg"], value="png", label="image format")
                with gr.Tab(label="extras") as tab3:
                    prompt_t3 = gr.Textbox(value="", lines=2, label="prompt")
                    extras_image = gr.Image(label="input image", type="pil", elem_id="image_extras")
                    danbooru_btn = gr.Button("Deepdanbooru", elem_id="deepdb_button")
                with gr.Tab(label="controlnet") as tab4:
                    prompt_t4 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t4 = gr.Textbox(value="", lines=2, label="negative prompt")
                    sch_t4 = gr.Radio(sched_list, value="DPMS_ms", label="scheduler")
                    preprocess_t4 = gr.Checkbox(value=False, label="Don't preprocess image")
                    image_t4 = gr.Image(label="input image", type="pil", elem_id="image_init")
                    with gr.Row():
                        iter_t4 = gr.Slider(1, 24, value=1, step=1, label="iteration count")
                        batch_t4 = gr.Slider(1, 1, value=1, step=1, label="batch size")
                    steps_t4 = gr.Slider(1, 300, value=16, step=1, label="steps")
                    guid_t4 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                    height_t4 = gr.Slider(192, 1536, value=512, step=64, label="height")
                    width_t4 = gr.Slider(192, 1536, value=512, step=64, label="width")
                    eta_t4 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
                    seed_t4 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t4 = gr.Radio(["png", "jpg"], value="png", label="image format")
            with gr.Column(scale=11, min_width=550):
                global image_out
                image_out = gr.Gallery(value=None, label="output images")
                status_out = gr.Textbox(value="", label="status")

        # config components
        tab0_inputs = [
            prompt_t0,
            neg_prompt_t0,
            sch_t0,
            iter_t0,
            batch_t0,
            steps_t0,
            guid_t0,
            height_t0,
            width_t0,
            eta_t0,
            seed_t0,
            fmt_t0,
        ]
        tab1_inputs = [
            prompt_t1,
            neg_prompt_t1,
            image_t1,
            sch_t1,
            iter_t1,
            batch_t1,
            steps_t1,
            guid_t1,
            height_t1,
            width_t1,
            eta_t1,
            denoise_t1,
            seed_t1,
            fmt_t1,
            loopback_t1,
        ]
        tab2_inputs = [
            prompt_t2,
            neg_prompt_t2,
            sch_t2,
            legacy_t2,
            image_t2,
            iter_t2,
            batch_t2,
            steps_t2,
            guid_t2,
            height_t2,
            width_t2,
            eta_t2,
            seed_t2,
            fmt_t2,
        ]
        tab3_inputs = [
            prompt_t3,
        ]
        tab4_inputs = [
            prompt_t4,
            neg_prompt_t4,
            image_t4,
            sch_t4,
            preprocess_t4,
            iter_t4,
            batch_t4,
            steps_t4,
            guid_t4,
            height_t4,
            width_t4,
            eta_t4,
            seed_t4,
            fmt_t4,
        ]
        all_inputs = [model_drop]
        all_inputs.extend([controlnet_drop])
        all_inputs.extend(tab0_inputs)
        all_inputs.extend(tab1_inputs)
        all_inputs.extend(tab2_inputs)
        all_inputs.extend(tab3_inputs)
        all_inputs.extend(tab4_inputs)
        all_prompts = [prompt_t0,prompt_t1,prompt_t2,prompt_t3, prompt_t4]

        extras_image.change(fn=danbooru_click, inputs=[extras_image], outputs=all_prompts)
        clear_btn.click(fn=clear_click, inputs=None, outputs=all_inputs, queue=False)
        gen_btn.click(fn=generate_click, inputs=all_inputs, outputs=[image_out, status_out])
        danbooru_btn.click(fn=danbooru_click, inputs=[extras_image], outputs=all_prompts)

        tab0.select(fn=select_tab0, inputs=None, outputs=None)
        tab1.select(fn=select_tab1, inputs=None, outputs=None)
        tab2.select(fn=select_tab2, inputs=None, outputs=None)
        tab3.select(fn=select_tab3, inputs=None, outputs=None)
        tab4.select(fn=select_tab4, inputs=None, outputs=None)

        sch_t0.change(fn=choose_sch, inputs=sch_t0, outputs=eta_t0, queue=False)
        sch_t1.change(fn=choose_sch, inputs=sch_t1, outputs=eta_t1, queue=False)
        sch_t2.change(fn=choose_sch, inputs=sch_t2, outputs=eta_t2, queue=False)
        sch_t4.change(fn=choose_sch, inputs=sch_t2, outputs=eta_t2, queue=False)

        image_out.style(grid=2)
        image_t1.style(height=402)
        image_t2.style(height=402)
        image_t4.style(height=402)

    # start gradio web interface on local host
    demo.launch()

    # use the following to launch the web interface to a private network
    # demo.queue(concurrency_count=1)
    # demo.launch(server_name="0.0.0.0")
