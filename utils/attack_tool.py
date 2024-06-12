from PIL import Image
import numpy as np
import torch
import gc
import os
import json
import random
import string
import tqdm
import time

import sys
sys.path.append("..")

from config import (MODEL_TEMPLATE_AND_PATH,
                    CLS_INSTRUCTIONS,
                    CAP_INSTRUCTIONS,
                    VQA_INSTRUCTIONS)

from utils.manager import InputInfo, AttackManager, ConversationManager


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def attack(args, model, processor):
    # Load the model and tokenizer
    conversation_manager = ConversationManager(args.model_name, MODEL_TEMPLATE_AND_PATH)
    path, template, patch_size, image_size = conversation_manager.get_template_name_and_path()

    # Open the image
    PIL_image = Image.open(args.image_path)
    
    # Prepare the input
    input_info_list = []

    if args.benchmark in ["vllm-attack", "vllm-attack-image", "vllm-attack-text"]:
        args.instructions = [args.instructions[0]]

    for instruction in args.instructions:
        input_info = InputInfo(processor,
                            PIL_image,
                            target_class=args.target_class,
                            original_instruction=instruction,
                            output_text=args.output_texts,
                            instruction_length=args.instruction_length,
                            padding_token=args.padding_token,
                            image_size = image_size,
                            patch_size = patch_size,
                            support_prefix=args.support_prefix)
        input_info_list.append(input_info)

    attack_manager = AttackManager(args.model_name, 
                                   input_info_list, 
                                   MODEL_TEMPLATE_AND_PATH,
                                   only_instruction=args.only_instruction)

    original_prompt_list = attack_manager.get_original_prompt_list(response=args.output_texts)
    target_prompt_list = attack_manager.get_target_prompt_list(response=args.output_texts)

    print("Original Prompt:", original_prompt_list)
    print("Target Prompt:", target_prompt_list)


    inputs_original_list = [processor(text = original_prompt, images = PIL_image, return_tensors="pt").to(model.device) for original_prompt in original_prompt_list]
    inputs_target_list = [processor(text = target_prompt, images = PIL_image, return_tensors="pt").to(model.device) for target_prompt in target_prompt_list]

    # Get the slices for different inputs
    image_slice, instruction_loss_slice, output_text_loss_slice, instruction_slice, output_text_slice, instruction_attention_slice = attack_manager.get_slice()

    # Prepare the attack inputs
    image_tokens, instruction_tokens, output_tokens = prepare_attack(args, 
                                                                        model, 
                                                                        processor,
                                                                        inputs_target_list[0], 
                                                                        image_slice, 
                                                                        instruction_slice, 
                                                                        output_text_slice)

    # Create copies of the original and adversarial attack pixels
    inputs_list = inputs_original_list.copy()
    inputs = inputs_original_list[0].copy()
    original_pixel = inputs["pixel_values"].clone().detach()
    adversial_attack_pixel = inputs["pixel_values"].clone().detach()

    adversial_attack_pixel.requires_grad = True

    ok, adversial_attack_pixel, losses = gradient_attack(args,
                                                    model,
                                                    processor,
                                                    attack_manager,
                                                    inputs_list,
                                                    original_pixel,
                                                    adversial_attack_pixel,
                                                    image_slice,
                                                    instruction_slice,
                                                    instruction_loss_slice,
                                                    output_text_loss_slice,
                                                    image_tokens,
                                                    instruction_tokens,
                                                    output_tokens)
    
    # Evaluate the attack
    result = evaluate(args, model, processor, attack_manager, adversial_attack_pixel)

    finish_log(args, ok, result, losses, adversial_attack_pixel, original_pixel)

def prepare_attack(args, model, processor, inputs_target, image_slice, instruction_slice, output_text_slice):
    """
    Prepare the attack inputs for the gradient attack.

    Args:
        args: arguments
        model: model
        inputs_target: target inputs
        image_slice: slice for image
        instruction_attention_slice: slice for instruction attention
        instruction_slice: slice for instruction tokens
        output_text_slice: slice for output tokens

    Returns:
        instruction_image_attention: target instruction attention for text
        instruction_tokens: target instruction tokens
        output_tokens: target output tokens
    """
    # Get image token
    PIL_image = Image.open(args.image_path)
    image_token = processor(text = args.target_class, images = PIL_image, return_tensors="pt").to(model.device)["input_ids"][0][1:]

    length_of_image = image_slice.stop-image_slice.start
    image_tokens = image_token.repeat(image_slice.stop-image_slice.start)[:length_of_image]

    # Get target instruction tokens and output tokens
    instruction_tokens = inputs_target["input_ids"][0][instruction_slice]
    output_tokens = inputs_target["input_ids"][0][output_text_slice]

    # Clean up
    gc.collect()

    return image_tokens, instruction_tokens, output_tokens


def gradient_attack(args, 
                    model, 
                    processor,
                    attack_manager,
                    inputs_list, 
                    original_pixel, 
                    adversial_attack_pixel, 
                    image_slice, 
                    instruction_slice,
                    instruction_loss_slice, 
                    output_text_loss_slice, 
                    image_tokens, 
                    instruction_tokens, 
                    output_tokens):

    losses_history = []
    max_loss, min_loss = -0x7fffffff, 0x7fffffff
    check_eplison = 20

    # Iterate over the specified number of iterations
    dm = tqdm.tqdm(range(args.max_iter))
    for iter in dm:
        # Forward pass through the model
        if args.model_name in ["instruct-blip"]:

            outputs = model(input_ids=inputs_list[0]["input_ids"],
                            attention_mask=inputs_list[0]["attention_mask"],
                            qformer_input_ids = inputs_list[0]["qformer_input_ids"],
                            qformer_attention_mask = inputs_list[0]["qformer_attention_mask"],
                            pixel_values=adversial_attack_pixel,
                            return_dict=True)
            
        elif args.model_name in ["blip2", "llava"]:

            outputs = model(input_ids=inputs_list[0]["input_ids"],
                            attention_mask=inputs_list[0]["attention_mask"],
                            pixel_values=adversial_attack_pixel,
                            return_dict=True)

        # Calculate the overall loss
        if args.benchmark == 'vllm-attack':
            # Calculate the losses
            loss_image = torch.nn.functional.cross_entropy(outputs.logits[0][image_slice, :], image_tokens)
            loss_instruction = torch.nn.functional.cross_entropy(outputs.logits[0][instruction_loss_slice, :], instruction_tokens)
            loss_output = torch.nn.functional.cross_entropy(outputs.logits[0][output_text_loss_slice, :], output_tokens)

            loss = (1 - args.beta) * (args.alpha * loss_image + (1 - args.alpha) * loss_instruction) + args.beta * loss_output

        else:
            raise ValueError("The benchmark {} is not supported".format(args.benchmark))
        
        # Zero the gradients
        model.zero_grad()

        # Calculate the gradients
        loss.backward(retain_graph=True)

        # Update the pixel values
        adversial_attack_pixel.data = adversial_attack_pixel.data - args.lr * adversial_attack_pixel.grad.sign()

        # Clamp the pixel values
        adversial_attack_pixel.data = torch.clamp(adversial_attack_pixel.data, original_pixel - args.epsilon, original_pixel + args.epsilon)

        # Check if attack is successful every check iterations
        if iter % check_eplison == 0:
            if args.max_new_tokens > 32:
                print("Warning: max_new_tokens is more than 32")
                
            if args.only_instruction:
                    input_text = attack_manager.input_info_list[0].instruction
            else:
                input_text = attack_manager.input_info_list[0].process_original_instruction()
            success, generated_text, _ = judge_attack_success(args, model, processor, input_text, adversial_attack_pixel)

            if success:
                print("Attack successful")
                break

        # Update loss history
        losses_history.append({"loss": loss.item()})
        max_loss = max(max_loss, loss.item())
        min_loss = min(min_loss, loss.item())
        dm.set_description("Iter: {}, MaxLoss: {:.3f}, MinLoss: {:.3f}, Loss: {:.3f}".format(iter + 1, max_loss, min_loss, loss.item()))

    print("Input text: \n{}".format(input_text))
    print("Generated text: \n{}".format(generated_text))

    del outputs
    gc.collect()

    return success, adversial_attack_pixel, losses_history


def judge_attack_success(args, model, processor, input_text, adversial_attack_pixel, max_length=50, do_sample=False):
    # Create a ConversationManager object
    conversation_manager = ConversationManager(args.model_name, MODEL_TEMPLATE_AND_PATH)

    # Get the conversation template
    conv, _ = conversation_manager.get_conv_template()

    # Append the input text to the conversation
    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)

    # Convert the adversarial attack pixel to an image
    adversial_attack_image = adversial_attack_pixel[0].detach().cpu().numpy().transpose(1, 2, 0)
    adversial_attack_image = np.clip(adversial_attack_image, 0, 1)
    adversial_attack_image = (adversial_attack_image * 255).astype(np.uint8)
    adversial_attack_image = Image.fromarray(adversial_attack_image)

    # Prepare the inputs for the model
    inputs = processor(text=conv.get_prompt(), images=adversial_attack_image, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = adversial_attack_pixel

    # Generate text using the model
    outputs = model.generate(**inputs, max_length=max_length, do_sample=do_sample, num_return_sequences=1)

    # Decode the generated text
    generated_text = processor.decode(outputs[0], skip_special_tokens=True).strip()

    if args.model_name == "llava":
        pure_generated_text = processor.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
    elif args.model_name in ["blip2", "instruct-blip"]:
        pure_generated_text = generated_text

    # Check if the check keyword is present in the generated text
    for keyword in args.check_keyword:
        if keyword.lower() in pure_generated_text.lower():
            return True, generated_text, pure_generated_text
        
    del outputs
    gc.collect()

    return False, generated_text, pure_generated_text


# Evaluate the attack
def evaluate(args, model, processor, attack_manager, adversial_attack_pixel):
    result = {}
    datasets = []

    # Prepare datasets
    for name in args.eval_dataset:
        if name == "CLS":
            datasets.append((name, CLS_INSTRUCTIONS))
        elif name == "CAP":
            datasets.append((name, CAP_INSTRUCTIONS))
        elif name == "VQA":
            datasets.append((name, VQA_INSTRUCTIONS))
        else:
            raise ValueError("The dataset {} is not supported".format(name))
    
    # Evaluate each dataset
    for instruction_set in datasets:
        meta_result = []
        success_count = 0

        print("Evaluating {}".format(instruction_set[0]))
        dm = tqdm.tqdm(instruction_set[1])

        # Iterate over instructions
        for instruction in dm:
            
            # Add prefix to instruction
            fix_str = attack_manager.input_info_list[0].process_original_instruction()
                
            if args.embel_setting == "prefix":
                instruction = fix_str + ' ' + instruction
            elif args.embel_setting == "suffix":
                instruction = instruction + ' ' + fix_str
            elif args.embel_setting == "mixed":
                instruction_words = instruction.split(' ')
                instruction = ""
                instruction += str(' ' + args.padding_token + ' ').join(instruction_words)
            elif args.embel_setting == "append":
                instruction = attack_manager.input_info_list[0].target_instruction + ' ' + instruction


            # Judge attack success
            success, generated_text, pure_generate_str = judge_attack_success(args, 
                                                                              model, 
                                                                              processor, 
                                                                              instruction, 
                                                                              adversial_attack_pixel, 
                                                                              max_length=50, 
                                                                              do_sample=False)

            if success:
                success_count += 1

            # Append meta result
            meta_result.append({"instruction": instruction, "output": pure_generate_str, "success": success})

            dm.set_description("ASR: {}".format(success_count / len(instruction_set[1])))

        # Store results for the dataset
        result[instruction_set[0]] = {
                "success_count": success_count,
                "total_count": len(instruction_set[1]),
                "ASR": success_count / len(instruction_set[1]),
                "meta": meta_result
        }

        print("ASR for {}: {}".format(instruction_set[0], success_count / len(instruction_set[1])))

    return result

def statsitic(args, result, losses):
    statistic_info = {}

    # Store meta information
    statistic_info["meta"] = args.__dict__

    # Store loss information
    if len(losses) > 0:
        statistic_info["loss"] = {
            "final_loss": losses[-1]["loss"],
            "max_loss": max([loss["loss"] for loss in losses]),
            "min_loss": min([loss["loss"] for loss in losses])
        }

    # Store evaluation results for each dataset
    statistic_info["evaluations"] = {}
    for key, value in result.items():
        statistic_info["evaluations"][key] = {
            "success_count": value["success_count"],
            "total_count": value["total_count"],
            "ASR": value["ASR"]
        }

    # Store overall evaluation results
    statistic_info["evaluations"]["overall"] = {
        "success_count": sum([value["success_count"] for value in result.values()]),
        "total_count": sum([value["total_count"] for value in result.values()]),
        "ASR": sum([value["success_count"] for value in result.values()]) / sum([value["total_count"] for value in result.values()])
    }

    return statistic_info


def finish_log(args, ok, result, losses, adversial_attack_pixel, original_pixel):
    # Generate a unique timestamp for the save directory
    timestamp = str(int(time.time()))

    # Create the save directory path
    save_dir = os.path.join(args.output_dir, args.benchmark, args.model_name, args.target_class, timestamp + '-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the adversarial attack image
    adversial_attack_image = adversial_attack_pixel[0].detach().cpu().numpy().transpose(1, 2, 0)
    adversial_attack_image = np.clip(adversial_attack_image, 0, 1)
    adversial_attack_image = (adversial_attack_image * 255).astype(np.uint8)
    adversial_attack_image = Image.fromarray(adversial_attack_image)
    adversial_attack_image.save(os.path.join(save_dir, "adversial_attack_image.jpg"))

    # Save the original image
    original_image = original_pixel[0].detach().cpu().numpy().transpose(1, 2, 0)
    original_image = np.clip(original_image, 0, 1)
    original_image = (original_image * 255).astype(np.uint8)
    original_image = Image.fromarray(original_image)
    original_image.save(os.path.join(save_dir, "original_image.jpg"))

    # Save the losses
    with open(os.path.join(save_dir, "losses.json"), "w") as f:
        json.dump(losses, f, indent=4, ensure_ascii=False)

    # Save the result
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    # Save the config args
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    # Generate statistic info
    statistic_info = statsitic(args, result, losses)

    statistic_info["passed"] = ok
    # Save the statistic info
    with open(os.path.join(save_dir, "statistic.json"), "w") as f:
        json.dump(statistic_info, f, indent=4, ensure_ascii=False)

    # Print the save directory path
    print("The result is saved in {}".format(save_dir))