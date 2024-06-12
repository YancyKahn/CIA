import sys
import argparse
import os
import random
import json
from argparse import Namespace

# Importing necessary modules from custom packages
base_work_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(base_work_dir)

from utils.attack_tool import set_seed, ConversationManager
from utils.vision_language_models import VLLM
from config import MODEL_TEMPLATE_AND_PATH
from utils.attack_tool import attack
from utils.common import calculate_ratio


def main(args):
    # Set the random seed
    set_seed(args.seed)

    # Check if the image folder path is provided
    image_folder = args.image_bench_path
    if image_folder is None:
        raise ValueError("image_bench_path is None")
    
    # Create a unique output directory for each run
    work_output_dir = os.path.join(args.output_dir, "visualQA-demo-" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8)))

    # List of argument names for the attack
    attack_args_name = [
        "--model_name",
        "--target_class",
        "--instructions",
        "--output_texts",
        "--image_path",
        "--epsilon",
        "--lr",
        "--max_iter",
        "--alpha",
        "--beta",
        "--instruction_length",
        "--embel_setting",
        "--output_dir",
        "--max_new_tokens",
        "--eval_dataset",
        "--padding_token",
        "--seed",
        "--check_keyword",
        "--device",
        "--benchmark",
        "--only_instruction",
        "--support_prefix"
    ]

    # Create an argument parser for the attack parameters
    attack_args = Namespace()
    # Add the attack arguments to the parser
    for arg_name in attack_args_name:
        name = arg_name[2:]
        attack_args.__setattr__(name, args.__getattribute__(name))

    # Load the target benchmark from a jsonl file
    if args.text_bench_path is not None:
        with open(args.text_bench_path, "r") as f:
            target_bench = [json.loads(line) for line in f]
    else:
        raise ValueError("text_bench_path is None")

    # Load the model and tokenizer
    conversation_manager = ConversationManager(attack_args.model_name, MODEL_TEMPLATE_AND_PATH)
    path, _, _, _ = conversation_manager.get_template_name_and_path()
    vllm = VLLM(attack_args.model_name, path, device_map=attack_args.device)
    processor, model = vllm.load_model_and_processor()

    # Create the output directory if it doesn't exist
    if not os.path.exists(work_output_dir):
        os.makedirs(work_output_dir)

    # Perform the attack for each image in the image folder
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        attack_args.image_path = image_path

        # Perform the attack for each target in the benchmark
        for target in target_bench:
            attack_args.target_class = target["target_class"]
            if args.use_fixed_instruction:
                attack_args.instructions = args.instructions
            else:
                attack_args.instructions = target["instructions"]
            attack_args.output_texts = target["output_texts"]
            attack_args.check_keyword = target["check_keyword"]
            attack_args.output_dir = work_output_dir
            attack_args.support_prefix = target["support_prefix"]

            print(f"Running argument: {attack_args}")
            # Call the attack function
            attack(attack_args, model, processor)


if __name__ == "__main__":
    # Create an argument parser for the main script
    parser = argparse.ArgumentParser(description='The Parameters of the vLLM attack for sigle attack experiments.')

    # Add the main script arguments
    parser.add_argument('--model_name',
                        type=str,
                        default='llava',
                        help='The model name',
                        choices=['llava', 'blip2', 'instruct-blip'])

    parser.add_argument('--epsilon',
                        type=calculate_ratio,
                        default=16/255,
                        help='The epsilon')

    parser.add_argument('--target_class',
                        type=str,
                        default='dog',
                        help='The target class')

    parser.add_argument('--instructions',
                        nargs = "+",
                        default=[None],
                        help='The instruction')

    parser.add_argument('--output_texts',
                        type=str,
                        default='There\'s only one dog here.',
                        help='The output texts')

    parser.add_argument('--lr',
                        type=float,
                        default=0.05,
                        help='The learning rate')

    parser.add_argument('--max_iter',
                        type=int,
                        default=2000,
                        help='The maximum iteration')

    parser.add_argument('--alpha',
                        type=float,
                        default=0.6,
                        help='The alpha')

    parser.add_argument('--beta',
                        type=float,
                        default=0.4,
                        help='The beta')

    parser.add_argument('--instruction_length',
                        type=int,
                        default=8,
                        help='The instruction length')

    parser.add_argument('--image_path',
                        type=str,
                        default='./data/demo/cat.jpg',
                        help='The image path')

    parser.add_argument('--embel_setting',
                        type=str,
                        default="nofix",
                        choices=["nofix", "prefix", "suffix", "mixed", "append"],
                        help='Use prefix, suffix or not')

    parser.add_argument('--output_dir',
                        type=str,
                        default='output/visualQA-demo',
                        help='The output directory')

    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=32,
                        help='The max new tokens')

    parser.add_argument("--eval_dataset",
                        default=["CLS", "CAP", "VQA"],
                        nargs="+",
                        help="The evaluation dataset")

    parser.add_argument("--padding_token",
                        type=str,
                        default='@',
                        help="The padding token")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="The seed")

    parser.add_argument("--text_bench_path",
                        type=str,
                        default="data/text-bench/target.jsonl",
                        help="The text bench")

    parser.add_argument("--image_bench_path",
                        type=str,
                        default="data/visualQA-minimal-demo",
                        help="The image bench")
    
    parser.add_argument('--check_keyword',
                        type=list,
                        default=['dog'],
                        help='The check keyword')
    
    parser.add_argument("--device",
                        type=str,
                        default=0,
                        help="cuda device.")

    parser.add_argument("--benchmark",
                    type=str,
                    default="vllm-attack",
                    choices=["single-P", "multi-P", "CroPA", "vllm-attack", "vllm-attack-image", "vllm-attack-text"],
                    help="benchmark select.")

    parser.add_argument("--only_instruction",
                    type=bool,
                    default=False,
                    help="Only use user instruction.")
    
    parser.add_argument("--use_fixed_instruction",
                        type=bool,
                        default=False,
                        help="Use fixed instruction.")
    
    parser.add_argument("--support_prefix",
                        type=str,
                        default="The image show",
                        help="support prefix for instruction target.")
    # Parse the main script arguments
    args = parser.parse_args()

    try:
        args.device = int(args.device)
    except:
        pass

    print(args)

    print(os.path.abspath(args.text_bench_path))
    print(os.path.abspath(args.image_bench_path))
    print(os.path.abspath(args.output_dir))
    # Call the main function
    main(args)
