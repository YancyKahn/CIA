import argparse
from utils.attack_tool import set_seed, attack
from config import (MODEL_TEMPLATE_AND_PATH)

from utils.vision_language_models import VLLM
from utils.manager import ConversationManager
from utils.common import calculate_ratio

import warnings
warnings.filterwarnings("ignore")

def main(args):
    set_seed(args.seed)

     # Load the model and tokenizer
    conversation_manager = ConversationManager(args.model_name, MODEL_TEMPLATE_AND_PATH)
    
    path, _, _, _ = conversation_manager.get_template_name_and_path()

    vllm = VLLM(args.model_name, path, device_map=args.device)
    processor, model = vllm.load_model_and_processor()
    
    attack(args, model, processor)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The Parameters of the vLLM attack')

    parser.add_argument('--model_name',
                        type=str,
                        default='llava',
                        help='The model name',
                        choices=['llava', 'blip2', 'instruct-blip'])

    parser.add_argument('--target_class',
                        type=str,
                        default='dog',
                        help='The target class')
    
    parser.add_argument('--check_keyword',
                        nargs = "+",
                        default=['dog'],
                        help='The check keyword')

    parser.add_argument('--instructions',
                        nargs = "+",
                        default=[None, "Is this a dog?", "Describe this image", "Tell me what animal in the image."],
                        help='The instruction')

    parser.add_argument('--output_texts',
                        type=str,
                        default='There\'s only one dog here.',
                        help='The output texts')

    parser.add_argument('--image_path',
                        type=str,
                        default='./data/demo/cat.jpg',
                        help='The image path')

    parser.add_argument('--epsilon',
                        type=calculate_ratio,
                        default=16/255,
                        help='The epsilon')

    parser.add_argument('--lr',
                        type=float,
                        default=0.05,
                        help='The learning rate')

    parser.add_argument('--max_iter',
                        type=int,
                        default=1000,
                        help='The maximum iteration')
    

    parser.add_argument('--alpha',
                        type=float,
                        default=1,
                        help='The alpha')
    
    parser.add_argument('--beta',
                        type=float,
                        default=1,
                        help='The beta')

    parser.add_argument('--instruction_length',
                        type=int,
                        default=8,
                        help='The instruction length')
    
    parser.add_argument('--embel_setting',
                        type=str,
                        default="nofix",
                        choices=["nofix", "prefix", "suffix", "mixed", "append"],
                        help='Use prefix, suffix or not')

    # output_dir
    parser.add_argument('--output_dir',
                        type=str,
                        default='output/test',
                        help='The output directory')
    
    # max new tokens
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=32,
                        help='The max new tokens')
    
    parser.add_argument("--eval_dataset", 
                        default = ["CLS", "CAP", "VQA"],
                        nargs = "+",
                        help = "The evaluation dataset")
    
    parser.add_argument("--padding_token",
                        type=str,
                        default='@',
                        help="The padding token")
    
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="The seed")
    
    parser.add_argument("--device",
                        type=str,
                        default="0",
                        help="cuda device.")
    
    parser.add_argument("--benchmark",
                        type=str,
                        default="vllm-attack",
                        choices=["vllm-attack"],
                        help="benchmark select.")

    parser.add_argument("--only_instruction",
                        type=bool,
                        default=False,
                        help="Only use user instruction.")
    
    parser.add_argument("--support_prefix",
                        type=str,
                        default="The image show",
                        help="support prefix for instruction target.")

    args = parser.parse_args()

    try:
        args.device = int(args.device)
    except:
        pass
    # show args
    print(args)

    main(args)
