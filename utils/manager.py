from fastchat.conversation import Conversation, SeparatorStyle


class ConversationManager():

    def __init__(self, model_name, MODEL_TEMPLATE_AND_PATH):
        self.model_name = model_name
        self.MODEL_TEMPLATE_AND_PATH = MODEL_TEMPLATE_AND_PATH

    def build_prompt(self, instruction, response=None):
        conv, _ = self.get_conv_template()
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], response)

        return conv.get_prompt()

    def get_conv_template(self):
        prefix_index = 1    # TODO: Modify this value if the conversation template is changed

        if self.model_name == "llava":

            conversation = Conversation(
                name="llava-chatml",
                system_message="<image>",
                roles=("USER:", "ASSISTANT:"),
                sep_style=SeparatorStyle.CHATML,
                sep="\n")

            conversation.messages = []
            prefix_index = 1
            
        elif self.model_name in ["blip2", "instruct-blip"]:

            conversation = Conversation(
                name="blip2-chatml",
                system_message="",
                roles=("Question", "Answer"),
                sep=" ")

            conversation.messages = []
            prefix_index = 0

        else:
            raise ValueError(f"Model {self.model_name} not supported")

        return conversation, prefix_index

    def get_template_name_and_path(self):
        if self.model_name not in self.MODEL_TEMPLATE_AND_PATH:
            raise ValueError(
                f"Not found model {self.model_name} in MODEL_TEMPLATE_AND_PATH, please add it to config.py")
        model_info = self.MODEL_TEMPLATE_AND_PATH.get(self.model_name)

        return model_info["path"], model_info["template"], model_info["patch_size"], model_info["image_size"]


class InputInfo():

    def __init__(self, tokenizer, image, target_class='dog', original_instruction=None, output_text=None, instruction_length=32, patch_size=24, image_size=576, padding_token='@', support_prefix="The image show"):
        """
        Args:
            image: image
            original_instruction: original instruction, default is "! ! ! ! ! ! ! !"
            target_instruction: target instruction
            output_text: output text
        """
        self.tokenizer = tokenizer
        self.image = image
        self.target_class = target_class
        self.output_text = output_text
        self.instruction = original_instruction
        self.original_instruction = original_instruction
        self.instruction_length = instruction_length
        self.patch_size = patch_size
        self.image_size = image_size
        self.padding_token = padding_token
        self.support_prefix = support_prefix

        self.original_instruction = self.process_original_instruction()
        self.target_instruction = self.process_target_instruction()

    def process_original_instruction(self):
        # padding with padding_token if the length of instruction is less than instruction_length
        if self.original_instruction is not None and len(self.original_instruction) > 0 and self.original_instruction != "":
            text = self.original_instruction
            input_ids = self.tokenizer(text=text, return_tensors="pt")[
                "input_ids"][0]

            if len(input_ids) > self.instruction_length + 1:
                # raise ValueError("Original instruction is too long")
                print("Original instruction is too long, cut it off")
                input_ids = input_ids[:self.instruction_length]

            # padding_token as padding
            nums_to_padding = self.instruction_length - len(input_ids)
            text = text + ' ' + ' '.join([self.padding_token for _ in range(nums_to_padding)])
        else:
            text = ' '.join([self.padding_token for _ in range(self.instruction_length)])

        return text

    def process_target_instruction(self):
        text = "{} {}".format(self.support_prefix, self.target_class)
        input_ids = self.tokenizer(text=text, return_tensors="pt")["input_ids"][0]

        if len(input_ids) > self.instruction_length + 1:
            # raise ValueError("Original instruction is too long")
            print("Original instruction is too long, cut it off")
            input_ids = input_ids[:self.instruction_length]

        target_instruction_tokens = self.tokenizer(text=
            self.target_class, return_tensors="pt")["input_ids"][0]

        # append target class to text
        length_of_target_class = len(target_instruction_tokens) - 1
        nums_to_append = (self.instruction_length -
                          len(input_ids)) // length_of_target_class

        target_instruction = text + ' ' + \
            ' '.join([self.target_class for _ in range(nums_to_append)])

        # '.' as padding
        nums_to_padding = self.instruction_length - \
            len(self.tokenizer(text=target_instruction,
                return_tensors="pt")["input_ids"][0]) + 1
        
        target_instruction = target_instruction + ' ' + \
            ' '.join(["." for _ in range(nums_to_padding)])

        return target_instruction


class AttackManager():

    def __init__(self, model_name, input_info_list, MODEL_TEMPLATE_AND_PATH, only_instruction=False):
        """
        Args:
            model_name (str): model name
            tokenizer: tokenizer
            image: image
            original_instruction: original instruction
            target_instruction: target instruction
            output_text: output text
        """

        self.only_instruction = only_instruction
        self.model_name = model_name
        self.input_info_list = input_info_list
        self.MODEL_TEMPLATE_AND_PATH = MODEL_TEMPLATE_AND_PATH
        self.tokenizer = input_info_list[0].tokenizer
        self.conv_manager = ConversationManager(
            model_name, MODEL_TEMPLATE_AND_PATH)

        self.conv, self.prefix_index = self.conv_manager.get_conv_template()

    def get_original_prompt(self, index=0, response=None):
        conv = self.conv.copy()

        if self.only_instruction:
            conv.append_message(conv.roles[0],  self.input_info_list[index].instruction)
        else:
            conv.append_message(conv.roles[0], self.input_info_list[index].process_original_instruction())
        conv.append_message(conv.roles[1], response)

        return conv.get_prompt()
    
    def get_original_prompt_list(self, response=None):
        return [self.get_original_prompt(index=index, response=response) for index in range(len(self.input_info_list))]

    def get_target_prompt(self, index = 0, response=None):
        conv = self.conv.copy()
        
        if self.only_instruction:
            conv.append_message(conv.roles[0], self.input_info_list[index].instruction)
        else:
            conv.append_message(conv.roles[0], self.input_info_list[index].process_target_instruction())
        conv.append_message(conv.roles[1], response)

        return conv.get_prompt()
    
    def get_target_prompt_list(self, response=None):
        return [self.get_target_prompt(index=index, response=response) for index in range(len(self.input_info_list))]

    
    def get_slice(self, index=0):
        conv, prefix_index = self.conv_manager.get_conv_template()

        if self.model_name in ["llava", "blip2", "instruct-blip"]:
            conv.messages = []

            conv.append_message(conv.roles[0], None)
            toks = self.tokenizer(text=conv.get_prompt()).input_ids
            if self.model_name == "llava":
                toks = toks[0]
            self._user_role_slice = slice(None, len(toks))

            if self.only_instruction:
                conv.update_last_message(self.input_info_list[index].instruction)
            else:
                conv.update_last_message(self.input_info_list[index].original_instruction)
            toks = self.tokenizer(text=conv.get_prompt()).input_ids
            if self.model_name == "llava":
                toks = toks[0]
            self._instruction_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))

            conv.append_message(conv.roles[1], None)
            toks = self.tokenizer(text=conv.get_prompt()).input_ids
            if self.model_name == "llava":
                toks = toks[0]
            self._assistant_role_slice = slice(self._user_role_slice.stop, len(toks))

            conv.update_last_message(self.input_info_list[index].output_text)
            toks = self.tokenizer(text=conv.get_prompt()).input_ids
            if self.model_name == "llava":
                toks = toks[0]
            self._output_text_slice = slice(self._assistant_role_slice.stop, len(toks))

            # With image tokens
            image_size = self.input_info_list[index].image_size
            patch_size = self.input_info_list[index].patch_size



            if self.model_name == "llava":
                self._instruction_loss_slice = slice(self._instruction_slice.start + image_size - 2, self._instruction_slice.stop + image_size - 2)
                self._output_text_loss_slice = slice(self._output_text_slice.start + image_size - 2, self._output_text_slice.stop + image_size - 2)
                self._instruction_attention_slice = slice(self._instruction_slice.start + image_size, self._instruction_slice.stop + image_size)
                self._image_slice = slice(prefix_index, prefix_index + image_size)

            elif self.model_name in ["blip2", "instruct-blip"]:
                self._instruction_loss_slice = slice(self._instruction_slice.start + patch_size - 1, self._instruction_slice.stop + patch_size - 1)
                self._output_text_loss_slice = slice(self._output_text_slice.start + patch_size - 1, self._output_text_slice.stop + patch_size - 1)
                self._instruction_attention_slice = slice(self._instruction_slice.start + image_size, self._instruction_slice.stop + image_size)
                self._image_slice = slice(prefix_index, prefix_index + patch_size)
    
            else:
                raise ValueError(f"Not found model {self.model_name}")

        return self._image_slice, self._instruction_loss_slice, self._output_text_loss_slice, self._instruction_slice, self._output_text_slice, self._instruction_attention_slice

    def get_slice_list(self):
        return [self.get_slice(index=index) for index in range(len(self.input_info_list))]
    
    
    def display_slice(self):
        if self.model_name == "llava":
            support_length = self.input_info.image_size
        elif self.model_name in ["blip2", "instruct-blip"]:
            support_length = self.input_info.patch_size

        print("Original instruction:")
        for i, token in enumerate(self.tokenizer(text=self.get_original_prompt(response=self.input_info.output_text), return_tensors="pt")["input_ids"][0]):
            idx = i
            if i > self.prefix_index:
                i = i + support_length
            print("{}\t{}\t{}\t{}".format(idx, i, token.cpu().detach().numpy(
            ), self.tokenizer.decode(token.item()).replace("\n", "\\n")))

        print("=======" * 10)

        print("Target instruction:")
        for i, token in enumerate(self.tokenizer(text=self.get_target_prompt(response=self.input_info.output_text), return_tensors="pt")["input_ids"][0]):
            idx = i
            if i > self.prefix_index:
                i = i + support_length
            print("{}\t{}\t{}\t{}".format(idx, i, token.cpu().detach().numpy(
            ), self.tokenizer.decode(token.item()).replace("\n", "\\n")))