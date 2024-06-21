# [Enhancing Cross-Prompt Transferability in Vision-Language Models through Contextual Injection of Target Tokens](https://arxiv.org/abs/2406.13294)

---
This project aims to attack vision-language models, such as blip2, instructBLIP, llava.

<div align="center">

![](https://camo.githubusercontent.com/83d3746e5881c1867665223424263d8e604df233d0a11aae0813e0414d433943/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)

</div align="center">

Vision-language models (VLMs) seamlessly integrate visual and textual data to perform tasks such as image classification, caption generation, and visual question answering. However, adversarial images often struggle to deceive all prompts effectively in the context of cross-prompt migration attacks, as the probability distribution of the tokens in these images tends to favor the semantics of the original image rather than the target tokens. To address this challenge, we propose a Contextual-Injection Attack (CIA) that employs gradient-based perturbation to inject target tokens into both visual and textual contexts, thereby improving the probability distribution of the target tokens. By shifting the contextual semantics towards the target tokens instead of the original image semantics, CIA enhances the cross-prompt transferability of adversarial images. 

![Overall Stucture](assets/framework.png)

## Dependencies

You can create and activate the same running environment and install dependencies as us by using the following commands:

```
conda env create -f environment.yaml
conda activate vllm-attack
```

## Datasets

Download VisualQA dataset from https://visualqa.org/

```
aria2c -x 5 -c http://images.cocodataset.org/zips/train2014.zip
```

## Run

You can run some examples using the following commands:

```
python main.py --device 0 --model_name "blip2" --image_path "./data/visualQA-train-demo/COCO_train2014_000000000009.jpg" --alpha 0.6 --beta 0.6 --benchmark "vllm-attack" --max_iter 1000 --embel_setting "nofix" --target_class "dog" --check_keyword 'dog' --output_texts "dog dog"
```

### Projects

This project has been modified from the following projects:

[CroPA](https://github.com/Haochen-Luo/CroPA) provide the baseline of this work.


## Citation

@misc{yang2024enhancing,
      title={Enhancing Cross-Prompt Transferability in Vision-Language Models through Contextual Injection of Target Tokens}, 
      author={Xikang Yang and Xuehai Tang and Fuqing Zhu and Jizhong Han and Songlin Hu},
      year={2024},
      eprint={2406.13294},
      archivePrefix={arXiv},
      primaryClass={id='cs.MM' full_name='Multimedia' is_active=True alt_name=None in_archive='cs' is_general=False description='Roughly includes material in ACM Subject Class H.5.1.'}
}

## License
This codebase is released under [MIT License](LICENSE).
