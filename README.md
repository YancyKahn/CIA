# vLLM-Attack

This project aims to attack vision-language models, such as blip2, instructBLIP, llava.


# Datasets

Download VisualQA dataset from https://visualqa.org/

```
aria2c -x 5 -c http://images.cocodataset.org/zips/train2014.zip
```

# Run
```
python main.py --device 0 --model_name "blip2" --image_path "./data/visualQA-train-demo/COCO_train2014_000000000009.jpg" --alpha 0.6 --beta 0.6 --benchmark "vllm-attack" --max_iter 1000 --embel_setting "nofix" --target_class "dog" --check_keyword 'dog' --output_texts "dog dog"
```