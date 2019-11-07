# cvtorch: Utilities for pytroch based on opencv

This repository is intended to offer some common augmentataion functions in computer vision task base on **opencv**. The functions' prototype comes from FAIR's [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/transforms/transforms.py). I tried my best to implement these functions strictly follow the details of torchvision and pillow. Any discussions are welcomed.

## Modules
- cvtorch.cvBox.BoxList <br>
This is a handy structure in object detection implemented by FAIR. [original code](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/structures/bounding_box.py)


- **cvtorch.cvTransforms.Compose** <br>
Compose a series of transforms to an opencv image (numpy ndarray in BGR order) and cvBox target

- **cvtorch.cvTransforms.Resize** <br>
Resize an opencv image and target. target is optional.

- **cvtorch.cvTransforms.RandomHorizontalFlip** <br>
random horizontally filp an opencv image and target. target is optional.

- **cvtorch.cvTransforms.RandomVerticalFlip** <br>
random vertically filp an opencv image and target. target is optional.

- **cvtorch.cvTransforms.ColorJitter** <br>
randomly jit brightness, contrast, saturation, hue of an opencv image and targe. target is optional.

- **cvtorch.cvFunctional.ColorJitter** <br>
randomly jit brightness, contrast, saturation, hue of an opencv image.

- **cvtorch.cvTransforms.ToTensor** <br>
convert an opencv image to tensor. (follow torchvision with transpose).

- **cvtorch.cvTransforms.NormalizeAsNumpy** <br>
Normalize an opencv image.

- **cvtorch.cvTransforms.NormalizeAsTorch** <br>
Normalize a tensor image.

Eg
```
from cvtorch import cvTransforms as T

normalize_transform = T.NormalizeAsTorch(mean, std)
color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
transforms = T.Compose(
	[
		color_jitter,
		T.Resize(min_size, max_size),
		T.RandomHorizontalFlip(0.5),
		T.RandomVerticalFlip(0.5),
		T.ToTensor(),
		normalize_transform,
	]
)
```

## Functions
- **cvtorch.cvFunctional.adjust_brightness**
- **cvtorch.cvFunctional.adjust_contrast**
- **cvtorch.cvFunctional.adjust_hue**
- **cvtorch.cvFunctional.adjust_saturation**

## installation
`pip install cvtorch`

