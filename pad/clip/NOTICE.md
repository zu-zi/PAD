# pad/clip

The files in this folder are adapted from the official OpenAI CLIP
release (https://github.com/openai/CLIP, MIT licensed).  The only
modification is in `model.py::VisionTransformer.forward`, which
optionally injects PAD's `VAPromptPool` tokens into each transformer
block.
