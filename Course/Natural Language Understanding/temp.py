import timm, tome

# Load a pretrained model, can be any vit / deit model.
model = timm.create_model("vit_base_patch16_224", pretrained=True)
# Patch the model with ToMe.
tome.patch.timm(model)
# Set the number of tokens reduced per layer. See paper for details.
model.r = 16