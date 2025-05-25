#### preprocessor-hint: private-file
"""Manually convert saved pytorch tensor to advrsarial examples."""

import torch
import sys

if len(sys.argv) < 4:
    print(f'Usage: {sys.argv[0]} adv_image_input adv_prediction_input cex_file_output')
    exit()

adv_image_input = sys.argv[1]
adv_prediction_input = sys.argv[2]
cex_file_output = sys.argv[3]

predictions = torch.load(adv_prediction_input).squeeze()
print('adv predictions:')
for i, p in enumerate(predictions):
    print(f'idx={i} prediction={p.item()}')

adv_examples = torch.load(adv_image_input).view(len(predictions), -1)
idx = input("Which index do you want to save?")
adv = adv_examples[int(idx)]
pred = predictions[int(idx)].item()


with open(cex_file_output, "w") as f:
    f.write("sat\n(")
    for i, v in enumerate(adv):
        f.write(f'(X_{i} {v.item() })\n')
    f.write(f"(Y_0 {pred}))\n")

