<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill RemoteCLIP Module

This repository contains the code supporting the RemoteCLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) is a vision-language CLIP model trained on remote sensing data. According to the RemoteCLIP README:

> RemoteCLIP outperforms previous SoTA by 9.14% mean recall on the RSICD dataset and by 8.92% on RSICD dataset. For zero-shot classification, our RemoteCLIP outperforms the CLIP baseline by up to 6.39% average accuracy on 12 downstream datasets.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [RemoteCLIP Autodistill documentation](https://autodistill.github.io/autodistill/base_models/remoteclip/).

## Installation

To use RemoteCLIP with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-remote-clip
```

## Quickstart

```python
from autodistill_remote_clip import RemoteCLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our RemoteCLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = RemoteCLIP(
    ontology=CaptionOntology(
        {
            "airport runway": "runway",
            "countryside": "countryside",
        }
    )
)

predictions = base_model.predict("runway.jpg")

print(predictions)
```

## License

This Autodistill module is licensed under an MIT license. At the time of publishing this project, the RemoteCLIP model and weights had no attached license. Refer to the [RemoteCLIP repository](https://github.com/ChenDelong1999/RemoteCLIP) for the most up-to-date licensing information regarding the model.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!