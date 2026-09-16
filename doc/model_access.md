Model and Datasets
==================

Pre-trained model weights for all deepcell models are available on
[huggingface](https://huggingface.co/).

To access model weights, you will need to [join the vanvalenlab org][vvhf]
on huggingface.
Navigate to the linked page and click on the "Request to join this org" button
in the upper right, which will require you to log in to your huggingface account.

```{admonition} Pre-trained model license terms
:class: attention

Use of pre-trained model weights is subject to terms of a
[modified Apache 2 license][mesmer-lic] with restrictions on commercial,
non-academic use.

**It is the responsibility of the user to ensure they are in compliance with
the license terms!**

Inquiries regarding commercial licensing should be directed to the
[Caltech Office of Technology Transfer][ottcp].
```

[vvhf]: https://huggingface.co/vanvalenlab
[mesmer-lic]: https://github.com/vanvalenlab/deepcell-auth/blob/main/ASSET_LICENSE
[ottcp]: https://innovation.caltech.edu/about-ottcp/contact-us#send-us-a-message

API Key Usage
-------------

Once you have successfully joined the [vanvalenlab huggingface org][vvhf], you
can [create a huggingface token][hf-token] to access model weights.
Navigate to your huggingface account page and select [Access Tokens][hf-create]
from the sidebar.
Create a new token with **Read** access, and assign it to the name `HF_TOKEN`
in your OS environment:

```bash
export HF_TOKEN="<your-token-here>"
```

[hf-token]: https://huggingface.co/docs/hub/en/security-tokens
[hf-create]: https://huggingface.co/docs/hub/main/en/security-tokens#what-are-user-access-tokens

Models
------

By default, the `Mesmer` instance attempts to download the latest version of
the pre-trained weights upon instantiation.
To see all available model versions, see the [Files and versions page][hf-versions]
on huggingface (note: you must be logged in to see the page).

Specific model versions can be downloaded with the [huggingface CLI][hf-cli].

[hf-versions]: https://huggingface.co/vanvalenlab/torch-mesmer/tree/main
[hf-cli]: https://huggingface.co/docs/huggingface_hub/en/guides/cli
