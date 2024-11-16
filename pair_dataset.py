import os
from PIL import Image
from log_utils import get_logger
from torch.utils.data import Dataset
import torchvision.transforms.functional as transforms

log = get_logger()
SUPPORTED_IMG_FORMATS = (".png", ".jpg", ".jpeg")

class ContentStylePairDataset(Dataset):
  def __init__(self, args):
    super(Dataset, self).__init__()

    if args.style.endswith(SUPPORTED_IMG_FORMATS):
      self.pairs_fn = [(args.content, args.style)]
    else:
      self.pairs_fn = [
        (os.path.join(args.content, c), os.path.join(args.style, s))
        for c in os.listdir(args.content)
        for s in os.listdir(args.style)
      ]
      for pair in self.pairs_fn:
        log.info(f"Adding: {pair} to the dataset")

  def __len__(self):
    return len(self.pairs_fn)

  def __getitem__(self, idx):
    pair = self.pairs_fn[idx]

    content = transforms.to_tensor(Image.open(pair[0]).convert("RGB"))
    style = transforms.to_tensor(Image.open(pair[1]).convert("RGB"))

    return {
      "content": content,
      "contentPath": pair[0],
      "style": style,
      "stylePath": pair[1],
    }
