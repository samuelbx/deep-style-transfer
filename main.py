import argparse
import os
from timeit import default_timer as timer
import torch
import torchvision
from log_utils import get_logger
from torch.utils.data import DataLoader

import autoencoder
from pair_dataset import ContentStylePairDataset

log = get_logger()

def parse_args():
  parser = argparse.ArgumentParser(description="Style transfer using VGG19 and Wasserstein-based feature transforms")
  parser.add_argument("content", type=str, help="Path of the content image (must be .jpg or .png format)")
  parser.add_argument("style", type=str, help="Path of the style image (must be .jpg or .png format)")
  parser.add_argument("method", choices=["wct", "gaussian", "gmmot-bary", "gmmot-rand"], help="Feature transform type to use for style transfer")
  parser.add_argument("--out", type=str, default="outputs", help="Directory where stylized results will be saved (default: ./outputs/)")
  parser.add_argument("--alpha", type=float, default=0.2, help="Balance between the original content and stylized features (0 to 1, default: 0.2)")
  parser.add_argument("--K", type=int, default=2, help="Number of Gaussian components for GMM-OT (default: 2)")
  return parser.parse_args()

def save_image(img, content_name, style_name, out_ext, args):
  alpha_str = f"{int(args.alpha * 100)}"
  filename = f"{args.method}_{content_name}_{style_name}_alpha{alpha_str}_K{args.K}.{out_ext}"
  output_path = os.path.join(args.out, filename)
  torchvision.utils.save_image(
    img.cpu().detach().squeeze(0),
    output_path
  )

def main():
  args = parse_args()
  os.makedirs(args.out, exist_ok=True)

  #if not args.no_cuda and torch.cuda.is_available():
  #    device = 'cuda:0'
  #else:
  device = 'cpu'
  log.info(f'device: {device}')
  args.device = torch.device(device)

  dataset = ContentStylePairDataset(args)
  loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

  log.info('Using multi-level stylization pipeline')
  model = autoencoder.MultiLevelStyleTransfer(args)
  model.to(device=args.device)
  model.eval()

  for i, sample in enumerate(loader):
    try:
      log.info(f"Starting {i}/{len(loader)} stylization iteration")
      log.info(f"content: {sample['contentPath']}\tstyle: {sample['stylePath']}")

      s_basename, _ = os.path.splitext(os.path.basename(sample['stylePath'][0]))
      s_basename = s_basename.strip(".")
      c_basename, c_ext = os.path.splitext(os.path.basename(sample['contentPath'][0]))
      c_basename = c_basename.strip(".")

      content = sample['content'].to(device=args.device)
      style = sample['style'].to(device=args.device)

      start = timer()
      out = model(content, style)
      end = timer()

      log.info(f"Wall-clock time for stylization: {end - start:.2f}s")
      save_image(out, c_basename, s_basename, c_ext.strip("."), args)
    except:
      log.info(f"Exception caught.")

  log.info("Stylization completed, exiting.")

if __name__ == "__main__":
    main()
