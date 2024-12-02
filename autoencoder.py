import torch
import torch.nn as nn
from log_utils import get_logger
from feature_transforms import style_transfer
from encoder_decoder_factory import Encoder, Decoder

log = get_logger()


def stylize(level, content, style, encoders, decoders, method, alpha, svd_device, cnn_device, style2=None, K=None):
  log.debug(
    f"Stylization up to ReLU {level} of content sized: {content.size()} and style sized: {style.size()}"
  )

  with torch.no_grad():
    content_features = encoders[level](content).data.to(device=svd_device).squeeze(0)
    style_features = encoders[level](style).data.to(device=svd_device).squeeze(0)
    if style2 is not None:
      style2_features = encoders[level](style2).data.to(device=svd_device).squeeze(0)
    else:
      style2_features = None
    log.debug(
        f"Transfer mode: content features size: {content_features.size()}, style features size: {style_features.size()}"
    )

    transformed_features = style_transfer(method, alpha, content_features, style_features, style2_features, K, level).to(device=cnn_device)
    return decoders[level](transformed_features)


class MultiLevelStyleTransfer(nn.Module):

  def __init__(self, args):
    super().__init__()
    self.svd_device = torch.device("cpu")
    self.cnn_device = args.device
    self.method = args.method
    self.alpha = args.alpha
    self.K = args.K

    self.encoders = [Encoder(level) for level in range(5, 0, -1)]
    self.decoders = [Decoder(level) for level in range(5, 0, -1)]

  def forward(self, content_img, style_img, style2_img):
    for level in range(len(self.encoders)):
      content_img = stylize(
        level,
        content_img,
        style_img,
        self.encoders,
        self.decoders,
        self.method,
        self.alpha,
        self.svd_device,
        self.cnn_device,
        style2 = style2_img,
        K = self.K,
      )
    return content_img