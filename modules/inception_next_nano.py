from timm.models.inception_next import _create_inception_next
from timm.models.inception_next import InceptionDWConv2d
from timm.models._registry import register_model

@register_model
def inception_next_nano(pretrained=False, **kwargs):
    print("register inception_next_nano")
    model_args = dict(
        depths=(2, 2, 8, 2), dims=(80, 160, 320, 640),
        token_mixers=InceptionDWConv2d,
    )
    return _create_inception_next('inception_next_nano', pretrained=False, **dict(model_args, **kwargs))
